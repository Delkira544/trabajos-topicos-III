# Análisis de Variantes del Algoritmo Genético

## 1. Sequential (CPU, un solo hilo)

**Archivo:** `include/ga/solvers/Sequential.hpp`

Línea base. Un solo hilo en CPU ejecuta todo secuencialmente.

### Flujo de `do_reproduction()` (línea 24)

1. Ordena la población por fitness (`std::sort`)
2. Loop hasta llenar `population_size`:
   - **Selección por torneo**: escoge 2 padres (`selection_op->select_pair`, línea 39)
   - **Cruzamiento de 1 punto** (`crossover_op->apply`, línea 42)
   - **Mutación bit-flip uniforme** (`mutation_op->apply`, línea 45)
   - **Reparación condicional**: solo si `validator->is_feasible()` es `false` (línea 48)
3. Aplica elitismo (5% superior pasa directo)
4. Reemplaza población

### Reparación CPU (`KnapsackValidator::repair`)

Tiene **4 fases** completas:
1. Resuelve incompatibilidades (elimina el de menor eficiencia)
2. Resuelve dependencias (elimina el dependiente)
3. Resuelve excesos de capacidad (peso/volumen)
4. **Resuelve categorías** — elimina ítems de menor eficiencia hasta cumplir el máximo por categoría

---

## 2. CUDABasic (GPU, memoria global)

**Archivos:**
- `include/ga/solvers/CUDABasic.cuh` — declaración de clase
- `src/ga/solvers/CUDABasic.cu` — implementación
- `src/ga/cuda/fitness_kernel.cu` — kernel de evaluación
- `src/ga/cuda/operators_kernel.cu` — kernel de reproducción

### Paradigma

**1 hilo GPU por individuo.** Todos los datos residen en **memoria global**. Las transferencias CPU↔GPU se miden con `cudaEvent_t`.

### Inicialización

`flatten_instance()` (línea 78 de `CUDABasic.cu`): convierte las estructuras C++ (`Item`, reglas de incompatibilidad, dependencias, categorías) en **arrays planos 1D** (`float[]`, `int[]`) para la GPU.

`upload_instance()` (línea 156): transfiere estos arrays a GPU con `cudaMalloc` + `cudaMemcpy` **una sola vez** al inicio. También inicializa estados cuRAND con `init_rng_kernel`.

### Evaluación (`evaluate_population`, línea 240)

```cuda
// fitness_kernel — cada hilo procesa UN individuo
int ind = blockIdx.x * blockDim.x + threadIdx.x;
const uint8_t* chrom = genes + (long long)ind * n_items;

for (int g = 0; g < n_items; ++g) {
    if (chrom[g]) {
        total_value  += values[g];   // ← memoria global
        total_weight += weights[g];  // ← memoria global
        total_volume += volumes[g];  // ← memoria global
    }
}
// Calcula violaciones: categorías, incompatibilidades, dependencias
// Fitness = obj_w * (valor/max_value) - pen_w * violación_normalizada
```

- Formato `genes[ind * n_items + gene]` → accesos **coalescentes**
- Recibe **12 punteros** a memoria global como parámetros del kernel
- Transfiere a CPU solo 4 arrays escalares: `fitness`, `penalty`, `hard_feas`, `is_valid` (~16 bytes/individuo)
- Descarga el cromosoma del **mejor individuo** solo (unos pocos KB), no la población completa

### Reproducción (`do_reproduction`, línea 325)

```cuda
// reproduce_kernel — cada hilo produce UN hijo
int ind = blockIdx.x * blockDim.x + threadIdx.x;
curandState local_state = rng_states[ind];

// 1. Torneo (tamaño 3) para padre 1 y padre 2
int best1 = curand_uniform(&local_state) * pop_size;
for (int t = 1; t < 3; ++t) {
    int cand = curand_uniform(&local_state) * pop_size;
    if (fitness[cand] > fitness[best1]) best1 = cand;
}
// (ídem para best2)

// 2. Cruzamiento de 1 punto (tasa 0.7)
if (r_cross <= 0.7) {
    int cut = curand_uniform(&local_state) * (n_items - 1);
    for (int g = 0; g < n_items; ++g)
        ch[g] = (g <= cut) ? p1[g] : p2[g];
}

// 3. Mutación bit-flip (tasa 0.04 por gen)
for (int g = 0; g < n_items; ++g)
    ch[g] ^= (r_mut < 0.04) ? 1 : 0;

// 4. Reparación POST-mutación (SIEMPRE, no condicional)
repair_chromosome_gpu(ch, ...);
```

### Reparación GPU (`repair_chromosome_gpu`, línea 24 de `operators_kernel.cu`)

```cuda
__device__ void repair_chromosome_gpu(...) {
    bool changed = true;
    int iter = 0;
    while (changed && iter < 100) {  // ← LOOP ITERATIVO
        // Fase 1: Incompatibilidades
        // Fase 2: Dependencias
        // Fase 3: Capacidades (peso/volumen)
        // NO tiene Fase 4 de categorías ← DIFERENCIA CLAVE con CPU
    }
}
```

**Características críticas:**
- Se aplica **siempre** a toda la descendencia (no condicional como en CPU)
- El loop `while` puede requerir de 1 a 100 iteraciones → **divergencia severa de warps**
- NO repara violaciones de categoría (solo se penalizan en fitness)
- Esto explica que el % de individuos factibles sea menor en GPU (5-23%) vs CPU (100%)

### Post-reproducción

1. **Elitismo**: copia los mejores individuos al offspring con `cudaMemcpy DeviceToDevice`
2. **Swap**: `swap_populations_kernel` copia offspring → population (todo en GPU)
3. No se descargan cromosomas a CPU cada generación (solo al final con `get_best()` lazy)

### Diagrama de flujo por generación

```
CPU                          GPU
│                            │
├─ inicia generación         │
│                            │
├─ evaluate_population() ───→│ fitness_kernel (1 hilo/ind)
│                            │   └─ lee genes + calcula fitness
│←─── escalares (fit, pen, ──┤
│     hard_feas, is_valid)   │
│                            │
├─ actualiza mejores en CPU  │
│                            │
├─ do_reproduction() ───────→│ reproduce_kernel (1 hilo/hijo)
│                            │   ├─ torneo + crossover + mutación
│                            │   └─ repair_chromosome (loop while)
│                            │
│                            │ elitismo D→D
│                            │ swap D→D (offspring → population)
│                            │
├─ fin generación ───────────│
```

---

## 3. CUDAOptimized (GPU optimizado)

**Archivos:**
- `include/ga/solvers/CUDAOptimized.cuh` — declaración (hereda de `CUDABasic`)
- `src/ga/solvers/CUDAOptimized.cu` — implementación de overrides

### Optimizaciones implementadas (7 en total)

#### 1. Memoria constante para parámetros

**Archivo:** `fitness_kernel.cuh` (líneas 15-18, 37)

```cuda
__constant__ float c_values[MAX_CONST_ITEMS];   // 3000 items max
__constant__ float c_weights[MAX_CONST_ITEMS];
__constant__ float c_volumes[MAX_CONST_ITEMS];
__constant__ int   c_cat_ids[MAX_CONST_ITEMS];
__constant__ FitnessParams c_params;
```

**Implementación:** `upload_instance_optimized()` (`CUDAOptimized.cu` línea 151)
```cuda
cudaMemcpyToSymbol(c_values, h_item_values.data(), n_items * sizeof(float));
cudaMemcpyToSymbol(c_params, &p, sizeof(FitnessParams));
```

**Condición:** solo cuando `n_items <= 3000` (límite de 64 KB de memoria constante).
**Problema:** la instancia `large` tiene 10000 ítems → cae a memoria global.

#### 2. Accesos coalescentes

Formato `genes[ind * n_items + gene]`. Ya presente en `CUDABasic`. Garantiza que hilos consecutivos en un warp lean genes consecutivos → una sola transacción de 128 bytes por grupo de 32 hilos.

#### 3. Evaluación intra-bloque con shared memory (`fitness_kernel_opt`)

**Archivo:** `fitness_kernel.cu` (líneas 111-193)

```cuda
__global__ void fitness_kernel_opt(...) {
    int ind = blockIdx.x;  // ← UN BLOQUE por individuo (NO 1 hilo)
    extern __shared__ float smem[];

    // Cada hilo del bloque acumula un subconjunto de genes
    for (int g = tid; g < n_items; g += blockDim.x) {
        if (chrom[g]) {
            lv  += c_values[g];   // ← desde memoria CONSTANTE
            lw  += c_weights[g];
            lvo += c_volumes[g];
        }
    }
    // Reducción paralela en shared memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) { s_value[tid] += s_value[tid+stride]; }
        __syncthreads();
    }
    // Solo hilo 0 calcula fitness final
    if (tid == 0) {
        // Solo valida peso y volumen (ignora categorías, incomp, dep)
        is_valid[ind] = hard_feas[ind];  // ← INCOMPLETO
    }
}
```

**Condición:** `shared_reduce=true` AND `n_items <= 3000` AND `const_memory=true` AND **sin restricciones complejas**.
**Problema:** la instancia `large` tiene 10000 ítems Y tiene incomp/dep/categorías → **nunca se activa**.

#### 4. Bloque óptimo dinámico

**Archivo:** `CUDAOptimized.cu` (línea 385)

```cuda
cudaOccupancyMaxPotentialBlockSize(&min_grid, &opt_bs, reproduce_kernel, 0, 0);
opt_bs = ((opt_bs + 31) / 32) * 32;  // alineado a warp
```

Elige el block_size que maximiza ocupancia de SMs. Fallback a 128 si falla.

#### 5. Memoria constante para arrays de ítems (ídem Opt 1)

Los arrays `c_values`, `c_weights`, `c_volumes`, `c_cat_ids` permiten broadcast desde caché constante L1, evitando lecturas repetidas de memoria global.

#### 6. Control de divergencia de warps

Principio de diseño: minimizar ramas dependientes del thread ID. En `reproduce_kernel` todos los hilos siguen el mismo código (torneo → crossover → mutación → reparación) sin bifurcaciones por `if (threadIdx.x ...)`. **No resuelve la divergencia real** que ocurre dentro del loop `while` de `repair_chromosome_gpu`, donde cada hilo itera distinto número de veces.

#### 7. Streams CUDA

**Archivo:** `CUDAOptimized.cu` (líneas 104-108, 215, 313)

```cuda
cudaStreamCreate(&stream_eval);   // fitness_kernel_opt
cudaStreamCreate(&stream_repro);  // reproduce_kernel
```

La memoria host se registra como **pinned** (`cudaHostRegister`) para permitir `cudaMemcpyAsync` verdadero.
**Problema:** el kernel de reproducción domina ~99% del tiempo de GPU; no hay tareas de peso comparable para solapar.

---

## Comparación lado a lado

| Aspecto | Sequential | CUDABasic | CUDAOptimized |
|---|---|---|---|
| **Ejecución** | 1 hilo CPU | N hilos GPU | N hilos GPU |
| **Memoria ítems** | RAM (structs) | Global (d_values, d_weights...) | `__constant__` si n≤3000, sino global |
| **Evaluación** | `KnapsackFitness::evaluate` | `fitness_kernel` (1 hilo/ind) | `fitness_kernel_opt` (1 bloque/ind) si aplica |
| **Reparación** | Condicional (solo si infactible) | **Siempre** (incondicional) | **Siempre** (incondicional) |
| **Fase categorías** | Sí (Fase 4) | **No** | **No** |
| **% factibles** | 100% (o 0% en datos de Daniel) | 5-23% | 5-23% |
| **Block size** | N/A | Fijo (128 por defecto) | Dinámico (`cudaOccupancyMax...`) |
| **Streams** | N/A | No (default stream 0) | 2 streams + pinned memory |
| **Transferencias D→H** | N/A | Solo escalares + cromosoma del mejor (lazy) | Ídem |

---

## Por qué CUDAOptimized fue más lento que CUDABasic

| Optimización | ¿Ayudó? | Explicación |
|---|---|---|
| **Memoria constante** | Sí, pero no aplica a large | large tiene 10000 ítems > límite de 3000 |
| **Shared memory reduction** | **No** | Overhead de `__syncthreads()` + conflictos de banco superan ganancia con n=10000 |
| **Streams** | **No** | Reprod domina 99% del tiempo — no hay nada con qué solapar |
| **Bloque óptimo** | Marginal | Diferencia entre bs=32 y bs=256 es solo ±10% |
| **Warp divergence control** | **No mitiga el problema real** | La divergencia severa está en `repair_chromosome_gpu` (loop `while`), no en selección/mutación |

### El cuello de botella real

El kernel de **reproducción** consume ~99.3% del tiempo total de GPU (`reproduce_kernel` + `repair_chromosome_gpu`). Dentro de este, el loop `while` iterativo (hasta 100 iteraciones) en la reparación causa:

- Hilos del mismo warp terminan en momentos distintos
- Los que terminan antes quedan **inactivos** esperando a los demás
- Ocupancia efectiva de SM baja drásticamente

Ninguna de las 7 optimizaciones de `CUDAOptimized` aborda este problema. La optimización que realmente ayudaría sería reescribir la reparación usando **reducciones warp-level** (prefijos de suma) para eliminar el loop iterativo.

### Instancias donde CUDAOptimized SÍ funciona (en teoría)

Para instancias **small** y **medium** (n ≤ 3000) donde:
- Aplica memoria constante (broadcast desde caché L1)
- Aplica shared memory reduction (sin restricciones complejas)

...el overhead de streams y el bloque óptimo dinámico sigue siendo neutro o negativo porque el kernel de reproducción sigue dominando el tiempo.

---

## Referencias a archivos

| Archivo | Contenido |
|---|---|
| `include/ga/solvers/Sequential.hpp` | Implementación completa de `Sequential` |
| `include/ga/solvers/CUDABasic.cuh` | Declaración de `CUDABasic` (punteros device, métricas) |
| `src/ga/solvers/CUDABasic.cu` | Implementación: `flatten_instance`, `upload_instance`, `evaluate_population`, `do_reproduction`, `get_best` |
| `include/ga/solvers/CUDAOptimized.cuh` | Declaración de `CUDAOptimized` (streams, flags, métodos override) |
| `src/ga/solvers/CUDAOptimized.cu` | Implementación: `upload_instance_optimized`, `evaluate_population` (opt), `do_reproduction` (opt), `get_optimal_block_size` |
| `src/ga/cuda/fitness_kernel.cuh` | Declaración de `fitness_kernel`, `fitness_kernel_opt`, `__constant__` symbols, `FitnessParams` |
| `src/ga/cuda/fitness_kernel.cu` | Implementación de ambos kernels de fitness |
| `src/ga/cuda/operators_kernel.cuh` | Declaración de `init_rng_kernel`, `reproduce_kernel`, `swap_populations_kernel` |
| `src/ga/cuda/operators_kernel.cu` | Implementación: `repair_chromosome_gpu` (device), `reproduce_kernel`, `swap_populations_kernel` |
| `src/ga/cuda/reduction_kernel.cuh` | Declaración de `reduce_best_block`, `reduce_best_warp` |
| `src/ga/cuda/reduction_kernel.cu` | Implementación de kernels de reducción |
