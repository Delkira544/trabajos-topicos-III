# Guía para la Presentación — Actividad 3

## Estructura Recomendada de la PPTX (10-15 diapositivas)

---

## Diapositiva 1 — Portada
- Título: "Optimización paralela en GPU del problema extendido de la mochila mediante CUDA"
- Integrantes: Daniel Burgos, Christian verd
- Fecha de entrega
- Asignatura: Tópicos III

---

## Diapositiva 2 — Introducción al Problema
**El problema de la mochila 0/1 clásico**
- Dados $n$ ítems con valor $v_i$ y peso $w_i$, seleccionar un subconjunto que maximice el valor total sin exceder la capacidad $W$.
- Formulación:
  ```
  Maximizar Z = Σ vᵢ·xᵢ
  sujeto a  Σ wᵢ·xᵢ ≤ W,  xᵢ ∈ {0,1}
  ```
- Es NP-difícil: no existe algoritmo polinomial conocido.

**Versión extendida (la que implementamos)**
Cada ítem tiene: ID, valor, peso, **volumen**, **categoría**.
Restricciones adicionales:
1. Peso máximo $W$
2. Volumen máximo $V$
3. Categorías (mín/máx ítems por categoría)
4. Incompatibilidades (pares que no pueden coexistir)
5. Dependencias (si A entonces B)

### Para decir en la exposición
> "Partimos de la mochila 0/1 clásica y la extendemos con 5 restricciones. Esto hace que el problema sea más realista pero también más complejo de resolver, justificando el uso de un algoritmo genético paralelizado en GPU."

---

## Diapositiva 3 — Algoritmo Genético: Visión General
Diagrama de flujo:

```
INICIO → Inicializar población aleatoria
              ↓
         Evaluar fitness (paralelizado en GPU)
              ↓
         ¿Criterio de término? → SÍ → FIN (reportar mejor)
              | NO
         Selección por torneo (paralelizado en GPU)
              ↓
         Cruzamiento de 1 punto (paralelizado en GPU)
              ↓
         Mutación uniforme (paralelizado en GPU)
              ↓
         Reparación heurística (paralelizado en GPU)
              ↓
         Elitismo (5% mejores pasan directo)
              ↓
         Reemplazar población ←── (vuelve a Evaluar)
```

### Parámetros:
| Parámetro | Valor |
|-----------|-------|
| Población | 512, 1024, 4096 |
| Generaciones | 300 |
| Cruzamiento | 0.7 (tasa) |
| Mutación | 0.04 (por gen) |
| Torneo | 3 participantes |
| Elitismo | 5% de la población |

### Para decir en la exposición
> "El corazón del algoritmo es un bucle de 300 generaciones. En cada una evaluamos fitness, seleccionamos padres, cruzamos, mutamos y reparamos. Todo esto está paralelizado en GPU. Solo el elitismo y el control del bucle principal están en CPU."

---

## Diapositiva 4 — Representación en Memoria (CPU y GPU)
**Cromosoma**: vector binario de largo $n$ (ítems).
```
Ejemplo: X = [1, 0, 1, 1, 0, 0, 1, 0]
          ítems: 0, 2, 3 y 6 están seleccionados
```

**Población en GPU**: arreglo plano 1D de tipo `uint8_t[pop_size × n_items]`
```
genes[ind × n_items + gene]
            ↑
    Cada fila = un individuo
```
Esto da **accesos coalescentes**: hilos consecutivos leen genes consecutivos.

**Memoria consumida** (ej: large + pop 4096):
- Población: 4096 × 10000 = 39.1 MB
- Offspring: otros 39.1 MB
- Total ≈ 78 MB (cómodo dentro de 8 GB VRAM)

### Para decir en la exposición
> "Usamos uint8_t en vez de bool porque CUDA no tiene un tipo bool de 1 byte garantizado. La indexación lineal con stride permite que hilos vecinos accedan a memoria vecina, maximizando el ancho de banda."

---

## Diapositiva 5 — Función de Aptitud (Fitness)
**Ecuación:**
```
fitness(X) = 0.4 × (valor_total / valor_máximo) - 0.7 × violación_total
```

**Violación total normalizada:**
```
viol = α·min(1, ExcesoPeso/W) + β·min(1, ExcesoVolumen/V)
     + γ·(ErrCategoría / N_cat) + δ·(ErrIncomp / N_incomp) + ε·(ErrDep / N_dep)
```

**Pesos calibrados** (con penalty tuner):
| α (peso) | β (volumen) | γ (categoría) | δ (incomp.) | ε (dep.) |
|----------|------------|--------------|------------|---------|
| 0.25 | 0.00 | 0.25 | 0.25 | 0.25 |

**Restricciones duras** (solución final debe cumplirlas): peso, volumen, incompatibilidades, dependencias.
**Restricciones blandas** (solo se penalizan): categorías.

### Kernel `fitness_kernel` (un hilo por individuo):
```cuda
// Pseudocódigo del kernel
int ind = blockIdx.x * blockDim.x + threadIdx.x;
if (ind >= pop_size) return;

// Cargar cromosoma del individuo
uint8_t* genes = d_population + ind * n_items;

float total_value = 0, total_weight = 0, total_volume = 0;
int cat_counts[64] = {0};

for (int g = 0; g < n_items; g++) {
    if (genes[g]) {
        total_value  += d_values[g];
        total_weight += d_weights[g];
        total_volume += d_volumes[g];
        if (d_cat_ids[g] >= 0) cat_counts[d_cat_ids[g]]++;
    }
}

// Calcular violaciones y fitness (idem CPU)...
// Escribir fitness[ind], penalty[ind], hard_feas[ind], is_valid[ind]
```

### Para decir en la exposición
> "Cada hilo procesa un individuo completo. Recorre sus n genes en un loop, acumulando valor, peso y volumen. Luego cuenta violaciones de categorías, incompatibilidades y dependencias. Al final aplica la fórmula de fitness con los pesos calibrados."

---

## Diapositiva 6 — Kernel de Reproducción (el más importante)
**Kernel fusionado**: selección + cruzamiento + mutación + reparación en un solo kernel.

### Pseudocódigo:
```cuda
int ind = blockIdx.x * blockDim.x + threadIdx.x;
curandState local_state = rng_states[ind];

// 1. Selección por torneo (tamaño 3)
int p1 = torneo(fitness, pop_size, local_state);
int p2 = torneo(fitness, pop_size, local_state);

// 2. Cruzamiento de 1 punto
float r = curand_uniform(&local_state);
int cut = (r < crossover_rate) ? curand(&local_state) % n_items : n_items;

for (int g = 0; g < cut; g++)
    offspring[ind * n_items + g] = population[p1 * n_items + g];
for (int g = cut; g < n_items; g++)
    offspring[ind * n_items + g] = population[p2 * n_items + g];

// 3. Mutación uniforme (bit-flip con prob mutation_rate)
for (int g = 0; g < n_items; g++)
    if (curand_uniform(&local_state) < mutation_rate)
        offspring[ind * n_items + g] ^= 1;

// 4. Reparación heurística
repair_chromosome_gpu(offspring + ind * n_items, ...);

rng_states[ind] = local_state;
```

### Reparación (`repair_chromosome_gpu`):
Loop de hasta 100 iteraciones:
1. Si hay par incompatible → eliminar el de menor eficiencia
2. Si hay dependencia incumplida → eliminar el dependiente
3. Si excede peso o volumen → eliminar el de peor relación valor/(peso+volumen)
4. Si todo OK → break

### Para decir en la exposición
> "Este es el kernel que más tiempo consume (~99% del tiempo GPU). La razón es la reparación: como cada individuo necesita un número distinto de iteraciones para volverse factible, los hilos terminan en momentos distintos, causando divergencia de warps. Los hilos más rápidos esperan inactivos a los más lentos."

---

## Diapositiva 7 — Flujo de Datos CPU ↔ GPU
**Diagrama de transferencias:**

```
┌─────────────────────────────┐
│         HOST (CPU)          │
│                             │
│  SolverFactory::create()    │
│       ↓                     │
│  upload_instance() ──────── H2D ──► d_values, d_weights, ...
│       ↓                     │
│  initialize_population()    │
│    crea en CPU              │
│       ↓                     │
│    cudaMemcpy ──────────── H2D ──► d_population (1 vez)
│       ↓                     │
│  LOOP 300 generaciones      │
│    reproduce_kernel (GPU)   │
│    fitness_kernel (GPU)     │
│       ↓                     │
│    cudaMemcpy ──────────── D2H ──► d_fitness, d_penalty,
│                                      d_hard_feas, d_is_valid
│                                      (~10 bytes/individuo)
│       ↓                     │
│    update_best_solution()   │
│    (CPU-side)               │
│       ↓                     │
│  get_best() ── lazy D2H ──► h_best_chrom (n_items bytes)
└─────────────────────────────┘
```

**Optimización clave (Req 6.2):**
- ❌ **Antes**: población completa D→H cada generación (ej: 4096×10000 = 39 MB por gen × 300 = ~12 GB)
- ✅ **Ahora**: solo escalares (~10 bytes/ind) cada generación + cromosoma del mejor al final (~n_items bytes)

### Para decir en la exposición
> "La población se mantiene siempre en GPU. Solo transferimos los fitness y flags de validez cada generación. El cromosoma del mejor individuo se descarga una sola vez al final. Esto reduce la transferencia acumulada de ~12 GB a ~6 MB, haciendo viable la paralelización."

---

## Diapositiva 8 — Las 3 Variantes Implementadas

| Variante | Descripción |
|----------|-------------|
| **Sequential** | CPU single-thread. Línea base para speed-up. |
| **CUDA Basic** | Kerneles en GPU con memoria global. 1 hilo por individuo. |
| **CUDA Optimized** | Memoria constante, shared memory, streams, bloque dinámico. |

### Arquitectura de clases:
```
GeneticSolver (abstracto)
  └── BaseGA (template method → run() es FINAL)
        ├── Sequential (CPU)
        ├── CUDABasic (GPU básico)
        │     └── CUDAOptimized (GPU optimizado)
```

- `run()` es el método plantilla: `initialize → evaluate → reproduce → evaluate → ...`
- `do_reproduction()` es virtual puro: cada variante implementa su propia reproducción
- `CUDABasic` mide tiempos con `cudaEvent_t` alrededor de cada kernel

### Para decir en la exposición
> "Las 3 variantes comparten el mismo flujo base gracias al patrón Template Method. Solo cambia cómo se ejecuta la reproducción y la evaluación. Esto nos permite comparar apples-to-apples."

---

## Diapositiva 9 — Optimizaciones de CUDA Optimized

| # | Optimización | ¿Mejora? | Explicación |
|---|-------------|----------|-------------|
| 1 | Memoria constante (`__constant__`) | ✅ Sí ($n≤3000$) | Broadcast desde caché L1, reduce latencia |
| 2 | Evaluación intra-bloque (`fitness_kernel_opt`) | ❌ No ($n>3000$) | Overhead de sincronización supera beneficio |
| 3 | Streams CUDA + pinned memory | ❌ No | Kernel de reproducción domina 99% del tiempo, no hay con qué solaparse |
| 4 | Bloque óptimo dinámico | ⚠️ Marginal | `cudaOccupancyMaxPotentialBlockSize` elige ~128, similar a calibración manual |
| 5 | Control de divergencia | ⚠️ Marginal | El repair loop es inherentemente divergente |
| 6 | Parámetros en constante (`c_params`) | ✅ Sí | Elimina tráfico de registros |

**Resultado**: CUDA Optimized es **más lento** que CUDA Basic en todas las instancias (hasta 67% más lento en large).

### Para decir en la exposición
> "Esta es una conclusión importante del trabajo: no todas las optimizaciones CUDA sirven para todos los problemas. En nuestro caso, intentar usar shared memory para fitness en instancias de 10,000 ítems empeoró el rendimiento. Las optimizaciones que realmente funcionaron fueron las más simples: memoria constante y acceso coalescente."

---

## Diapositiva 10 — Diseño Experimental

| Experimento | Descripción | Ejecuciones |
|-------------|-------------|-------------|
| **EXP 1** | 3 instancias × 3 poblaciones × 3 variantes × 10 semillas | 270 |
| **EXP 2** | Efecto del bloque (32,64,128,256) × 2 variantes × 10 semillas | 80 |
| **EXP 3** | Speed-up = T_seq / T_GPU (derivado de EXP 1) | 270 |
| **EXP 4** | Efecto población en large × 3 variantes × 10 semillas | 90 |

**Poblaciones**: 512, 1024, 4096
**Semillas**: 42, 43, 44, 45, 46, 47, 48, 49, 50, 51

**Hardware usado:**
| | Daniel Burgos | Christian Alarcón |
|--|--------------|-------------------|
| CPU | Ryzen 5 5600X (6C/12T) | Ryzen 5 5500 (6C/12T) |
| RAM | 32 GB | 32 GB |
| GPU | RTX 3070 (8 GB) | RTX 4060 (8 GB) |
| SO | Windows 11 | Linux LTS 6.18.23 |
| CUDA | 13.2 | 13.2 |

### Para decir en la exposición
> "Seguimos el diseño experimental mínimo de la actividad: 3 instancias, 3 tamaños de población, 10 repeticiones con semillas registradas para garantizar reproducibilidad total."

---

## Diapositiva 11 — Resultados Principales

### Tiempo de ejecución (instancia large, pop 4096):
| Variante | Tiempo promedio |
|----------|----------------|
| Sequential | 1,114,741 ms (~18.6 min) |
| CUDA Basic | 78,480 ms (~1.3 min) |
| CUDA Optimized | 112,651 ms (~1.9 min) |

### Speed-up (CUDA Basic vs Sequential):
| Instancia | pop=512 | pop=1024 | pop=4096 |
|-----------|---------|----------|----------|
| small | 2.56× | 4.08× | 9.29× |
| medium | 7.62× | 16.80× | 46.71× |
| large | 2.40× | 3.66× | 14.20× |

### Desglose de tiempo GPU (large, pop 4096, CUDA Basic):
| Componente | Tiempo (ms) | % del total |
|------------|-------------|-------------|
| Kernel reproducción | 76,634 | 99.3% |
| Kernel fitness | 511 | 0.7% |
| Transferencia H→D | 6.4 | ~0.01% |
| Transferencia D→H | 21.0 | ~0.03% |

### Para decir en la exposición
> "El speed-up máximo fue de 46.7× en la instancia mediana con población 4096. El kernel de reproducción domina el 99.3% del tiempo GPU, y las transferencias son prácticamente despreciables (0.03% del tiempo total). La variante optimizada resultó más lenta que la básica en todas las configuraciones."

---

## Diapositiva 12 — Efecto del Bloque y Población

### EXP 2: Tamaño de bloque (large, pop 4096, 100 generaciones)
| Variante | bs=32 | bs=64 | bs=128 | bs=256 |
|----------|-------|-------|--------|--------|
| CUDA Basic | **1,713** | **1,713** | 1,772 | 1,909 |
| CUDA Optimized | 2,579 | 2,594 | 2,575 | **2,564** |

- Efecto marginal (±10%), óptimo en bs=32/64 para Basic, bs=256 para Optimized

### EXP 4: Efecto de población (large)
| Variante | pop=512 | pop=1024 | pop=4096 |
|----------|---------|----------|----------|
| Sequential | 143,549 ms | 255,574 ms | 1,114,741 ms |
| CUDA Basic | 59,905 ms | 69,907 ms | 78,480 ms |
| CUDA Optimized | 75,027 ms | 103,817 ms | 112,651 ms |

**Fitness mejora con población**: 0.1936 → 0.1953 → 0.1966

### Para decir en la exposición
> "En GPU el tiempo escala de forma sub-lineal: al aumentar la población 8× (de 512 a 4096), el tiempo solo crece 1.3× en CUDA Basic. En CPU, el mismo aumento multiplica el tiempo por 7.8×."

---

## Diapositiva 13 — Errores Corregidos (si preguntan)

| Bug | Impacto | Solución |
|-----|---------|----------|
| Penalizaciones inconsistentes entre CPU y CUDA | Speed-up inválido | Unificadas a 0.20 como default |
| `(int)` cast en max_weight/volume | Pérdida de precisión | Cambiado a `float` |
| RNG no reproducible | Resultados no replicables | `initial_seed` en vez de `rng() ^ rd()` |
| Transferencia masiva D→H cada gen | ~12 GB acumulados | Lazy download del mejor |
| `reduce_best_block/warp` nunca llamados | Código muerto | Eliminados del flujo |

### Para decir en la exposición
> "Encontramos y corregimos varios bugs durante el desarrollo. El más crítico era la transferencia masiva de población completa cada generación, que agregaba ~12 GB de tráfico PCIe y destruía cualquier speed-up potencial."

---

## Diapositiva 14 — Live Demo (ejecución en vivo)
Comando a ejecutar (mostrar en pantalla):
```bash
./Release/run.exe -i ../data/small -v cuda_basic -t 1 -s 42 -p 1024 -g 300 \
  --pen-weight 0.25 --pen-volume 0.0 --pen-category 0.25 \
  --pen-incomp 0.25 --pen-dep 0.25 --block-size 128
```

### Mostrar en la salida:
- [ ] "Best fitness: ..."
- [ ] "Feasible: Yes"
- [ ] "Wall-clock time (ms): ..."
- [ ] "Kernel fitness total (ms): ..."
- [ ] "Kernel repro total (ms): ..."
- [ ] "Transfer H→D total (ms): ..."
- [ ] "Transfer D→H total (ms): ..."
- [ ] "Feasible solutions (%): ..."

### Mostrar en CSV (tener abierto el archivo de resultados):
- [ ] Una fila completa con las columnas
- [ ] Señalar: seed, best_fitness, feasible, wall_time_ms

---

## Diapositiva 15 — Conclusiones

1. **Speed-up significativo**: hasta 46.7× en instancia mediana con GPU vs CPU.
2. **Cuello de botella**: kernel de reproducción (99.3% del tiempo GPU) por divergencia de warps en la reparación.
3. **Optimizaciones contraproducentes**: shared memory y streams empeoraron el rendimiento en todas las instancias.
4. **Transferencias despreciables**: solo 21 ms acumulados D→H en 300 generaciones.
5. **Bloque: efecto marginal**: ±10% entre configuraciones, óptimo en bs=32/64.
6. **Mejor solución siempre factible**: tracking de `best_valid_individual` (Req 5.1).

**Trabajo futuro**: Reducir divergencia de warps en la reparación usando reducciones warp-level en vez de loops iterativos.

### Para decir en la exposición
> "La principal lección es que en algoritmos genéticos con reparación heurística, el cuello de botella no está en la evaluación de fitness ni en las transferencias, sino en la reparación post-mutación. Cualquier optimización debe enfocarse en mitigar la divergencia de warps durante esa fase."

---

## Preguntas Frecuentes de la Defensa (preparar respuestas)

### Memoria
- **¿Cómo están los datos en memoria GPU?** Arreglo lineal `uint8_t[pop_size × n_items]`, acceso `genes[ind × n_items + gene]`. Esto da coalescencia porque hilos consecutivos leen genes consecutivos.
- **¿Cuánta VRAM usa?** Para large+pop4096: ~78 MB (población + offspring). Sobran ~7.9 GB en una RTX 4060.
- **¿Por qué uint8_t y no bool?** CUDA no garantiza que bool ocupe 1 byte. `uint8_t` es explícito.

### Indexación
- **¿Cómo se mapean hilos a individuos?** `ind = blockIdx.x * blockDim.x + threadIdx.x`, con chequeo `if (ind >= pop_size) return`.
- **¿Y los genes dentro de un individuo?** Loop `for (g = 0; g < n_items; g++)` dentro del mismo hilo.

### Aleatoriedad
- **¿Cómo se generan números aleatorios en GPU?** CuRAND: `curand_init(seed, ind, 0, &state)` inicializa un estado por hilo. Cada hilo usa `curand_uniform(&local_state)`.
- **¿Es reproducible?** Sí, misma seed genera exactamente la misma secuencia. En CPU usamos `std::mt19937(seed)`.

### Sincronización
- **¿Hay condiciones de carrera?** No. Cada hilo escribe en posiciones distintas: `fitness[ind]`, `offspring[ind * n_items + gene]`, etc.
- **¿Usan __syncthreads()?** Solo en `fitness_kernel_opt` (que tiene un bloque por individuo). En los demás kernels no hay sincronización porque cada hilo es independiente.

### Transferencias
- **¿Qué se transfiere cada generación?** Solo `fitness[pop_size]`, `penalty[pop_size]`, `hard_feas[pop_size]`, `is_valid[pop_size]` (~10 bytes/ind).
- **¿Y los cromosomas?** Solo el del mejor individuo, una vez al final via `get_best()`.
- **¿Cuánto tiempo toman las transfers?** ~21 ms acumulados en 300 generaciones (<0.03% del total).

### Medición de tiempos
- **¿Cómo miden los kernels?** Con pares `cudaEvent_t`:
  ```cuda
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start);
  kernel<<<...>>>(...);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  ```
- **¿El speed-up incluye transfers?** Sí, el tiempo CUDA reportado incluye kernels + transfers (H→D + D→H).

### Correctitud
- **¿Cómo aseguran que la solución final es factible?** `best_valid_individual` tracking en `update_best_solution()`. `get_best()` retorna el mejor factible si el de mayor fitness es inválido.
- **¿Qué valida `is_valid`?** Peso, volumen, incompatibilidades, dependencias y categorías.
- **¿Qué pasa si nunca hay un individuo factible?** Se reporta el de mejor fitness aunque sea inválido (no ocurrió en nuestras ejecuciones).

---

## Checklist Pre-Presentación
- [ ] PPTX con todas las diapositivas listas
- [ ] Fragmentos de código insertados en la PPTX (no abrir el proyecto)
- [ ] Ejecutable compilado en la máquina de presentación
- [ ] Datos CSV listos para mostrar
- [ ] Comando de ejecución copiado en diapositiva
- [ ] Semilla, instancia y parámetros visibles durante la demo
- [ ] Resultados de los 4 experimentos en tablas
- [ ] Respuestas preparadas para preguntas técnicas
