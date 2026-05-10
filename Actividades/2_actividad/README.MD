# Actividad 2 — Mochila Extendida con Algoritmo Genético Paralelo

Implementación de un Algoritmo Genético (AG) para resolver el problema de la
mochila con múltiples restricciones, en tres variantes: **secuencial**,
**paralela con OpenMP** y **modelo de islas**. Curso INFO1194.

---

## Restricciones del problema

| Tipo | Restricción | Tratamiento |
|------|-------------|-------------|
| **DURA** | Peso máximo `W` | `hard_feasible=false` si se excede; el orden lexicográfico en `IsBetter` la trata como bloqueante. Penalización **cuadrática** sobre la capacidad. |
| **DURA** | Volumen máximo `V` | Idem peso. |
| **DURA** | Incompatibilidades entre pares | Penalización proporcional. La solución no se reporta como `is_valid` si hay pares violados. |
| **DURA** | Dependencias obligatorias | Penalización proporcional. La solución no se reporta como `is_valid` si hay dependencias rotas. |
| **BLANDA** | Reglas de categoría (min/max ítems) | Penalización proporcional al número de categorías fuera de rango. |

> **Nota sobre duras vs blandas**: todas las restricciones se penalizan en el
> fitness (estrategia clásica de AG con penalización), pero el flag
> `hard_feasible` y la función `IsBetter` aplican un **orden lexicográfico**
> que prioriza absolutamente la factibilidad de peso/volumen, y `is_valid`
> exige cero violaciones de incompatibilidades y dependencias para considerar
> la solución como factible final.

---

## Compilación

```bash
make           # build con -O2 -fopenmp
make clean     # limpiar
```

Compilación manual equivalente:
```bash
g++ -O2 -fopenmp -std=c++17 src/*.cpp -o mochila_ga
```

## Generación de instancias

```bash
python3 generate_instances.py [--seed N] [--output DIR] [--only NOMBRE]
```

Genera 6 archivos CSV por instancia en `data/{small,medium,large}/`:

| Archivo | Columnas | Descripción |
|---|---|---|
| `items.csv` | `id, valor, peso, volumen, categoria` | Catálogo de ítems |
| `category_rules.csv` | `categoria, minimo, maximo` | Cuotas por categoría |
| `incompatibilities.csv` | `id_item_a, id_item_b` | Pares incompatibles |
| `dependencies.csv` | `id_item, id_requerido` | Dependencias |
| `knapsack_config.csv` | `max_weight, max_volume` | Capacidad |
| `penalty_config.csv` | `penalty_type, value` | Pesos `α,β,γ,δ,ε` (Σ=1) |

Tamaños: `small`=100, `medium`=1.000, `large`=10.000 ítems.

## Ejecución

```bash
# Atajos por instancia (ver Makefile)
make small
make medium
make large

# Genérico
./mochila_ga --instance data/<dir> --variant <standard|islands> \
             --threads <n> --seed <n> --population <n> \
             --generations <n> --report-file <ruta.csv>
```

### Argumentos CLI

| Flag | Descripción | Default |
|------|-------------|---------|
| `--instance <dir>` | Directorio de la instancia | (requerido) |
| `--variant <type>` | `standard` (secuencial+OpenMP) o `islands` | `standard` |
| `--threads <n>` | Hilos OpenMP (>1 dispara `RunParallel`) | `1` |
| `--seed <n>` | Semilla del RNG | `0` |
| `--population <n>` | Tamaño de población | `100` |
| `--generations <n>` | Generaciones máximas | `50` |
| `--mutation-rate <f>` | Tasa de mutación bit-flip ∈ [0,1] | `0.03` |
| `--convergence-threshold <f>` | Umbral de convergencia clásica | `1e-5` |
| `--report-file <ruta>` | CSV con stats por generación | (opcional) |
| `--num-islands <n>` | Solo `islands` | `4` |
| `--pop-per-island <n>` | Solo `islands` | `25` |
| `--migration-freq <n>` | Solo `islands` | `10` |
| `--num-migrants <n>` | Solo `islands` | `2` |
| `--topology <type>` | `ring` o `random` | `ring` |
| `--benchmark` | Ejecutar suite de benchmarks | off |
| `--verbose` | Detalle extra | off |
| `--help, -h` | Ayuda | — |

### Reproducibilidad

Configuración por defecto en el Makefile (semilla, hilos, etc.) usada para los reportes adjuntos:

```
SEED        = 132512
GENERATIONS = 700
POPULATION  = 120         # small / medium
POPULATION_LARGE = 250    # large
THEADS      = 8
```

---

## Estructura del proyecto

```
src/
├── main.cpp                      # CLI, parseo, dispatch de variantes
├── genetic_algorithm.{hpp,cpp}   # Variantes secuencial (Run) y paralela (RunParallel)
├── island_model.{hpp,cpp}        # Variante modelo de islas (Run / RunParallel)
├── selection.{hpp,cpp}           # Tournament selection
├── crossover.{hpp,cpp}           # SinglePoint y Uniform
├── mutation.{hpp,cpp}            # BitFlip, BitFlipAsymmetric, TargetedFix
├── fitness.{hpp,cpp}             # Evaluate + PrintConstraintDetails
├── instance_loader.{hpp,cpp}     # Carga de los 6 CSV
└── benchmark.{hpp,cpp}           # Suite de benchmarks (--benchmark)

data/{small,medium,large}/        # Instancias generadas
result/                           # CSV de reportes (--report-file)
generate_instances.py             # Generador Python
```

---

## Algoritmo

### Representación
Cromosoma binario de largo `n` (1 = ítem seleccionado).

### Inicialización
`Bernoulli(0.25)` con corte greedy por peso/volumen → la población inicial
empieza **bajo capacidad**, dejando margen para que crossover/mutación no
ingresen de inmediato a territorio infactible.

### Selección
**Tournament selection** con `k=5`. Comparación lexicográfica
(`IsBetter`): hard_feasible → is_valid → menor penalty → mayor fitness.

### Crossover
**Uniform crossover** con `inclusion_bias = 0.45`. Cuando los padres
coinciden en un gen, el hijo lo hereda; cuando difieren, se samplea con
sesgo a no incluir → preserva mejor la factibilidad de capacidad que el
single-point.

### Mutación (3 capas)
1. **BitFlipAsymmetric**: si el individuo excede capacidad, sesga fuerte a
   eliminar (P(1→0) × 3, P(0→1) × 0.25); si está bajo capacidad, simétrica.
2. **TargetedFix** (probabilística, 50% por hijo): recopila las violaciones
   soft activas (incompat + dep) y fixea **una al azar** elegida random.
   No es greedy ni determinista — sólo da al GA un sesgo a tocar genes
   problemáticos. NO es "Repair" (no resuelve todo de una pasada).

### Elitismo
**Multi-elite del 10%**: el top-N (con N = population × 0.10, mín 1) se
preserva intacto cada generación.

### Anti-estancamiento (inyección de diversidad)
Si el `best_ever` no mejora en `stagnation_limit_ = 50` gens, se reemplaza
el peor 30% de la población con individuos nuevos re-inicializados.

### Criterios de término

Se evalúan en este orden:

1. **`--generations` alcanzado** (límite duro de la corrida).
2. **Factibilidad + ventana de refinamiento**: cuando el `best_ever_` se vuelve `is_valid` por primera vez, se registra la generación en `first_feasible_gen_`. Una vez transcurridas `gens_after_feasible_limit_ = 75` generaciones desde ese momento, el algoritmo termina. Esto pone un techo predecible al tiempo post-factibilidad: el AG dispone de 75 gens para refinar el valor objetivo y luego sale sí o sí.
3. **Convergencia clásica** (fallback): `avg_delta < threshold` sobre ventana de 10 gens, **y** `best_ever_.is_valid`. Si el fitness se aplana antes de los 75 gens, el algoritmo sale por aquí.

> Si la corrida termina por (2) o (3), la solución reportada es **factible respecto a TODAS las restricciones** (peso, volumen, categorías, incompatibilidades, dependencias). Sólo si termina por (1) — agotamiento de generaciones — puede no haber alcanzado factibilidad.

### Función de fitness

```
violacion_norm = α · norm_peso² + β · norm_vol² + γ · norm_cat
               + δ · norm_incomp + ε · norm_dep         (Σ=1)

fitness = obj_weight · valor_norm − pen_weight · violacion_norm
```

Donde:
- **norm_peso**, **norm_vol**: cuadráticas amplificadas — `min(1, (exceso/cap)² · 100)`.
  Un 2.5% de exceso ya da norm=0.0625 (vs. 0.0165 con normalización lineal por exceso máximo).
- **valor_norm** = Σ valores seleccionados / Σ valores totales.
- `obj_weight = 0.3`, `pen_weight = 0.7` → fuerte presión a la factibilidad.
- Pesos por defecto: α=β=0.30 (HARD), γ=0.05, δ=0.20, ε=0.15 — sobrescribibles desde `penalty_config.csv`.

### Variante OpenMP (`RunParallel`)

| Bloque | Estrategia |
|---|---|
| Evaluación inicial | `#pragma omp parallel for schedule(static)` sobre toda la población |
| Selección + Crossover + Mutación | `#pragma omp parallel for` sobre los pares de hijos; **un RNG dedicado por par** (`pair_rng` derivado de `seed + gen·10000 + pair`) → resultados reproducibles e independientes del nº de hilos |
| Re-evaluación de hijos | `#pragma omp parallel for` |
| Variables shared | `population` (lectura), `instance` (lectura), `children[]` (escritura por índice disjunto) |
| Variables private/local | `pair_rng`, `tournament_size`, `i`, `p1/p2/c1/c2` (definidas dentro del cuerpo del `for` paralelo → automáticamente privadas) |
| Race conditions | Ninguna: el RNG es por par, los índices de `children[]` son disjuntos, y `population` no se modifica durante el bucle |

### Variante Modelo de Islas (`IslandModel`)

- N islas (`--num-islands`) con población `--pop-per-island` cada una y RNG independiente.
- Evolución local con tournament + SinglePoint + BitFlip + elitismo de 1 dentro de cada isla.
- **Migración** cada `--migration-freq` gens: cada isla envía sus `--num-migrants` mejores; reemplazan a los peores de la isla destino (no se altera el tamaño de población).
- **Topologías**: `ring` (envía a la izquierda en círculo) o `random` (destino aleatorio ≠ origen).
- `RunParallel` paraleliza por isla con `#pragma omp parallel for schedule(dynamic)` — sin race porque cada hilo trabaja sobre `islands[i]` e `island_rngs[i]` exclusivos.

---

## Salida y reporte

Tras la corrida se imprime:

```
Mejor Fitness: <float>
Solución válida: Sí|No
Ítems seleccionados, peso/capacidad, volumen/capacidad, valor total
Detalle por restricción (✓/✗ peso, vol, cada categoría, incompat, dep)
Tiempo de ejecución
```

Con `--report-file <path>` se vuelca un CSV por generación:
`generation, best_fitness, avg_fitness, worst_fitness, valid_count, convergence_delta`.

---

## Resumen de mejoras incrementales (por iteración del proyecto)

| # | Mejora | Archivo |
|---|---|---|
| 1 | Penalización cuadrática `(exceso/cap)²·100` para hard constraints | `fitness.cpp` |
| 2 | `PenaltyConfig` redistribuido (α=β=0.30, obj=0.3 / pen=0.7) | `genetic_algorithm.hpp`, `generate_instances.py` |
| 3 | `IsBetter` lexicográfico (hard_feasible → is_valid → penalty → fitness) | `genetic_algorithm.hpp` |
| 4 | Uniform crossover con sesgo 0.45 | `crossover.{hpp,cpp}` |
| 5 | Mutación asimétrica consciente de capacidad | `mutation.{hpp,cpp}` |
| 6 | Anti-estancamiento (50 gens → reemplaza 30%) | `genetic_algorithm.cpp` |
| 7 | Elitismo del 10% (top-N) | `genetic_algorithm.cpp` |
| 8 | Tournament k=5 | `genetic_algorithm.cpp` |
| 9 | `POPULATION_LARGE = 250` para large | `Makefile` |
| 10 | Inicialización Bernoulli(0.25) | `genetic_algorithm.cpp` |
| 11 | `Mutation::TargetedFix` — mutación dirigida a violaciones | `mutation.{hpp,cpp}` |
| 12 | Early-stop con ventana de refinamiento de 75 gens post-factibilidad | `genetic_algorithm.cpp`, `island_model.cpp` |
| 13 | Modelo de islas con todas las mejoras integradas (Uniform, BitFlipAsymmetric, TargetedFix, multi-elite, anti-estancamiento, early-stop) | `island_model.{hpp,cpp}` |

---

## Resultados con configuración por defecto (`SEED=132512`, 8 hilos)

### Variante `standard` (paralela, OpenMP)

| Instancia | n | Población | Resultado | Salida | Tiempo |
|---|---|---|---|---|---|
| small | 100 | 120 | ✓ válida | convergencia gen 62 | ~0.09 s |
| medium | 1.000 | 120 | ✓ válida | convergencia gen 38 | ~0.40 s |
| large | 10.000 | 250 | ✓ válida | convergencia gen 249 (factible desde gen 239) | ~46 s |

### Variante `islands` (4 islas, 8 hilos, ring)

| Instancia | n | Pop/isla | Resultado | Salida | Tiempo |
|---|---|---|---|---|---|
| small | 100 | 30 | ✓ válida | convergencia gen 68 | ~0.13 s |
| medium | 1.000 | 50 | ✓ válida | convergencia gen 46 (factible desde gen 1) | ~1.2 s |
| large | 10.000 | 80 | ✓ válida | convergencia gen 199 (factible desde gen 182) | ~70 s |

> En todas las corridas la salida fue por **convergencia clásica** porque el fitness se aplanó antes de agotar la ventana de 75 gens post-factibilidad. La ventana de 75 gens actúa como techo: si el fitness siguiera mejorando lentamente, el algoritmo saldría a las 75 gens exactas tras la primera factibilidad — garantizando un tiempo de ejecución acotado y reproducible.
