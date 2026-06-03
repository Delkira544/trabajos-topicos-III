# Penalty Parameter Tuner - Guía de Uso Actualizada

## Estado: ✅ FUNCIONAL CON LOGGING COMPLETO

El **Penalty Tuner** ahora cuenta con logging detallado en **cada paso** del proceso para facilitar debugging y seguimiento.

## Características de Logging

Niveles de log implementados:

- `[INIT]` - Inicialización del programa
- `[LOG]` - Información de progreso general
- `[PROGRESS]` - Indicador de progreso por configuración
- `[SEED_RESULT]` - Resultado individual por semilla
- `[CONFIG_RESULT]` - Resultado promediado de una configuración
- `[SUCCESS]` - Ejecución exitosa
- `[ERROR]` - Errores

## Compilación

```bash
cd algoritmo-genetico
mkdir -p build
cd build
cmake ..
make
```

## Uso Rápido

### Prueba Rápida (2 minutos)

```bash
./penalty_tuner \
  --instance ../data/small \
  --step 1.0 \
  --seeds 1 \
  --generations 10 \
  --top-n 5
```

**Parámetros:**
- `--instance ../data/small` - Instancia pequeña para testing
- `--step 1.0` - Solo 5 configuraciones
- `--seeds 1` - Una semilla por config
- `--generations 10` - 10 generaciones (muy rápido)
- `--top-n 5` - Mostrar top 5

### Búsqueda Standard (1-2 horas)

```bash
./penalty_tuner \
  --instance ../data/medium \
  --step 0.20 \
  --seeds 2 \
  --generations 50 \
  --top-n 30 \
  --output results/penalty_results.csv
```

**Parámetros:**
- `--step 0.20` - ~7,700 configuraciones
- `--seeds 2` - 2 semillas por config = 15,400 AG
- `--generations 50` - 50 generaciones por AG
- Tiempo: ~1-2 horas sin paralelización

### Búsqueda Exhaustiva (12+ horas)

```bash
./penalty_tuner \
  --instance ../data/large \
  --step 0.10 \
  --seeds 2 \
  --generations 100 \
  --top-n 30 \
  --output results/exhaustive_search.csv
```

## Parámetros Disponibles

| Flag | Descripción | Default | Rango |
|------|-------------|---------|-------|
| `-i, --instance` | Ruta a instancia | - | Requerido |
| `-s, --step` | Paso de discretización | 0.10 | 0.01-1.0 |
| `-k, --seeds` | Semillas por config | 2 | 1+ |
| `-g, --generations` | Generaciones de AG | 50 | 1+ |
| `-n, --top-n` | Top N resultados | 30 | 1+ |
| `-o, --output` | Archivo de salida | `results/penalty_search_results.csv` | - |

## Ejemplo de Salida

### Console Output (Logs + Tabla Final)

```
[INIT] Penalty Tuner starting...
[LOG] Setting up CLI options...
[LOG] Parsing command line arguments...
[LOG] CLI parsing completed
[LOG] Parameters:
  - Instance path: ../data/small
  - Step size: 0.5
  - Number of seeds: 1
  - Top N: 5
  - Generations per AG: 10
  - Output file: ../results/test_results.csv

[LOG] Creating PenaltyTuner instance...
[LOG] Loading instance from: ../data/small
[LOG] Instance loaded successfully
[LOG] PenaltyTuner created successfully
[LOG] Tuning generations set to: 10

[LOG] Starting grid search...

╔════════════════════════════════════════════════════════════════╗
║     PENALTY TUNING: Grid Search with Normalization            ║
╚════════════════════════════════════════════════════════════════╝

[LOG] Starting penalty tuner...
[LOG] Instance path: ../data/small
[LOG] Instance loaded with 100 items
[LOG] Step size: 0.5
[LOG] Seeds per configuration: 1

[LOG] Generating grid search space...
[LOG] Starting grid generation with step size: 0.5
[LOG] Grid generation completed. Total configurations: 15

[PROGRESS] Config 1/15: WE=0.00 VE=0.00 CV=0.00 IC=0.00 DV=1.00
[LOG] Seed 1/1 for this config...
[LOG] Evaluating config with seed 0: WE=0.00 VE=0.00 CV=0.00 IC=0.00 DV=1.00
[LOG] Penalties applied successfully
[LOG] Set generations for tuning: 10
[LOG] Solver config created
[LOG] Solver created, starting run...

=== Starting Genetic Algorithm ===
Population size: 120
Generations: 10

[LOG] Solver finished running
[SEED_RESULT] Seed 1/1: Validity=1.0, Fitness=0.723, Generations=10
[CONFIG_RESULT] Config average - Validity: 1.0, Fitness: 0.723, Generations: 10

[PROGRESS] Config 2/15: WE=0.00 VE=0.00 CV=0.00 IC=0.50 DV=0.50
... más configuraciones ...

[LOG] Sorting results by validity rate...
[LOG] Results sorted successfully!
[LOG] Search completed!

╔════════════════════════════════════════════════════════════════╗
║                  TOP 5 PENALTY CONFIGURATIONS                 ║
║         Metric: Validity Rate (tasa de soluciones válidas)    ║
╚════════════════════════════════════════════════════════════════╝

Rank | WE    VE    CV    IC    DV   | Valid% | Fitness | Gen
-----|-----|-----|-----|-----|-----|---------|---------|---------
  1  | 0.00 0.00 0.00 0.50 0.50 | 100.0% | 0.745   |   10
  2  | 0.00 0.00 0.00 0.00 1.00 | 100.0% | 0.723   |   10
  3  | 0.50 0.00 0.00 0.00 0.50 |  80.0% | 0.712   |   10
  4  | 0.50 0.00 0.00 0.50 0.00 |  80.0% | 0.698   |   10
  5  | 0.00 0.50 0.00 0.50 0.00 |  80.0% | 0.681   |   10

[LOG] Opening file for writing: ../results/test_results.csv
[LOG] File opened successfully
[LOG] Writing CSV header...
[LOG] Header written
[LOG] Writing 15 results to CSV...
[LOG] Written 100 results...
[LOG] File closed successfully
[SUCCESS] Results saved to: ../results/test_results.csv

[SUCCESS] Penalty tuner completed successfully!
```

### CSV Output (`results/penalty_search_results.csv`)

```csv
rank,weight_excess,volume_excess,category_violation,incompatibility,dependency_violation,avg_validity_rate,avg_best_fitness,avg_generations_to_valid,seed1_validity,seed1_fitness,seed1_generations
1,0.00,0.00,0.00,0.50,0.50,1.0,0.745,10,1.0,0.745,10
2,0.00,0.00,0.00,0.00,1.00,1.0,0.723,10,1.0,0.723,10
3,0.50,0.00,0.00,0.00,0.50,0.8,0.712,10,0.8,0.712,10
...
```

## Ventajas del Sistema de Logging

1. **Visibilidad completa**: Sigue cada paso del proceso
2. **Debugging fácil**: Identifica dónde se detiene si hay error
3. **Monitoreo**: Sabe cuándo se completa cada configuración
4. **Validación**: Verifica que penalizaciones se aplican correctamente
5. **Timing**: Ve cuánto tarda cada evaluación

## Interpretación de Resultados

### Validity Rate

- **100%** → Excelente (ambas semillas encontraron solución válida)
- **50%** → Bueno (al menos una encontró solución)
- **0%** → Malo (ninguna encontró solución válida)

### Fitness

- Mayor es mejor
- Se promedia entre semillas
- Se toma el mejor individuo de cada ejecución AG

### Generaciones

- Tiempo en convergencia
- Cuantas menos generaciones, más rápido convergió
- No es métrica principal, es informativo

## Recomendaciones

### Para Testing Rápido:
```bash
--step 1.0 --seeds 1 --generations 10
```
Tiempo: < 1 minuto

### Para Búsqueda Estándar:
```bash
--step 0.20 --seeds 2 --generations 50
```
Tiempo: ~1-2 horas

### Para Búsqueda Exhaustiva:
```bash
--step 0.10 --seeds 3 --generations 100
```
Tiempo: ~12-24 horas

## Troubleshooting

### "Error al cargar las reglas..."
- **Causa**: Los archivos CSV faltan algunas reglas
- **Solución**: Es normal, el sistema continúa funcionando

### "Could not open file"
- **Causa**: El directorio `results/` no existe
- **Solución**: `mkdir -p results`

### Ejecución muy lenta
- **Solución 1**: Reducir `--step` (ej: 0.5 en lugar de 0.1)
- **Solución 2**: Reducir `--generations` (ej: 20 en lugar de 50)
- **Solución 3**: Usar `--seeds 1` en lugar de 2

## Aplicar Resultados

Una vez encontrada la mejor configuración, actualizar:

```bash
# Editar include/config/constants.hpp o src/config/constants.cpp
float WEIGHT_EXCESS_PENALTY = 0.20f;  // Tu valor
float VOLUME_EXCESS_PENALTY = 0.20f;  // Tu valor
// ... etc

# Recompilar
cd build && make
```

## Notas Finales

- El tuning es **completamente independiente** del AG normal
- Las penalizaciones ahora son **dinámicas** (no constexpr)
- Los logs ayudan a monitorear progreso en tiempo real
- Los CSV contienen todos los datos para análisis posterior

¡El sistema está **listo para usar**!
