# Penalty Parameter Tuner

## Descripción

El **Penalty Tuner** es una herramienta de búsqueda de parámetros óptimos para las penalizaciones del algoritmo genético en el problema de la mochila multidimensional.

Utiliza **grid search** con restricción de normalización (todos los parámetros suman 1.0) para encontrar la mejor combinación de penalidades que maximice la **tasa de soluciones válidas**.

## Características

- ✅ Grid search discretizado con restricción suma = 1.0
- ✅ Múltiples semillas por configuración para robustez
- ✅ Métrica: tasa de validez de soluciones
- ✅ Salida en CSV para análisis posterior
- ✅ Top N configuraciones visualizadas en consola

## Compilación

```bash
cd algoritmo-genetico
mkdir -p build
cd build
cmake ..
make
```

Esto generará dos ejecutables:
- `run`: Ejecutable principal del AG
- `penalty_tuner`: Herramienta de tuning (NUEVO)

## Uso

### Opción 1: Grid Search Completo (Recomendado)

```bash
./penalty_tuner \
  --instance path/to/instance \
  --step 0.10 \
  --seeds 2 \
  --top-n 30 \
  --output results/penalty_results.csv
```

**Parámetros:**
- `--instance` (requerido): Ruta a los archivos de instancia (carpeta con items.csv, etc.)
- `--step` (default: 0.10): Tamaño de paso para discretización
  - 0.10 → ~15K configuraciones
  - 0.05 → ~4M configuraciones (muy lento)
  - 0.20 → ~7.7K configuraciones (rápido)
- `--seeds` (default: 2): Semillas por configuración
- `--top-n` (default: 30): Mostrar top N mejores
- `--output` (default: results/penalty_search_results.csv): Archivo de salida

### Ejemplo Completo

```bash
mkdir -p results

./penalty_tuner \
  --instance ../instances/knapsack_instance_1 \
  --step 0.10 \
  --seeds 2 \
  --top-n 30 \
  --output results/penalty_search_results.csv
```

## Salida

### Consola (Top 30)

```
╔════════════════════════════════════════════════════════════════╗
║                  TOP 30 PENALTY CONFIGURATIONS                ║
║         Metric: Validity Rate (tasa de soluciones válidas)    ║
╚════════════════════════════════════════════════════════════════╝

Rank | WE    VE    CV    IC    DV   | Valid% | Fitness | Gen
-----|-----|-----|-----|-----|-----|---------|---------|---------
  1  | 0.20 0.20 0.05 0.30 0.25 |  85.0% | 0.756   |   42
  2  | 0.20 0.20 0.10 0.25 0.25 |  83.5% | 0.741   |   45
  3  | 0.20 0.15 0.10 0.30 0.25 |  82.0% | 0.728   |   48
  ...
 30  | 0.10 0.30 0.10 0.30 0.20 |  72.0% | 0.698   |   67
```

### Archivo CSV

```csv
rank,weight_excess,volume_excess,category_violation,incompatibility,dependency_violation,avg_validity_rate,avg_best_fitness,avg_generations_to_valid,seed1_validity,seed1_fitness,seed1_generations,...
1,0.20,0.20,0.05,0.30,0.25,0.85,0.756,42,0.85,0.751,42,0.85,0.761,42
2,0.20,0.20,0.10,0.25,0.25,0.835,0.741,45,0.83,0.735,45,0.84,0.747,45
...
```

## Interpretación de Resultados

### Columnas

- **Rank**: Posición en ranking (ordenado por Valid%)
- **WE, VE, CV, IC, DV**: Parámetros de penalización
- **Valid%**: Porcentaje promedio de soluciones válidas (métrica principal)
- **Fitness**: Fitness promedio del mejor individuo
- **Gen**: Generaciones promedio para convergencia
- **seed*_xxx**: Resultados específicos de cada semilla

### Interpretar la Métrica

```
Valid% = 85.0%  → De 2 ejecuciones, 1.7 llegaron a solución válida (bueno)
Valid% = 50.0%  → De 2 ejecuciones, 1 llegó a solución válida (medio)
Valid% = 0.0%   → Ninguna ejecución llegó a solución válida (malo)
```

## Parámetros de Penalización

Los 5 parámetros principales controlados:

- **WE (Weight Excess)**: Penalidad por exceso de peso
- **VE (Volume Excess)**: Penalidad por exceso de volumen
- **CV (Category Violation)**: Penalidad por violación de reglas de categoría
- **IC (Incompatibility)**: Penalidad por items incompatibles
- **DV (Dependency Violation)**: Penalidad por violación de dependencias

**Restricción**: WE + VE + CV + IC + DV = 1.0

## Tiempo de Ejecución

Con `step=0.10` y `seeds=2`:
- Configuraciones: ~15,000
- Total de AG ejecutados: ~30,000
- Tiempo estimado: 
  - Sin paralelización: **~40-50 horas**
  - Con OpenMP (4 threads): **~10-12 horas** (EN DESARROLLO)

## Próximos Pasos

1. **Ejecutar tuning** en tu instancia
2. **Analizar top 30** resultados
3. **Seleccionar mejor configuración** (balance entre validity rate y fitness)
4. **Aplicar manualmente** en `include/config/constants.hpp`:
   ```cpp
   float WEIGHT_EXCESS_PENALTY        = 0.20f;  // Tu valor encontrado
   float VOLUME_EXCESS_PENALTY        = 0.20f;  // Tu valor encontrado
   // ... etc
   ```
5. **Recompilar** y ejecutar AG con nuevos parámetros

## Notas Importantes

- El tuning es **independiente** del ejecutable principal
- Las penalizaciones ahora son **dinámicas** (no constexpr) para permitir cambios en tiempo de ejecución
- Cada configuración se ejecuta con `verbose=false` para no saturar consola
- Los resultados CSV son completos para análisis estadístico posterior

## Troubleshooting

### Error: "Could not open file"
- Asegúrate de que el directorio `results/` existe
- Crea con: `mkdir -p results`

### Error: "Instance not found"
- Verifica la ruta a la instancia
- Debe apuntar a la carpeta que contiene `items.csv`

### Compilación fallida
- Asegúrate de tener C++17 habilitado
- Verifica que OpenMP esté instalado: `apt-get install libomp-dev`

## Autor

Implementado como extensión del AG para tuning automático de parámetros.
