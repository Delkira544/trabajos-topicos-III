# Actividad 2: Mochila Extendida con Algoritmo Genético

## Descripción
Implementación de un Algoritmo Genético para resolver el problema de la mochila extendido con múltiples restricciones:

- **Restricciones duras:** Peso y volumen (soluciones inválidas si se exceden)
- **Restricciones blandas:** Categoría, incompatibilidades y dependencias (penalizadas en el fitness)

## Compilación

```bash
make        # Compilar
make clean  # Limpiar archivos de build
```

## Uso

```bash
# Instancia pequeña (obligatorio --instance)
./mochila_ga --instance data/small --seed 123

# Con parámetros personalizados
./mochila_ga --instance data/medium --population 200 --generations 100 --mutation-rate 0.05 --verbose

# Ver ayuda
./mochila_ga --help
```

## Parámetros de línea de comandos

| Flag | Descripción | Default |
|------|-------------|---------|
| `--instance <dir>` | Directorio de la instancia | (requerido) |
| `--population <n>` | Tamaño de población | 100 |
| `--generations <n>` | Número de generaciones | 50 |
| `--mutation-rate <f>` | Tasa de mutación (0-1) | 0.01 |
| `--seed <n>` | Semilla aleatoria | 0 |
| `--verbose` | Mostrar estadísticas de población | off |
| `--variant <type>` | Variante: standard, islands | standard |
| `--threads <n>` | Número de hilos (para islands) | 1 |
| `--help, -h` | Mostrar ayuda | - |

## Generar instancias

```bash
python generate_instances.py --seed 42
```

Esto genera instancias `small` (100 ítems), `medium` (1000 ítems) y `large` (10000 ítems) en `data/`.

## Estructura del Proyecto

```
src/
├── main.cpp                  # Entry point, parseo de argumentos
├── genetic_algorithm.{hpp,cpp} # Algoritmo genético principal con elitismo
├── crossover.{hpp,cpp}       # Cruce SinglePoint
├── mutation.{hpp,cpp}        # Mutación BitFlip
├── selection.{hpp,cpp}       # Selección por torneo
├── fitness.{hpp,cpp}         # Evaluación con penalización gradual
├── instance_loader.{hpp,cpp} # Carga de instancias desde CSV
├── island_model.{hpp,cpp}    # Modelo de islas (pendiente)

data/
├── small/                    # 100 ítems
├── medium/                   # 1000 ítems (no versionado)
└── large/                    # 10000 ítems (no versionado)
```

## Formato de Instancias

Cada carpeta de instancia debe contener 5 archivos CSV:

| Archivo | Columnas | Descripción |
|---------|----------|-------------|
| `items.csv` | id, valor, peso, volumen, categoria | Ítems disponibles |
| `category_rules.csv` | categoria, minimo, maximo | Límites por categoría |
| `incompatibilities.csv` | id_item_a, id_item_b | Pares incompatibles |
| `dependencies.csv` | id_item, id_requerido | Dependencias entre ítems |
| `knapsack_config.csv` | max_weight, max_volume | Capacidad de la mochila |

## Algoritmo

### Características implementadas
- **Representación:** Cromosoma binario (1 = ítem seleccionado, 0 = no seleccionado)
- **Inicialización:** Aleatoria respetando restricciones de peso/volumen
- **Selección:** Torneo de tamaño 3
- **Cruce:** Single-point crossover
- **Mutación:** Bit-flip con tasa configurable
- **Elitismo:** Preservación del mejor individuo entre generaciones
- **Fitness:** Valor total con penalización gradual por restricciones violadas

### Complejidad
La evaluación de fitness es O(n + i + d) por individuo gracias al uso de `unordered_set` para búsquedas O(1) de IDs seleccionados, donde:
- n = número de ítems
- i = número de incompatibilidades
- d = número de dependencias

## Restricciones

| Tipo | Restricción | Tratamiento |
|------|-------------|-------------|
| Dura | Peso máximo | Penalización proporcional al exceso |
| Dura | Volumen máximo | Penalización proporcional al exceso |
| Blanda | Reglas de categoría | Penalización por ítem fuera de rango |
| Blanda | Incompatibilidades | Penalización fija por par violado |
| Blanda | Dependencias | Penalización fija por dependencia incumplida |
