## 📐 Arquitectura Implementada

### Jerarquía de Clases

```
GeneticSolver (abstracta)
    │
    └── BaseGA (clase intermedia con Template Method)
            │
            ├── Sequential        (ejecución secuencial)
            ├── Parallel          (ejecución paralela con OpenMP)
            ├── IslandsSequential (islas secuenciales)
            └── IslandsParallel   (islas paralelas)
```

### Operadores Inyectables

```
Crossover (interfaz)
    └── SinglePointCrossover

Mutation (interfaz)
    └── UniformMutation

Selection (interfaz)
    └── TournamentSelection

FitnessEvaluator (interfaz)
    └── KnapsackFitness

ConstraintValidator (interfaz)
    └── KnapsackValidator
```

---

## 🏗️ Estructura de Carpetas

```
include/ga/
├── GeneticSolver.hpp              # Clase base abstracta
├── BaseGA.hpp                      # Template Method pattern
├── SolverFactory.hpp               # Factory para crear solvers
│
├── operators/                      # Interfaces de operadores
│   ├── Crossover.hpp
│   ├── Mutation.hpp
│   ├── Selection.hpp
│   ├── FitnessEvaluator.hpp
│   ├── ConstraintValidator.hpp
│   │
│   └── implementations/            # Implementaciones concretas
│       ├── SinglePointCrossover.hpp
│       ├── UniformMutation.hpp
│       ├── TournamentSelection.hpp
│       ├── KnapsackFitness.hpp
│       └── KnapsackValidator.hpp
│
└── solvers/                        # Solvers específicos
    ├── Sequential.hpp
    ├── Parallel.hpp
    └── Islands/
        ├── IslandsSequential.hpp
        └── IslandsParallel.hpp
```

---

## 🔄 Flujo de Ejecución (Template Method)

```
BaseGA::run() [FINAL - no se sobrescribe]
    │
    ├─> initialize_population()        [virtual - puede sobrescribirse]
    │
    ├─> LOOP para cada generación:
    │   │
    │   ├─> evaluate_population()       [virtual]
    │   │
    │   ├─> do_reproduction()           [PURO VIRTUAL - cada solver]
    │   │   ├─ Sequential: reproducción en 1 thread
    │   │   ├─ Parallel: reproducción paralela
    │   │   ├─ Islands: evoluciona cada isla
    │   │   └─ IslandsParallel: islas en paralelo
    │   │
    │   ├─> do_migration()              [OPCIONAL - solo islas]
    │   │
    │   ├─> synchronize_populations()   [OPCIONAL]
    │   │
    │   └─> log_generation_stats()
    │
    └─> Retornar resultado
```

---

## 💡 Cómo Usar

### 1. Línea de Comandos

```bash
# Ejecución secuencial (baseline)
./run -i instancia.txt -v sequential -t 1 -s 42

# Ejecución paralela con 4 threads
./run -i instancia.txt -v parallel -t 4 -s 42

# Islas secuenciales (4 islas, migración cada 5 gen)
./run -i instancia.txt -v islands_sequential -t 1 -s 42 \
  --num-islands 4 --migration-interval 5

# Islas paralelas (4 islas, 4 threads)
./run -i instancia.txt -v islands_parallel -t 4 -s 42 \
  --num-islands 4 --migration-interval 5

# Con parámetros personalizados
./run -i instancia.txt -v parallel -t 8 -s 42 \
  --crossover-rate 0.8 --mutation-rate 0.02 \
  --tournament-size 5
```

---

### 2. Crear un Nuevo Solver

Para agregar un nuevo tipo de solver (ej: CUDA):

```cpp
// include/ga/solvers/CUDA.hpp
#pragma once
#include "ga/BaseGA.hpp"

class CUDA : public BaseGA {
protected:
    void do_reproduction() override {
        // Implementar reproducción en GPU
        // kernels CUDA aquí
    }
    
    void do_migration() override {
        // Si es necesario
    }

public:
    CUDA(KnapsackInstance& inst, ...)
        : BaseGA(inst, ...) {}
};
```

Luego agregar a `SolverFactory::create()`:

```cpp
else if (variant == "cuda") {
    return std::make_unique<CUDA>(instance, config);
}
```

---

### 3. Cambiar Operadores

Los operadores son inyectables. Para usar un crossover diferente:

```cpp
// Crear nuevo crossover
class UniformCrossover : public ga::operators::Crossover {
    Individual apply(...) override { ... }
};

// En SolverFactory::create():
auto crossover = std::make_unique<UniformCrossover>(config.crossover_rate);
```

---

## 🎯 Ventajas de la Arquitectura

| Ventaja | Descripción |
|---------|-------------|
| **Reutilización** | Mismos operadores en todos los solvers |
| **Extensibilidad** | Agregar nuevos solvers sin modificar existentes |
| **Testabilidad** | Cada componente es independiente |
| **Bajo acoplamiento** | Inyección de dependencias |
| **Template Method** | Flujo genérico, personalización específica |
| **Escalabilidad** | Soporta secuencial, paralelo e islas |
| **Mantenibilidad** | Cambios centralizados (factory, interfaces) |

---

## ⚡ Cuellos de Botella Evitados

| Problema | Solución |
|----------|----------|
| **Evaluación lenta** | Paralelizable en `Parallel::do_reproduction()` |
| **RNG no thread-safe** | Cada thread genera su propio RNG con seed único |
| **Sincronización costosa** | Migración cada N generaciones, no cada gen |
| **Memoria** | Pre-asignación de offspring, reutilización de buffers |
| **Reproducibilidad** | Seed garantiza determinismo en Sequential |
| **Diversidad (islas)** | Cada isla evoluciona independientemente |

---

## 📊 Ejemplo de Salida

```
========== Problem Instance ==========
Instance: instancia.txt
Items: 100
Max weight: 5000.0
Max volume: 2000.0

========== Algorithm Configuration ==========
Variant: islands_parallel
Threads: 4
Seed: 42
Crossover rate: 0.7
Mutation rate: 0.04
Tournament size: 3
Number of islands: 4
Migration interval: 5
====================================

=== Starting Genetic Algorithm ===
Population size: 120
Generations: 300
[Gen 0] Best fitness: 1234.5 (Feasible: Yes)
[Gen 10] Best fitness: 1456.8 (Feasible: Yes)
[Migration] Gen 5
[Gen 20] Best fitness: 1567.2 (Feasible: Yes)
...
[Gen 290] Best fitness: 1892.3 (Feasible: Yes)

=== Results ===
Best fitness: 1892.3
Feasible: Yes
Execution time: 2345 ms
==============================
```

---

## 🔍 Detalles Técnicos Importantes

### BaseGA::run() es FINAL
No se puede sobrescribir. Define el flujo exacto que todos los solvers siguen.

### do_reproduction() es PURO VIRTUAL
Cada solver DEBE implementarlo. Aquí es donde varía la lógica:
- **Sequential**: loop simple
- **Parallel**: #pragma omp parallel for
- **IslandsSequential**: loop sobre islas
- **IslandsParallel**: #pragma omp sobre islas

### initialize_population() es VIRTUAL
- **BaseGA**: inicializa `population`
- **Islands**: inicializa `islands[]` (múltiples poblaciones)

### evaluate_population() es VIRTUAL
- **BaseGA**: evalúa `population`
- **Islands**: evalúa cada `islands[i]` en paralelo

---

## 🧪 Próximos Pasos (Futuro)

1. **CUDA Solver**: Paralelización en GPU
   - Transferir población a GPU
   - Reproducción masivamente paralela
   - Evaluación en GPU

2. **Mejor Comunicación entre Islas**
   - Migración asincrónica
   - Ring topology vs. completa

3. **Adaptive Parameters**
   - Ajustar tasas según convergencia
   - Algoritmos PBIL o CMA-ES como alternativas

4. **Estadísticas Avanzadas**
   - Guardar histórico por isla
   - Análisis de convergencia
   - Visualización gráfica

---

## ✅ Validación de Implementación

- ✅ Compilación sin errores
- ✅ Patrón Template Method correcto
- ✅ Inyección de dependencias
- ✅ Factory pattern implementado
- ✅ 4 solvers funcionando
- ✅ CLI configurado
- ✅ Código escalable y mantenible
