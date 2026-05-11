#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP

#include <string>
#include <vector>
#include <iostream>

// =============================================================================
// Benchmark — suite estadística para comparar variantes y configuraciones
// =============================================================================
// Define el contenedor de parámetros (`BenchmarkConfig`) y la clase estática
// `Benchmark` que ejecuta una matriz de experimentos:
//
//   instancias × hilos × repeticiones
//
// produciendo dos CSV:
//   - resumen (configurable)           con stats agregadas por (instancia, hilos)
//   - results/detailed_benchmark.csv   con una fila por repetición individual
//
// Métricas reportadas: tiempo promedio, desviación estándar, mejor valor
// factible, mejor fitness, % factibles, speedup (T1/Tp), eficiencia (Sp/p).
// =============================================================================

/**
 * @brief Bundle de configuración para una corrida de benchmark.
 *
 * Los campos se sobreescriben en `main.cpp` según el `--config <n>` seleccionado
 * (1..4) y la `--variant` indicada. Los defaults aquí son sólo sane fallbacks.
 */
struct BenchmarkConfig {
    int config_id = 1;                          ///< Identificador 1..4 (para registro)
    std::vector<std::string> instances;         ///< Rutas a directorios de instancia
    std::vector<int> threads_list;              ///< {1, 2, 4, 8} típicamente
    int repetitions;                            ///< Reps por (instancia, hilos)
    std::string variant;                        ///< "standard" o "islands"

    // Parámetros del AG estándar
    int population_size = 100;
    int generations = 50;
    float mutation_rate = 0.03f;

    // Parámetros del modelo de islas
    int num_islands = 4;
    int population_per_island = 25;
    int migration_frequency = 10;
    int num_migrants = 2;
    std::string migration_topology = "ring";

    // Salida y reproducibilidad
    std::string report_file = "results/benchmark_results.csv";
    bool verbose = false;
    int base_seed = 0;                          ///< seed_r = base_seed + r·13
};

/**
 * @brief Ejecutor estático de la suite de benchmark.
 *
 * No requiere instanciación: toda la lógica vive en `Run(config)`.
 */
class Benchmark {
public:
    /**
     * @brief Ejecuta el matriz de experimentos descrita en `config`.
     *
     * Para cada (instancia, nº de hilos, repetición):
     *   1. Construye un AG nuevo (semilla derivada de `base_seed + r·13`).
     *   2. Llama Run() si hilos=1, RunParallel(t) si hilos>1.
     *   3. Mide tiempo con `omp_get_wtime()`.
     *   4. Acumula métricas y escribe filas al CSV detallado.
     * Al cerrar cada bloque (instancia, hilos), calcula avg/std/speedup/eficiencia
     * y escribe al CSV resumen.
     */
    static void Run(const BenchmarkConfig& config);
};

#endif
