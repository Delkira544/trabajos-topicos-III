#ifndef GENETIC_ALGORITHM_HPP
#define GENETIC_ALGORITHM_HPP

#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <omp.h>

// =============================================================================
// genetic_algorithm.hpp — Estructuras del dominio + clase del AG estándar
// =============================================================================
// Este header define TODAS las estructuras de datos del proyecto:
//   - Datos del problema (Item, CategoryRule, Incompatibility, etc.)
//   - Configuración (KnapsackConfig, PenaltyConfig, Instance)
//   - Solución (Individual, GenerationStats)
//   - Comparador lexicográfico (IsBetter) — usado por TODOS los selectores
//
// También declara la clase `GeneticAlgorithm`, que implementa la variante
// estándar del AG en dos modos: secuencial (`Run`) y paralelo OpenMP
// (`RunParallel`). El modelo de islas (clase `IslandModel`, en otro archivo)
// reutiliza estos tipos.
// =============================================================================

/// Ítem disponible para la mochila. ID, valor, peso, volumen y categoría.
struct Item {
    int id;
    float value;
    float weight;
    float volume;
    std::string category;
};

/// Cantidades mínima/máxima permitidas de ítems en una categoría.
struct CategoryRule {
    int min;
    int max;
};

/// Mapa categoría → regla. Usado en Instance.
using CategoryMap = std::unordered_map<std::string, CategoryRule>;

/// Par incompatible: dos ítems que no pueden estar seleccionados simultáneamente.
struct Incompatibility {
    int id_a;
    int id_b;
};

/// Lista de dependencias. Cada pair = (id_item, id_requerido).
/// Si id_item está seleccionado, id_requerido también debe estarlo.
using DependencyMap =
    std::vector<std::pair<int, int>>;

/// Capacidades de la mochila (restricciones duras).
struct KnapsackConfig {
    float max_weight;
    float max_volume;
};

struct PenaltyConfig {
    // wᵢ: pesos de cada restricción dentro de violacion_norm (Σwᵢ = 1)
    // Por defecto las restricciones HARD (peso, volumen) tienen el mayor peso
    // dentro de la violación normalizada.
    float alpha = 0.30f;   // w₁: exceso de peso (HARD)
    float beta = 0.30f;    // w₂: exceso de volumen (HARD)
    float gamma = 0.05f;   // w₃: errores de categoría
    float delta = 0.20f;   // w₄: incompatibilidades
    float epsilon = 0.15f; // w₅: dependencias
    // α y β del modelo: fitness = α·valor_norm − β·violacion_norm (α+β=1)
    // pen_weight alto → fuerte presión de factibilidad sobre el valor objetivo.
    float obj_weight = 0.3f; // α
    float pen_weight = 0.7f; // β
};

/**
 * @brief Bundle inmutable de TODOS los datos del problema.
 *
 * Construido una sola vez por `InstanceLoader::load()`. Los campos
 * `max_*` son pre-cálculos auxiliares usados para normalizar en
 * `Fitness::Evaluate`.
 */
struct Instance {
    std::vector<Item> items;                          ///< Catálogo completo
    CategoryMap category_rules;                       ///< Cuotas por categoría
    std::vector<Incompatibility> incompatibilities;   ///< Pares incompatibles
    DependencyMap dependencies;                       ///< Dependencias item→requerido
    KnapsackConfig knapsack;                          ///< Capacidades W, V
    PenaltyConfig penalties;                          ///< Pesos α..ε + obj/pen weights
    float max_value = 0.0f;                           ///< Σ valor de todos los ítems
    float max_excess_weight = 0.0f;                   ///< Σ pesos − max_weight
    float max_excess_volume = 0.0f;                   ///< Σ volúmenes − max_volume
};

struct Individual {
    std::vector<bool> chromosome;
    float fitness;
    float penalty;      // violacion_norm ∈ [0,1]: 0 = sin violaciones
    bool is_valid;      // todas las restricciones (hard + soft) satisfechas
    bool hard_feasible; // peso y volumen dentro de capacidad

    Individual()
        : fitness(0.0f), penalty(1.0f), is_valid(false), hard_feasible(false) {
    }
};

// Orden lexicográfico (mejora #3):
//   1) hard-factible (peso+volumen) gana siempre — son restricciones DURAS
//   2) válido total (incluye soft) gana sobre inválido
//   3) ambos inválidos → menor penalización gana
//   4) ambos válidos → mayor fitness gana
inline bool IsBetter(const Individual &a, const Individual &b) {
    if (a.hard_feasible != b.hard_feasible) return a.hard_feasible;
    if (a.is_valid != b.is_valid) return a.is_valid;
    if (!a.is_valid) return a.penalty < b.penalty;
    return a.fitness > b.fitness;
}

struct GenerationStats {
    int generation;
    float best_fitness;
    float avg_fitness;
    float worst_fitness;
    float std_fitness;   // desviación estándar del fitness de la población
    int valid_count;
    bool best_is_valid;
    float convergence_delta;
};

/**
 * @brief Algoritmo Genético estándar para la mochila extendida.
 *
 * Implementa dos modos de ejecución equivalentes funcionalmente:
 *   - `Run()`         secuencial puro.
 *   - `RunParallel(t)` paralelo con OpenMP en 3 zonas críticas.
 *
 * El AG incluye todas las mejoras de diseño:
 *   #1 Penalización cuadrática para hard constraints
 *   #3 IsBetter lexicográfico
 *   #4 Uniform crossover con sesgo
 *   #5 BitFlipAsymmetric (mutación consciente de capacidad)
 *   #6 Anti-estancamiento por inyección de diversidad
 *   #7 Elitismo del top 10%
 *   #8 Tournament k=5
 *   #11 TargetedFix (mutación dirigida a violaciones)
 *   #12 Parada anticipada al alcanzar factibilidad + ventana de refinamiento
 */
class GeneticAlgorithm {
  private:
    // ── Parámetros del problema ───────────────────────────────────────────
    Instance instance;          ///< Datos del problema (referencia local)
    int population_size;        ///< Tamaño total de la población
    int generations;            ///< Máximo de generaciones (techo duro)
    float mutation_rate;        ///< Tasa base de mutación (0..1)
    int seed;                   ///< Semilla del RNG principal
    std::mt19937 rng;           ///< Generador aleatorio (sembrado con `seed`)

    // ── Estado del bucle evolutivo ────────────────────────────────────────
    std::vector<Individual> population;             ///< Población actual
    std::vector<GenerationStats> stats_;            ///< Stats por generación
    float previous_best_fitness_;                   ///< Para calcular Δ entre gens
    float convergence_threshold_;                   ///< Umbral de Δ promedio

    // Mejora #7 — elitismo del 10% del top (en vez de un único campeón)
    float elite_fraction_ = 0.10f;
    // Mejora #6 — inyección de diversidad ante estancamiento prolongado
    int stagnation_limit_ = 50;
    float diversity_inject_fraction_ = 0.30f;
    int gens_without_improvement_ = 0; // resetea con injection
    Individual best_ever_;

    // Early-stopping inteligente: si el mejor está cerca de factibilidad
    // (≤ N violaciones totales) y no mejora en M gens, abortar.
    // Este contador NO se resetea al inyectar diversidad — sólo al mejorar.
    int near_feasible_max_violations_ = 0;
    int near_feasible_stall_limit_ = 250;
    int gens_no_improve_total_ = 0;

    // Early-stop por "factibilidad alcanzada + N gens de refinamiento":
    // Cuando best_ever_ se vuelve válido por primera vez, registramos la gen.
    // 75 generaciones después (independientemente de si mejora o no), salimos.
    int gens_after_feasible_limit_ = 75;
    int first_feasible_gen_ = -1; // -1 = aún no encontrada

    void Initialize_Population();
    Individual FindBest(const std::vector<Individual> &pop) const;
    std::mt19937 get_rng_for_thread(int thread_id) const;
    void RecordStats(int gen);
    bool HasConverged() const;
    Individual CreateRandomIndividual(std::mt19937 &r) const;
    void InjectDiversity(std::mt19937 &r);
    int CountTotalViolations(const Individual &ind) const;

  public:
    /**
     * @brief Constructor — sólo configura, NO inicializa la población.
     * La inicialización ocurre dentro de Run() / RunParallel().
     */
    GeneticAlgorithm(const Instance &instance, int population_size,
                     int generations, float mutation_rate, int seed);

    /// Imprime stats resumidas de la población actual (debug).
    void View_Population();

    /// Ejecuta el bucle evolutivo en modo secuencial (un solo hilo).
    void Run();

    /**
     * @brief Ejecuta el bucle en modo paralelo con OpenMP.
     * @param num_threads  Si > 0, llama a omp_set_num_threads(num_threads).
     *                     Si 0, OpenMP usa su default (típicamente OMP_NUM_THREADS).
     */
    void RunParallel(int num_threads = 0);

    /// Devuelve el mejor individuo de la población final (no `best_ever_`).
    Individual GetBestSolution() const;

    /// Acceso de sólo lectura al log de stats por generación.
    const std::vector<GenerationStats> &GetStats() const;

    /// Ajusta el umbral de Δ usado por HasConverged(). Default: 0.001.
    void SetConvergenceThreshold(float threshold);
};

#endif
