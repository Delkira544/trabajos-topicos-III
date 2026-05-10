#ifndef GENETIC_ALGORITHM_HPP
#define GENETIC_ALGORITHM_HPP

#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <omp.h>

struct Item {
    int id;
    float value;
    float weight;
    float volume;
    std::string category;
};

struct CategoryRule {
    int min;
    int max;
};

using CategoryMap = std::unordered_map<std::string, CategoryRule>;

struct Incompatibility {
    int id_a;
    int id_b;
};

using DependencyMap =
    std::vector<std::pair<int, int>>; // (id_item, id_requerido)

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

struct Instance {
    std::vector<Item> items;
    CategoryMap category_rules;
    std::vector<Incompatibility> incompatibilities;
    DependencyMap dependencies;
    KnapsackConfig knapsack;
    PenaltyConfig penalties;
    float max_value = 0.0f;
    float max_excess_weight = 0.0f; // Pᵢ_max real: Σpesos − max_weight
    float max_excess_volume = 0.0f; // Pᵢ_max real: Σvolúmenes − max_volume
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
    int valid_count;
    bool best_is_valid;
    float convergence_delta;
};

class GeneticAlgorithm {
  private:
    Instance instance;
    int population_size;
    int generations;
    float mutation_rate;
    int seed;
    std::mt19937 rng;

    std::vector<Individual> population;
    std::vector<GenerationStats> stats_;
    float previous_best_fitness_;
    float convergence_threshold_;

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

    void Initialize_Population();
    Individual FindBest(const std::vector<Individual> &pop) const;
    std::mt19937 get_rng_for_thread(int thread_id) const;
    void RecordStats(int gen);
    bool HasConverged() const;
    Individual CreateRandomIndividual(std::mt19937 &r) const;
    void InjectDiversity(std::mt19937 &r);
    int CountTotalViolations(const Individual &ind) const;

  public:
    GeneticAlgorithm(const Instance &instance, int population_size,
                     int generations, float mutation_rate, int seed);

    void View_Population();
    void Run();
    void RunParallel(int num_threads = 0);
    Individual GetBestSolution() const;
    const std::vector<GenerationStats> &GetStats() const;
    void SetConvergenceThreshold(float threshold);
};

#endif
