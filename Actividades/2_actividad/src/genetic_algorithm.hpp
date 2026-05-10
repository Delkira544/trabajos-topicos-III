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

using DependencyMap = std::vector<std::pair<int, int>>; // (id_item, id_requerido)

struct KnapsackConfig {
    float max_weight;
    float max_volume;
};

struct PenaltyConfig {
    // wᵢ: pesos de cada restricción dentro de violacion_norm (Σwᵢ = 1)
    float alpha = 0.1f;   // w₁: exceso de peso
    float beta = 0.1f;    // w₂: exceso de volumen
    float gamma = 0.2f;   // w₃: errores de categoría
    float delta = 0.3f;   // w₄: incompatibilidades
    float epsilon = 0.3f; // w₅: dependencias
    // α y β del modelo: fitness = α·valor_norm − β·violacion_norm (α+β=1)
    float obj_weight = 0.6f; // α: importancia del valor objetivo
    float pen_weight = 0.4f; // β: importancia de la violación (1−α)
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
    float penalty; // violacion_norm ∈ [0,1]: 0 = sin violaciones
    bool is_valid;

    Individual() : fitness(0.0f), penalty(1.0f), is_valid(false) {
    }
};

// Válido gana a inválido; ambos inválidos → menor penalización (más factible);
// ambos válidos → mayor fitness.
inline bool IsBetter(const Individual &a, const Individual &b) {
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

    void Initialize_Population();
    Individual FindBest(const std::vector<Individual> &pop) const;
    std::mt19937 get_rng_for_thread(int thread_id) const;
    void RecordStats(int gen);
    bool HasConverged() const;

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
