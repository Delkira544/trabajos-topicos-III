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

using DependencyMap = std::unordered_map<int, int>;

struct KnapsackConfig {
    float max_weight;
    float max_volume;
};

struct PenaltyConfig {
    float alpha = 0.1f;   // Peso (convexo) para exceso de peso
    float beta = 0.2f;    // Peso (convexo) para exceso de volumen
    float gamma = 0.3f;   // Peso (convexo) para errores de categoría
    float delta = 0.1f;   // Peso (convexo) para incompatibilidades
    float epsilon = 0.3f; // Peso (convexo) para dependencias faltantes
};

struct Instance {
    std::vector<Item> items;
    CategoryMap category_rules;
    std::vector<Incompatibility> incompatibilities;
    DependencyMap dependencies;
    KnapsackConfig knapsack;
    PenaltyConfig penalties;
    float max_value = 0.0f;
};

struct Individual {
    std::vector<bool> chromosome;
    float fitness;
    bool is_valid;

    Individual() : fitness(0.0), is_valid(true) {
    }
};

// Válido siempre gana a inválido; entre iguales en validez gana mayor fitness.
inline bool IsBetter(const Individual &a, const Individual &b) {
    if (a.is_valid != b.is_valid) return a.is_valid;
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
    float base_mutation_rate_;
    float current_mutation_rate_;
    int generations_since_improvement_;
    static constexpr int STALL_LIMIT = 200;
    static constexpr float BOOST_RATE = 0.05f;
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
