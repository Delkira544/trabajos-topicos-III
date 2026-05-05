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
    float weight_penalty = 2.0f;
    float volume_penalty = 2.0f;
    float incompatibility_penalty = 0.15f;
    float dependency_penalty = 0.15f;
    float category_penalty = 0.1f;
};

struct Instance {
    std::vector<Item> items;
    CategoryMap category_rules;
    std::vector<Incompatibility> incompatibilities;
    DependencyMap dependencies;
    KnapsackConfig knapsack;
    PenaltyConfig penalties;
};

struct Individual {
    std::vector<bool> chromosome;
    float fitness;
    bool is_valid;

    Individual() : fitness(0.0), is_valid(true) {
    }
};

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
