#ifndef ISLAND_MODEL_HPP
#define ISLAND_MODEL_HPP

#include "genetic_algorithm.hpp"
#include <vector>
#include <string>
#include <random>

class IslandModel {
private:
    Instance instance;
    int num_islands;
    int population_per_island;
    int generations;
    float mutation_rate;
    int migration_frequency;
    int num_migrants;
    std::string topology;
    int seed;
    float convergence_threshold_;

    std::vector<std::vector<Individual>> islands;
    std::vector<std::mt19937> island_rngs;
    std::vector<GenerationStats> stats_;

    void InitializeIslands();
    void Migrate();
    Individual FindBestOverall() const;
    Individual FindBestInIsland(int island_idx) const;
    void RecordStats(int gen);
    void EvaluateIsland(int island_idx, int gen);

public:
    IslandModel(const Instance& inst, int islands_count, int pop_per_island, 
                int gens, float mut_rate, int mig_freq, int n_migrants, 
                const std::string& top, int s);

    void Run();
    void RunParallel(int num_threads = 0);
    
    void SetConvergenceThreshold(float threshold) { convergence_threshold_ = threshold; }
    const std::vector<GenerationStats>& GetStats() const { return stats_; }
    Individual GetBestSolution() const { return FindBestOverall(); }
    void View_Population() const;
};

#endif