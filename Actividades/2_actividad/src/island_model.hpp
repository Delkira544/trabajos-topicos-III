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

    // ── Mejoras integradas del AG estándar ─────────────────────────
    // #4 Uniform crossover con sesgo a no incluir
    float crossover_bias_ = 0.45f;
    // #5 Mutación dirigida a violaciones soft (probabilidad por hijo)
    float targeted_fix_prob_ = 0.5f;
    // #7 Multi-elite del 10% dentro de cada isla
    float elite_fraction_ = 0.10f;
    // #8 Tournament k=5
    int   tournament_size_ = 5;

    // #6 Anti-estancamiento global (migración ya aporta diversidad → límite mayor)
    int   stagnation_limit_ = 100;
    float diversity_inject_fraction_ = 0.30f;
    int   gens_without_improvement_ = 0;
    Individual best_ever_;

    // #12 Early-stop por estancamiento total (NO se resetea con injection)
    int   near_feasible_max_violations_ = 0;
    int   near_feasible_stall_limit_ = 250;
    int   gens_no_improve_total_ = 0;

    void InitializeIslands();
    void Migrate();
    Individual FindBestOverall() const;
    Individual FindBestInIsland(int island_idx) const;
    void RecordStats(int gen);

    // Helpers nuevos
    Individual CreateRandomIndividual(std::mt19937 &r) const;
    void EvolveIsland(int idx, int gen);
    void InjectDiversityAllIslands();
    int  CountTotalViolations(const Individual &ind) const;

public:
    IslandModel(const Instance& inst, int islands_count, int pop_per_island,
                int gens, float mut_rate, int mig_freq, int n_migrants,
                const std::string& top, int s);

    void Run();
    void RunParallel(int num_threads = 0);

    void SetConvergenceThreshold(float threshold) { convergence_threshold_ = threshold; }
    bool HasConverged() const;
    const std::vector<GenerationStats>& GetStats() const { return stats_; }
    Individual GetBestSolution() const { return FindBestOverall(); }
    void View_Population() const;
};

#endif
