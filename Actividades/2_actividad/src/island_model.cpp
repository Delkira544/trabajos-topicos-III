#include "island_model.hpp"
#include "fitness.hpp"
#include "selection.hpp"
#include "crossover.hpp"
#include "mutation.hpp"
#include <iostream>
#include <algorithm>
#include <omp.h>

IslandModel::IslandModel(const Instance& inst, int islands_count, int pop_per_island, 
                         int gens, float mut_rate, int mig_freq, int n_migrants, 
                         const std::string& top, int s)
    : instance(inst), num_islands(islands_count), population_per_island(pop_per_island),
      generations(gens), mutation_rate(mut_rate), migration_frequency(mig_freq),
      num_migrants(n_migrants), topology(top), seed(s), convergence_threshold_(0.001f) {
    InitializeIslands();
}

void IslandModel::InitializeIslands() {
    islands.resize(num_islands, std::vector<Individual>(population_per_island));
    island_rngs.resize(num_islands);
    
    for (int i = 0; i < num_islands; ++i) {
        island_rngs[i].seed(seed + i);
        for (int p = 0; p < population_per_island; ++p) {
            islands[i][p].chromosome.resize(instance.items.size());
            std::uniform_int_distribution<> dis(0, 1);
            for (size_t j = 0; j < instance.items.size(); ++j) {
                islands[i][p].chromosome[j] = dis(island_rngs[i]);
            }
            Fitness::Evaluate(islands[i][p], instance, 0, generations);
        }
    }
}

Individual IslandModel::FindBestInIsland(int island_idx) const {
    Individual best = islands[island_idx][0];
    for (const auto& ind : islands[island_idx]) {
        if (ind.fitness > best.fitness) {
            best = ind;
        }
    }
    return best;
}

Individual IslandModel::FindBestOverall() const {
    Individual best = islands[0][0];
    for (int i = 0; i < num_islands; ++i) {
        for (const auto& ind : islands[i]) {
            if (ind.fitness > best.fitness) {
                best = ind;
            }
        }
    }
    return best;
}

void IslandModel::RecordStats(int gen) {
    GenerationStats stat;
    stat.generation = gen;
    stat.best_fitness = -1e9f;
    stat.worst_fitness = 1e9f;
    stat.valid_count = 0;
    
    float total_fitness = 0.0f;
    int total_pop = num_islands * population_per_island;
    
    Individual best_overall;
    bool first = true;
    
    for (int i = 0; i < num_islands; ++i) {
        for (const auto& ind : islands[i]) {
            if (first || ind.fitness > stat.best_fitness) {
                stat.best_fitness = ind.fitness;
                best_overall = ind;
                first = false;
            }
            if (ind.fitness < stat.worst_fitness) stat.worst_fitness = ind.fitness;
            total_fitness += ind.fitness;
            if (ind.is_valid) stat.valid_count++;
        }
    }
    
    stat.avg_fitness = total_fitness / total_pop;
    stat.best_is_valid = best_overall.is_valid;
    
    if (stats_.empty()) stat.convergence_delta = 0.0f;
    else stat.convergence_delta = stat.best_fitness - stats_.back().best_fitness;
    
    stats_.push_back(stat);
}

void IslandModel::Migrate() {
    std::vector<std::vector<Individual>> emigrants(num_islands);
    for (int i = 0; i < num_islands; ++i) {
        std::vector<Individual> sorted_island = islands[i];
        std::sort(sorted_island.begin(), sorted_island.end(),
                  [](const Individual& a, const Individual& b) { return a.fitness > b.fitness; });
        for (int m = 0; m < num_migrants; ++m) {
            emigrants[i].push_back(sorted_island[m]);
        }
    }
    
    for (int i = 0; i < num_islands; ++i) {
        int source_idx = 0;
        if (topology == "ring") {
            source_idx = (i - 1 + num_islands) % num_islands;
        } else { // random
            std::uniform_int_distribution<> dis(0, num_islands - 1);
            source_idx = dis(island_rngs[i]);
            while(source_idx == i && num_islands > 1) {
                source_idx = dis(island_rngs[i]);
            }
        }
        
        std::vector<Individual>& target_island = islands[i];
        std::sort(target_island.begin(), target_island.end(),
                  [](const Individual& a, const Individual& b) { return a.fitness > b.fitness; });
                  
        for (int m = 0; m < num_migrants; ++m) {
            target_island[target_island.size() - 1 - m] = emigrants[source_idx][m];
        }
    }
}

void IslandModel::Run() {
    RecordStats(0);
    
    for (int gen = 1; gen <= generations; ++gen) {
        for (int i = 0; i < num_islands; ++i) {
            std::vector<Individual> new_island;
            new_island.reserve(population_per_island);
            // Elitismo local
            new_island.push_back(FindBestInIsland(i));
            
            while ((int)new_island.size() < population_per_island) {
                Individual p1 = Selection::Tournament(islands[i], 3, island_rngs[i]);
                Individual p2 = Selection::Tournament(islands[i], 3, island_rngs[i]);
                
                Individual c1, c2;
                Crossover::SinglePoint(p1, p2, c1, c2, island_rngs[i]);
                
                Mutation::BitFlip(c1, mutation_rate, island_rngs[i]);
                Fitness::Evaluate(c1, instance, gen, generations);
                new_island.push_back(c1);
                
                if ((int)new_island.size() < population_per_island) {
                    Mutation::BitFlip(c2, mutation_rate, island_rngs[i]);
                    Fitness::Evaluate(c2, instance, gen, generations);
                    new_island.push_back(c2);
                }
            }
            islands[i] = std::move(new_island);
        }
        
        if (gen % migration_frequency == 0) {
            Migrate();
        }
        
        RecordStats(gen);
    }
}

void IslandModel::RunParallel(int num_threads) {
    if (num_threads > 0) omp_set_num_threads(num_threads);
    
    RecordStats(0);
    
    for (int gen = 1; gen <= generations; ++gen) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_islands; ++i) {
            std::vector<Individual> new_island;
            new_island.reserve(population_per_island);
            // Elitismo local
            new_island.push_back(FindBestInIsland(i));
            
            while ((int)new_island.size() < population_per_island) {
                Individual p1 = Selection::Tournament(islands[i], 3, island_rngs[i]);
                Individual p2 = Selection::Tournament(islands[i], 3, island_rngs[i]);
                
                Individual c1, c2;
                Crossover::SinglePoint(p1, p2, c1, c2, island_rngs[i]);
                
                Mutation::BitFlip(c1, mutation_rate, island_rngs[i]);
                Fitness::Evaluate(c1, instance, gen, generations);
                new_island.push_back(c1);
                
                if ((int)new_island.size() < population_per_island) {
                    Mutation::BitFlip(c2, mutation_rate, island_rngs[i]);
                    Fitness::Evaluate(c2, instance, gen, generations);
                    new_island.push_back(c2);
                }
            }
            islands[i] = std::move(new_island);
        }
        
        if (gen % migration_frequency == 0) {
            Migrate();
        }
        
        RecordStats(gen);
    }
}

void IslandModel::View_Population() const {
    std::cout << "Poblaciones de Islas no mostradas detalladamente aún.\n";
}