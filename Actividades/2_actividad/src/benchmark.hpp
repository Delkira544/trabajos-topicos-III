#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP

#include <string>
#include <vector>
#include <iostream>

struct BenchmarkConfig {
    int config_id = 1;
    std::vector<std::string> instances;
    std::vector<int> threads_list;
    int repetitions;
    std::string variant; // "standard" o "islands"
    int population_size = 100;
    int generations = 50;
    float mutation_rate = 0.03f;
    int num_islands = 4;
    int population_per_island = 25;
    int migration_frequency = 10;
    int num_migrants = 2;
    std::string migration_topology = "ring";
    std::string report_file = "results/benchmark_results.csv";
    bool verbose = false;
    int base_seed = 0;
};

class Benchmark {
public:
    static void Run(const BenchmarkConfig& config);
};

#endif
