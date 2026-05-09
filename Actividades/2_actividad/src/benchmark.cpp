#include "benchmark.hpp"
#include "genetic_algorithm.hpp"
#include "island_model.hpp"
#include "instance_loader.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>

#include <omp.h>

struct BenchmarkResult {
    std::string instance_name;
    int threads;
    double avg_time;
    double std_time;
    float best_feasible_value;
    float best_fitness;
    float feasible_percentage;
    double speedup;
    double efficiency;
};

void Benchmark::Run(const BenchmarkConfig& config) {
    std::cout << "=== Configuración del Benchmark ===\n";
    std::cout << "ID de Configuración: " << config.config_id << "\n";
    std::cout << "Variante: " << config.variant << "\n";
    
    std::cout << "Repeticiones: " << config.repetitions << "\n";
    std::cout << "Generaciones: " << config.generations << "\n";
    std::cout << "Tasa de mutación: " << config.mutation_rate << "\n";
    if (config.variant == "standard") {
        std::cout << "Población: " << config.population_size << "\n";
    } else {
        std::cout << "Número de islas: " << config.num_islands << "\n";
        std::cout << "Población por isla: " << config.population_per_island << "\n";
        std::cout << "Frecuencia de migración (en generaciones): " << config.migration_frequency << "\n";
        std::cout << "Cantidad de migrantes: " << config.num_migrants << "\n";
        std::cout << "Topología de migración: " << config.migration_topology << "\n";
    }
    std::cout << "Hilos a evaluar: ";
    for (int t : config.threads_list) std::cout << t << " ";
    std::cout << "\n===================================\n\n";

    std::ofstream report(config.report_file);
    if (report.is_open()) {
        report << "Instance,Threads,AvgTime(s),StdTime(s),BestFeasibleValue,BestFitness,Feasible%,Speedup,Efficiency\n";
    }

    std::ofstream detailed_report("detailed_benchmark.csv");
    if (detailed_report.is_open()) {
        detailed_report << "Instance,Threads,Repetition,Time(s),BestFeasibleValue,BestFitness,Feasible%\n";
    }

    std::vector<BenchmarkResult> all_results;

    for (const auto& inst_path : config.instances) {
        std::cout << "\n=== Benchmarking Instancia: " << inst_path << " ===\n";
        
        Instance instance;
        try {
            instance = InstanceLoader::load(inst_path);
        } catch(const std::exception& e) {
            std::cerr << "Error cargando " << inst_path << ": " << e.what() << "\n";
            continue;
        }
        
        double t1_time = 0.0; // Tiempo para 1 hilo

        for (int t : config.threads_list) {
            std::cout << "  -> Evaluando con " << t << " hilos...\n";
            
            std::vector<double> times(config.repetitions);
            float best_feat = -1.0f;
            float best_fit = -1e9f;
            long long total_valid_accumulated = 0;
            long long total_evaluated_accumulated = 0;

            for (int r = 0; r < config.repetitions; ++r) {
                int seed = 1000 + r * 13; // Semillas registradas y reproducibles

                Individual mejor;
                std::vector<GenerationStats> stats;
                double start_time = omp_get_wtime();

                if (config.variant == "islands") {
                    IslandModel ga(instance, config.num_islands, config.population_per_island,
                                   config.generations, config.mutation_rate, config.migration_frequency,
                                   config.num_migrants, config.migration_topology, seed);
                    if (t > 1) ga.RunParallel(t);
                    else ga.Run();
                    mejor = ga.GetBestSolution();
                    stats = ga.GetStats();
                } else {
                    GeneticAlgorithm ga(instance, config.population_size, config.generations,
                                        config.mutation_rate, seed);
                    if (t > 1) ga.RunParallel(t);
                    else ga.Run();
                    mejor = ga.GetBestSolution();
                    stats = ga.GetStats();
                }

                double end_time = omp_get_wtime();
                times[r] = end_time - start_time;
                best_fit = std::max(best_fit, mejor.fitness);
                
                long long valid_in_run = 0;
                for(const auto& s : stats) {
                    valid_in_run += s.valid_count;
                }
                total_valid_accumulated += valid_in_run;
                
                long long current_evaluated = 0;
                if (config.variant == "islands") {
                    current_evaluated = stats.size() * (config.num_islands * config.population_per_island);
                } else {
                    current_evaluated = stats.size() * config.population_size;
                }
                total_evaluated_accumulated += current_evaluated;

                float current_feat = -1.0f;
                if (mejor.is_valid) {
                    float val = 0.0f;
                    for (size_t i = 0; i < mejor.chromosome.size(); ++i) {
                        if (mejor.chromosome[i]) val += instance.items[i].value;
                    }
                    current_feat = val;
                    best_feat = std::max(best_feat, val);
                }

                double rep_feasible_percentage = (static_cast<double>(valid_in_run) / static_cast<double>(current_evaluated)) * 100.0;
                
                if (detailed_report.is_open()) {
                    detailed_report << inst_path << ","
                                    << t << ","
                                    << (r + 1) << ","
                                    << times[r] << ","
                                    << current_feat << ","
                                    << mejor.fitness << ","
                                    << rep_feasible_percentage << "\n";
                }
                
                std::cout << "    [Rep " << (r + 1) << "] Tiempo(s): " << std::fixed << std::setprecision(4) << times[r] 
                          << "s | bestFit: " << mejor.fitness 
                          << " | Factibles(%): " << rep_feasible_percentage << "%\n";
            }

            // Calcular estadisticas
            double sum_time = 0.0;
            for (double tm : times) sum_time += tm;
            double avg_t = sum_time / config.repetitions;

            double sum_sq_diff = 0.0;
            for (double tm : times) sum_sq_diff += (tm - avg_t) * (tm - avg_t);
            double std_t = std::sqrt(sum_sq_diff / config.repetitions);

            if (t == 1) {
                t1_time = avg_t;
            }

            double speedup = t1_time / avg_t;
            double efficiency = speedup / t;

            BenchmarkResult res;
            res.instance_name = inst_path;
            res.threads = t;
            res.avg_time = avg_t;
            res.std_time = std_t;
            res.best_feasible_value = best_feat;
            res.best_fitness = best_fit;
            res.feasible_percentage = (static_cast<double>(total_valid_accumulated) / static_cast<double>(total_evaluated_accumulated)) * 100.0;
            res.speedup = speedup;
            res.efficiency = efficiency;

            all_results.push_back(res);

            if (report.is_open()) {
                report << res.instance_name << ","
                       << res.threads << ","
                       << res.avg_time << ","
                       << res.std_time << ","
                       << res.best_feasible_value << ","
                       << res.best_fitness << ","
                       << res.feasible_percentage << ","
                       << res.speedup << ","
                       << res.efficiency << "\n";
            }
        }
    }
    
    std::cout << "\n=== Resultados Globales del Benchmark ===\n";
    std::cout << std::left << std::setw(15) << "Instancia" 
              << std::setw(8) << "Hilos" 
              << std::setw(12) << "Tiempo(s)" 
              << std::setw(12) << "Desv.Std(s)" 
              << std::setw(10) << "Speedup"
              << std::setw(10) << "Efic(%)" 
              << std::setw(12) << "MejorFact" 
              << std::setw(15) << "MejorFitn" 
              << "Factibles%\n";
    std::cout << std::string(105, '-') << "\n";
    for(const auto& r : all_results) {
        std::cout << std::left << std::setw(15) << r.instance_name 
                  << std::setw(8) << r.threads 
                  << std::setw(12) << r.avg_time 
                  << std::setw(12) << r.std_time 
                  << std::setw(10) << r.speedup
                  << std::setw(10) << (r.efficiency * 100.0)
                  << std::setw(12) << r.best_feasible_value 
                  << std::setw(15) << r.best_fitness 
                  << r.feasible_percentage << "%\n";
    }
}
