#pragma once
#include "ga/GeneticSolver.hpp"
#include "ga/operators/implementations/KnapsackFitness.hpp"
#include "ga/operators/implementations/KnapsackValidator.hpp"
#include "ga/operators/implementations/SinglePointCrossover.hpp"
#include "ga/operators/implementations/TournamentSelection.hpp"
#include "ga/operators/implementations/UniformMutation.hpp"
#include "ga/solvers/Islands/IslandsParallel.hpp"
#include "ga/solvers/Islands/IslandsSequential.hpp"
#include "ga/solvers/Parallel.hpp"
#include "ga/solvers/Sequential.hpp"
#include "ga/solvers/CUDABasic.cuh"
#include "ga/solvers/CUDAOptimized.cuh"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

class SolverFactory
{
  public:
  struct SolverConfig
  {
    int num_threads        = 1;
    int num_islands        = 4;
    int migration_interval = 5;
    float crossover_rate   = 0.7f;
    float mutation_rate    = 0.04f;
    int tournament_size    = 3;
    int block_size         = 128;   // tamaño de bloque CUDA
    int population_size    = 100;   // Tamaño de la población
    int num_generations    = 300;   // Número total de generaciones
    float penalty_weight   = 0.2f;  // Penalización por exceso de peso
    float penalty_volume   = 0.2f;  // Penalización por exceso de volumen
    float penalty_category = 0.2f;  // Penalización por violación de categoría
    float penalty_incomp   = 0.2f;  // Penalización por incompatibilidad
    float penalty_dep      = 0.2f;  // Penalización por dependencia
    int seed               = 0;
    bool verbose           = false;
  };

  static std::unique_ptr<GeneticSolver> create(const std::string& variant,
                                               KnapsackInstance& instance,
                                               const SolverConfig& config)
  {
    std::cout << "Creating solver: " << variant << std::endl;

    auto crossover  = std::make_unique<ga::operators::SinglePointCrossover>(config.crossover_rate);
    auto mutation   = std::make_unique<ga::operators::UniformMutation>(config.mutation_rate);
    auto selection  = std::make_unique<ga::operators::TournamentSelection>(config.tournament_size);
    auto fitness    = std::make_unique<ga::operators::KnapsackFitness>();
    auto validator  = std::make_unique<ga::operators::KnapsackValidator>();

    if (variant == "sequential")
    {
      return std::make_unique<Sequential>(
        instance, std::move(crossover), std::move(mutation),
        std::move(selection), std::move(fitness), std::move(validator),
        config.verbose, config.seed);
    }
    else if (variant == "parallel")
    {
      return std::make_unique<Parallel>(
        instance, config.num_threads, std::move(crossover), std::move(mutation),
        std::move(selection), std::move(fitness), std::move(validator),
        config.verbose, config.seed);
    }
    else if (variant == "islands_sequential")
    {
      return std::make_unique<IslandsSequential>(
        instance, config.num_islands, config.migration_interval,
        std::move(crossover), std::move(mutation), std::move(selection),
        std::move(fitness), std::move(validator), config.verbose, config.seed);
    }
    else if (variant == "islands_parallel")
    {
      return std::make_unique<IslandsParallel>(
        instance, config.num_islands, config.num_threads,
        config.migration_interval, std::move(crossover), std::move(mutation),
        std::move(selection), std::move(fitness), std::move(validator),
        config.verbose, config.seed);
    }
    else if (variant == "cuda_basic")
    {
      auto solver = std::make_unique<CUDABasic>(
        instance, std::move(crossover), std::move(mutation),
        std::move(selection), std::move(fitness), std::move(validator),
        config.verbose, config.seed, config.block_size);
      solver->set_penalty_weights(config.penalty_weight, config.penalty_volume,
                                   config.penalty_category, config.penalty_incomp,
                                   config.penalty_dep);
      return solver;
    }
    else if (variant == "cuda_optimized")
    {
      auto solver = std::make_unique<CUDAOptimized>(
        instance, std::move(crossover), std::move(mutation),
        std::move(selection), std::move(fitness), std::move(validator),
        config.verbose, config.seed, config.block_size);
      solver->set_penalty_weights(config.penalty_weight, config.penalty_volume,
                                   config.penalty_category, config.penalty_incomp,
                                   config.penalty_dep);
      return solver;
    }
    else
    {
      throw std::invalid_argument("Unknown solver variant: " + variant);
    }
  }

  static void print_available_variants()
  {
    std::cout << "Available solver variants:"          << std::endl;
    std::cout << "  - sequential          : Single-threaded CPU"     << std::endl;
    std::cout << "  - parallel            : Multi-threaded (OpenMP)" << std::endl;
    std::cout << "  - islands_sequential  : Islands (sequential)"    << std::endl;
    std::cout << "  - islands_parallel    : Islands (parallel)"      << std::endl;
    std::cout << "  - cuda_basic          : CUDA básico (Variante 2)"    << std::endl;
    std::cout << "  - cuda_optimized      : CUDA optimizado (Variante 3)" << std::endl;
  }
};