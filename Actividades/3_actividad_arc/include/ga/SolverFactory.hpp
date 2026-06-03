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
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

/**
 * @brief Factory pattern para crear solvers del Algoritmo Genético
 *
 * Esta clase es responsable de:
 * - Crear los componentes (operadores)
 * - Crear el solver apropiado basado en el tipo solicitado
 * - Inyectar las dependencias
 *
 * Uso:
 * auto solver = SolverFactory::create("sequential", instance, config);
 */
class SolverFactory
{
  public:
  /**
   * @brief Configuración para crear solvers
   */
  struct SolverConfig
  {
    int num_threads        = 1;
    int num_islands        = 4;
    int migration_interval = 5;
    float crossover_rate   = 0.7f;
    float mutation_rate    = 0.04f;
    int tournament_size    = 3;
    int seed               = 0;
    bool verbose           = false;
  };

  /**
   * @brief Crea un solver del tipo especificado
   *
   * @param variant Tipo de solver: "sequential", "parallel",
   *                "islands_sequential", "islands_parallel"
   * @param instance Instancia del problema
   * @param config Configuración del solver
   * @return Unique pointer al solver creado
   * @throws std::invalid_argument Si el variant no es reconocido
   */
  static std::unique_ptr<GeneticSolver> create(const std::string& variant,
                                               KnapsackInstance& instance,
                                               const SolverConfig& config)
  {
    std::cout << "Creating solver: " << variant << std::endl;

    // Crear componentes (operadores) inyectables
    auto crossover = std::make_unique<ga::operators::SinglePointCrossover>(
      config.crossover_rate);
    auto mutation =
      std::make_unique<ga::operators::UniformMutation>(config.mutation_rate);
    auto selection = std::make_unique<ga::operators::TournamentSelection>(
      config.tournament_size);
    auto fitness   = std::make_unique<ga::operators::KnapsackFitness>();
    auto validator = std::make_unique<ga::operators::KnapsackValidator>();

    // Crear el solver apropiado
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
    else if (variant == "cuda")
    {
      throw std::invalid_argument("CUDA solver not yet implemented");
    }
    else
    {
      throw std::invalid_argument("Unknown solver variant: " + variant);
    }
  }

  /**
   * @brief Lista las variantes disponibles
   */
  static void print_available_variants()
  {
    std::cout << "Available solver variants:" << std::endl;
    std::cout << "  - sequential      : Single-threaded execution" << std::endl;
    std::cout << "  - parallel        : Multi-threaded with OpenMP"
              << std::endl;
    std::cout << "  - islands_sequential : Multiple populations (sequential)"
              << std::endl;
    std::cout << "  - islands_parallel   : Multiple populations (parallel)"
              << std::endl;
  }
};
