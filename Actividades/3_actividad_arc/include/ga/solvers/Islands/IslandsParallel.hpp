#pragma once
#include "ga/BaseGA.hpp"
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <thread>
#include <vector>

/**
 * @brief Implementación paralela de Algoritmos Genéticos con Islas
 *
 * Cada isla evoluciona en su propio thread, con sincronización periódica
 * para migración.
 *
 * Características:
 * - Paralelización a nivel de islas
 * - Cada isla evoluciona de forma independiente
 * - Migración sincronizada cada N generaciones
 * - Máximo paralelismo para problemas grandes
 */
class IslandsParallel : public BaseGA
{
  private:
  int num_islands;
  int num_threads;
  std::vector<std::vector<Individual>> islands;
  size_t migration_interval;

  protected:
  /**
   * @brief Evoluciona una isla específica
   */
  void evolve_island(int island_id)
  {
    auto& island_pop = islands[island_id];
    std::vector<Individual> elite = island_pop;
    std::sort(elite.begin(), elite.end(),
              [](const Individual& a, const Individual& b) {
                return a.fitness > b.fitness;
              });

    std::vector<Individual> local_offspring;
    local_offspring.reserve(island_pop.size());

    // Cada thread con su propio RNG
    std::mt19937 local_rng(rng() + island_id);

    while (local_offspring.size() < island_pop.size())
    {
      // Seleccionar dos padres de esta isla
      auto [parent1, parent2] =
        selection_op->select_pair(island_pop, local_rng);

      // Cruzamiento
      Individual child = crossover_op->apply(parent1, parent2, local_rng);

      // Mutación
      mutation_op->apply(child, local_rng);

      // Reparación si es necesario
      if (!validator->is_feasible(child, instance))
      {
        validator->repair(child, instance);
      }

      local_offspring.push_back(child);
    }

    apply_elitism(elite, local_offspring);

    // Reemplazar población de la isla
    islands[island_id] = local_offspring;
  }

  void do_reproduction() override
  {
    // Paralelizar evolución de islas
#pragma omp parallel for num_threads(num_islands) schedule(static)
    for (int i = 0; i < num_islands; ++i)
    {
      evolve_island(i);
    }
    // Barrera implícita aquí con omp parallel for
  }

  void do_migration() override
  {
    // Migración cada N generaciones
    if (current_generation % migration_interval == 0 && current_generation > 0)
    {
      migrate_individuals();
    }
  }

  /**
   * @brief Implementa la estrategia de migración
   */
  void migrate_individuals()
  {
    if (num_islands < 2)
      return;

    std::cout << "[Migration] Gen " << current_generation << std::endl;

    // Encontrar mejores individuos de cada isla
    std::vector<size_t> best_indices(num_islands);

#pragma omp parallel for num_threads(num_islands)
    for (int i = 0; i < num_islands; ++i)
    {
      best_indices[i] = 0;
      for (size_t j = 1; j < islands[i].size(); ++j)
      {
        if (islands[i][j].fitness > islands[i][best_indices[i]].fitness)
        {
          best_indices[i] = j;
        }
      }
    }

    // Intercambiar mejores individuos en forma cíclica (con sincronización crítica)
    #pragma omp critical
    {
      Individual temp = islands[0][best_indices[0]];
      for (int i = 0; i < num_islands - 1; ++i)
      {
        islands[i][best_indices[i]] = islands[i + 1][best_indices[i + 1]];
      }
      islands[num_islands - 1][best_indices[num_islands - 1]] = temp;
    }
  }

  void initialize_population() override
  {
    // Inicializar islas
    islands.clear();
    islands.resize(num_islands);

    size_t island_size = population_size / num_islands;

    for (int i = 0; i < num_islands; ++i)
    {
      for (size_t j = 0; j < island_size; ++j)
      {
        islands[i].push_back(CreateRandomIndividual(rng));
      }
    }

    // Llenar la población base con todos los individuos
    population.clear();
    for (const auto& island : islands)
    {
      population.insert(population.end(), island.begin(), island.end());
    }
  }

  void evaluate_population() override
  {
    // Paralelizar evaluación de islas
#pragma omp parallel for num_threads(num_islands)
    for (int i = 0; i < num_islands; ++i)
    {
      fitness_evaluator->evaluate_population(islands[i], instance);
    }

    // Mantener sincronizada la población base
    population.clear();
    for (const auto& island : islands)
    {
      population.insert(population.end(), island.begin(), island.end());
    }
  }

  public:
  IslandsParallel(
    KnapsackInstance& inst, int num_isl, int num_th, int mig_interval = 5,
    std::unique_ptr<ga::operators::Crossover> cross         = nullptr,
    std::unique_ptr<ga::operators::Mutation> mut            = nullptr,
    std::unique_ptr<ga::operators::Selection> sel           = nullptr,
    std::unique_ptr<ga::operators::FitnessEvaluator> fit    = nullptr,
    std::unique_ptr<ga::operators::ConstraintValidator> val = nullptr,
    bool verbose = false, int seed = 0,
    size_t pop_size = 0, size_t num_gens = 0)
    : BaseGA(inst, std::move(cross), std::move(mut), std::move(sel),
             std::move(fit), std::move(val), verbose, seed, pop_size, num_gens),
      num_islands(num_isl), num_threads(num_th),
      migration_interval(mig_interval)
  {
  }

  virtual ~IslandsParallel() = default;
};
