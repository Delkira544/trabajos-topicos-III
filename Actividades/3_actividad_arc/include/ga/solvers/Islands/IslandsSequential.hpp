#pragma once
#include "ga/BaseGA.hpp"
#include <algorithm>
#include <iostream>
#include <vector>

/**
 * @brief Implementación secuencial de Algoritmos Genéticos con Islas
 *
 * Divide la población en múltiples islas que evolucionan de forma
 * independiente y se comunican mediante migración periódica.
 *
 * Características:
 * - Evolución secuencial de islas
 * - Migración de mejores individuos entre islas
 * - Mayor diversidad genética
 * - Ideal para escapar de óptimos locales
 */
class IslandsSequential : public BaseGA
{
  private:
  int num_islands;
  std::vector<std::vector<Individual>> islands;
  std::vector<Individual> island_offspring;
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

    island_offspring.clear();
    island_offspring.reserve(island_pop.size());

    while (island_offspring.size() < island_pop.size())
    {
      // Seleccionar dos padres de esta isla
      auto [parent1, parent2] = selection_op->select_pair(island_pop, rng);

      // Cruzamiento
      Individual child = crossover_op->apply(parent1, parent2, rng);

      // Mutación
      mutation_op->apply(child, rng);

      // Reparación si es necesario
      if (!validator->is_feasible(child, instance))
      {
        validator->repair(child, instance);
      }

      island_offspring.push_back(child);
    }

    apply_elitism(elite, island_offspring);

    // Reemplazar población de la isla
    islands[island_id] = island_offspring;
  }

  void do_reproduction() override
  {
    // Evolucionar cada isla secuencialmente
    for (int i = 0; i < num_islands; ++i)
    {
      evolve_island(i);
    }
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
   * Estrategia: los mejores de cada isla se intercambian
   */
  void migrate_individuals()
  {
    if (num_islands < 2)
      return;

    std::cout << "[Migration] Gen " << current_generation << std::endl;

    // Encontrar mejores individuos de cada isla
    std::vector<size_t> best_indices(num_islands);
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

    // Intercambiar mejores individuos en forma cíclica
    Individual temp = islands[0][best_indices[0]];
    for (int i = 0; i < num_islands - 1; ++i)
    {
      islands[i][best_indices[i]] = islands[i + 1][best_indices[i + 1]];
    }
    islands[num_islands - 1][best_indices[num_islands - 1]] = temp;
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
    // Evaluar todas las islas
    for (auto& island : islands)
    {
      fitness_evaluator->evaluate_population(island, instance);
    }

    // Mantener sincronizada la población base
    population.clear();
    for (const auto& island : islands)
    {
      population.insert(population.end(), island.begin(), island.end());
    }
  }

  public:
  IslandsSequential(
    KnapsackInstance& inst, int num_isl, int mig_interval = 5,
    std::unique_ptr<ga::operators::Crossover> cross         = nullptr,
    std::unique_ptr<ga::operators::Mutation> mut            = nullptr,
    std::unique_ptr<ga::operators::Selection> sel           = nullptr,
    std::unique_ptr<ga::operators::FitnessEvaluator> fit    = nullptr,
    std::unique_ptr<ga::operators::ConstraintValidator> val = nullptr,
    bool verbose = false, int seed = 0,
    size_t pop_size = 0, size_t num_gens = 0)
    : BaseGA(inst, std::move(cross), std::move(mut), std::move(sel),
             std::move(fit), std::move(val), verbose, seed, pop_size, num_gens),
      num_islands(num_isl), migration_interval(mig_interval)
  {
  }

  virtual ~IslandsSequential() = default;
};
