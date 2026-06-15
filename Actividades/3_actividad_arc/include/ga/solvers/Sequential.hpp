#pragma once
#include "ga/BaseGA.hpp"
#include <algorithm>
#include <vector>
#include <iostream>

/**
 * @brief Implementación secuencial del Algoritmo Genético
 *
 * Todo se ejecuta en un solo thread. La población se reproduce
 * de forma secuencial en cada generación.
 *
 * Características:
 * - Determinista (si se usa el mismo seed)
 * - Bajo overhead de sincronización
 * - Ideal para debugging y línea base
 */
class Sequential : public BaseGA
{
  private:
  std::vector<Individual> offspring;

  protected:
  void do_reproduction() override
  {
    std::vector<Individual> elite = population;
    std::sort(elite.begin(), elite.end(),
              [](const Individual& a, const Individual& b) {
                return a.fitness > b.fitness;
              });

    offspring.clear();
    offspring.reserve(population_size);

    // Reproducir hasta llenar la nueva población
    while (offspring.size() < population_size)
    {
      // Seleccionar dos padres
      auto [parent1, parent2] = selection_op->select_pair(population, rng);

      // Cruzamiento
      Individual child = crossover_op->apply(parent1, parent2, rng);

      // Mutación
      mutation_op->apply(child, rng);

      // Reparación si es necesario
      if (!validator->is_feasible(child, instance))
      {
        validator->repair(child, instance);
      }

      offspring.push_back(child);
    }

    apply_elitism(elite, offspring);

    // Reemplazar población
    population = offspring;
  }

  public:
  Sequential(KnapsackInstance& inst,
             std::unique_ptr<ga::operators::Crossover> cross,
             std::unique_ptr<ga::operators::Mutation> mut,
             std::unique_ptr<ga::operators::Selection> sel,
             std::unique_ptr<ga::operators::FitnessEvaluator> fit,
             std::unique_ptr<ga::operators::ConstraintValidator> val,
             bool verbose = false, int seed = 0, 
             size_t pop_size = 0, size_t num_gens = 0)
    : BaseGA(inst, std::move(cross), std::move(mut), std::move(sel),
             std::move(fit), std::move(val), verbose, seed, pop_size, num_gens)
  {
  }

  virtual ~Sequential() = default;
};