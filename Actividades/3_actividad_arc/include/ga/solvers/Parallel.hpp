#pragma once
#include "ga/BaseGA.hpp"
#include <algorithm>
#include <omp.h>
#include <thread>
#include <vector>

/**
 * @brief Implementación paralela del Algoritmo Genético
 *
 * Utiliza OpenMP para paralelizar la reproducción de la población
 * en múltiples threads.
 *
 * Características:
 * - Paralelización a nivel de reproducción (crossover + mutación)
 * - Evaluación también paralelizada
 * - Bajo overhead de comunicación
 * - Ideal para máquinas multicore
 */
class Parallel : public BaseGA
{
  private:
  int num_threads;
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
    offspring.resize(population_size);

    // CORRECCIÓN 4: Evitar Data Race y creación excesiva de RNG
    // Inicializar el RNG base ANTES de la región paralela
    int base_seed = rng() ^ (std::random_device{}());

    // Usar structured binding para OpenMP: cada thread inicializa
    // su propio mt19937 UNA SOLA VEZ (no en cada iteración del for)
#pragma omp parallel num_threads(num_threads)
    {
      // Cada thread crea su propio RNG basado en su ID
      // omp_get_thread_num() retorna el número de thread de 0 a num_threads-1
      std::mt19937 local_rng(base_seed + omp_get_thread_num());

      // Ahora el for loop paralelo puede usar local_rng sin reinicializarlo
#pragma omp for schedule(dynamic)
      for (int i = 0; i < static_cast<int>(population_size); ++i)
      {
        // Seleccionar dos padres
        auto [parent1, parent2] =
          selection_op->select_pair(population, local_rng);

        // Cruzamiento
        Individual child = crossover_op->apply(parent1, parent2, local_rng);

        // Mutación
        mutation_op->apply(child, local_rng);

        // Reparación si es necesario
        if (!validator->is_feasible(child, instance))
        {
          validator->repair(child, instance);
        }

        offspring[i] = child;
      }
    }  // Fin de región paralela

    apply_elitism(elite, offspring);

    // Reemplazar población
    population = offspring;
  }

  public:
  Parallel(KnapsackInstance& inst, int num_th,
           std::unique_ptr<ga::operators::Crossover> cross,
           std::unique_ptr<ga::operators::Mutation> mut,
           std::unique_ptr<ga::operators::Selection> sel,
           std::unique_ptr<ga::operators::FitnessEvaluator> fit,
           std::unique_ptr<ga::operators::ConstraintValidator> val,
           bool verbose = false, int seed = 0,
           size_t pop_size = 0, size_t num_gens = 0)
    : BaseGA(inst, std::move(cross), std::move(mut), std::move(sel),
             std::move(fit), std::move(val), verbose, seed, pop_size, num_gens),
      num_threads(num_th)
  {
  }

  virtual ~Parallel() = default;
};
