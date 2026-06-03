#pragma once
#include "ga/GeneticSolver.hpp"
#include "ga/operators/ConstraintValidator.hpp"
#include "ga/operators/Crossover.hpp"
#include "ga/operators/FitnessEvaluator.hpp"
#include "ga/operators/Mutation.hpp"
#include "ga/operators/Selection.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

/**
 * @brief Clase base para todos los algoritmos genéticos
 *
 * Implementa el patrón Template Method definiendo el flujo genérico
 * del algoritmo genético. Las subclases solo necesitan implementar
 * los métodos virtuales para:
 * - do_reproduction(): cómo se reproduce la población
 * - do_migration(): cómo se comunican entre islas (si aplica)
 * - synchronize_populations(): sincronización entre islas (si aplica)
 */
class BaseGA : public GeneticSolver
{
  protected:
  // COMPONENTES INYECTADOS
  std::unique_ptr<ga::operators::Crossover> crossover_op;
  std::unique_ptr<ga::operators::Mutation> mutation_op;
  std::unique_ptr<ga::operators::Selection> selection_op;
  std::unique_ptr<ga::operators::FitnessEvaluator> fitness_evaluator;
  std::unique_ptr<ga::operators::ConstraintValidator> validator;

  // POBLACIÓN BASE
  std::vector<Individual> population;
  size_t generations_without_improvement = 0;
  float last_best_fitness = -std::numeric_limits<float>::infinity();

  struct GenerationStats
  {
    float avg_fitness  = 0.0f;
    float std_fitness  = 0.0f;
    float avg_penalty  = 0.0f;
    float std_penalty  = 0.0f;
    int feasible_count = 0;
  };

  std::vector<GenerationStats> generation_stats_history;

  // ESTADÍSTICAS
  std::chrono::high_resolution_clock::time_point start_time;
  size_t current_generation = 0;

  /**
   * @brief Inicializa la población con individuos aleatorios
   */
  virtual void initialize_population()
  {
    population.clear();
    for (size_t i = 0; i < population_size; ++i)
    {
      population.push_back(CreateRandomIndividual(rng));
    }
  }

  /**
   * @brief Evalúa toda la población
   */
  virtual void evaluate_population()
  {
    fitness_evaluator->evaluate_population(population, instance);
  }

  /**
   * @brief Actualiza el mejor individuo encontrado
   */
  void update_best_solution()
  {
    best_individual = fitness_evaluator->get_best(population);
  }

  /**
   * @brief Registra estadísticas de la generación actual
   */
  void log_generation_stats(size_t generation)
  {
    update_best_solution();
    fitness_history.push_back(best_individual.fitness);
    // NUEVA LÓGICA: Detectar convergencia
    if (best_individual.fitness > last_best_fitness)
    {
      // Hubo mejora
      last_best_fitness               = best_individual.fitness;
      generations_without_improvement = 0;
    }
    else
    {
      // No hubo mejora
      generations_without_improvement++;
    }

    GenerationStats stats = calculate_generation_stats();
    generation_stats_history.push_back(stats);

    // Logging cada 10 generaciones
    if (verbose && generation % 10 == 0)
    {
      std::cout << "[Gen " << generation
                << "] Best: " << best_individual.fitness
                << " | Avg: " << stats.avg_fitness << " (±" << stats.std_fitness
                << ")"
                << " | Penalty Avg: " << stats.avg_penalty << " (±"
                << stats.std_penalty << ")"
                << " | Feasible: " << stats.feasible_count << "/"
                << population.size()
                << " | No improvement: " << generations_without_improvement
                << "/" << Config::GeneticAlgorithm::CONVERGENCE_GENERATIONS
                << std::endl;
    }
    // NUEVA LÓGICA: Información de convergencia detectada
    if (verbose && generations_without_improvement ==
                     Config::GeneticAlgorithm::CONVERGENCE_GENERATIONS)
    {
      std::cout << "\n[CONVERGENCE DETECTED] No improvement in last "
                << Config::GeneticAlgorithm::CONVERGENCE_GENERATIONS
                << " generations." << std::endl;

      if (best_individual.is_valid)
      {
        std::cout << "[VALID SOLUTION FOUND] → Algorithm will STOP"
                  << std::endl;
      }
      else
      {
        std::cout << "[INVALID SOLUTION] → Algorithm will CONTINUE searching..."
                  << std::endl;
      }
    }
  }

  /**
   * @brief Aplica elitismo: copia el top ELITISM_PERCENTAGE de la generación
   * anterior reemplazando los peores del offspring
   */
  void apply_elitism(std::vector<Individual>& elite,
                     std::vector<Individual>& offspring)
  {
    if (elite.empty() || offspring.empty())
    {
      return;
    }

    size_t base_size     = std::min(elite.size(), offspring.size());
    size_t elitism_count = static_cast<size_t>(
      base_size * Config::GeneticAlgorithm::ELITISM_PERCENTAGE);
    elitism_count = std::min(elitism_count, base_size);

    if (elitism_count == 0)
    {
      return;
    }

    std::vector<Individual> best_elite(elite.begin(),
                                       elite.begin() + elitism_count);

    std::vector<size_t> indices(offspring.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + elitism_count,
                      indices.end(), [&offspring](size_t a, size_t b) {
                        return offspring[a].fitness < offspring[b].fitness;
                      });

    for (size_t i = 0; i < elitism_count; ++i)
    {
      offspring[indices[i]] = best_elite[i];
    }
  }

  GenerationStats calculate_generation_stats()
  {
    GenerationStats stats;
    if (population.empty())
    {
      return stats;
    }

    float sum_fitness  = 0.0f;
    float sum_penalty  = 0.0f;
    int feasible_count = 0;

    for (const auto& ind : population)
    {
      sum_fitness += ind.fitness;
      sum_penalty += ind.penalty;
      if (ind.is_valid)
      {
        feasible_count++;
      }
    }

    stats.avg_fitness    = sum_fitness / population.size();
    stats.avg_penalty    = sum_penalty / population.size();
    stats.feasible_count = feasible_count;

    float sum_sq_fitness = 0.0f;
    float sum_sq_penalty = 0.0f;
    for (const auto& ind : population)
    {
      float diff_fitness = ind.fitness - stats.avg_fitness;
      float diff_penalty = ind.penalty - stats.avg_penalty;
      sum_sq_fitness += diff_fitness * diff_fitness;
      sum_sq_penalty += diff_penalty * diff_penalty;
    }

    stats.std_fitness = std::sqrt(sum_sq_fitness / population.size());
    stats.std_penalty = std::sqrt(sum_sq_penalty / population.size());

    return stats;
  }

  /**
   * @brief MÉTODO VIRTUAL: Implementar reproducción
   * Cada solver debe definir cómo se reproduce la población
   */
  virtual void do_reproduction() = 0;

  /**
   * @brief MÉTODO VIRTUAL (opcional): Implementar migración entre islas
   * Solo para solvers tipo Islands
   */
  virtual void do_migration() {}

  /**
   * @brief MÉTODO VIRTUAL (opcional): Sincronización entre poblaciones
   * Solo para solvers tipo Islands
   */
  virtual void synchronize_populations() {}

  public:
  BaseGA(KnapsackInstance& inst,
         std::unique_ptr<ga::operators::Crossover> cross,
         std::unique_ptr<ga::operators::Mutation> mut,
         std::unique_ptr<ga::operators::Selection> sel,
         std::unique_ptr<ga::operators::FitnessEvaluator> fit,
         std::unique_ptr<ga::operators::ConstraintValidator> val,
         bool verbose = false, int seed = 0)
    : GeneticSolver(inst, seed), crossover_op(std::move(cross)),
      mutation_op(std::move(mut)), selection_op(std::move(sel)),
      fitness_evaluator(std::move(fit)), validator(std::move(val))
  {
    this->verbose = verbose;
  }

  virtual ~BaseGA() = default;

  /**
   * @brief MÉTODO FINAL: Ejecuta el algoritmo genético completo
   *
   * Este es el Template Method que define el flujo genérico.
   * Las subclases solo personalizan do_reproduction(), do_migration(), etc.
   */
  void run() final
  {
    start_time = std::chrono::high_resolution_clock::now();

    std::cout << "\n=== Starting Genetic Algorithm ===" << std::endl;
    std::cout << "Population size: " << population_size << std::endl;
    std::cout << "Generations: " << generations << std::endl;

    initialize_population();
    evaluate_population();
    update_best_solution();

    fitness_history.push_back(best_individual.fitness);

    size_t gen = 0;
    while (true)
    {
      current_generation = gen;

      // Fase 1: Reproducción (varía según solver)
      do_reproduction();

      // Fase 2: Evaluación
      evaluate_population();

      // Fase 3: Migración entre islas (si aplica)
      do_migration();

      // Fase 4: Sincronización (si aplica)
      synchronize_populations();

      // Fase 5: Logging
      log_generation_stats(gen);

      if (generations_without_improvement >=
            Config::GeneticAlgorithm::CONVERGENCE_GENERATIONS &&
          best_individual.is_valid)
      {
        std::cout << "\n=== STOPPING: Converged AND Valid Solution Found ==="
                  << std::endl;
        break;
      }

      gen++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Best: " << best_individual.fitness << std::endl;
    std::cout << "Valid: " << (best_individual.is_valid ? "Yes" : "No")
              << std::endl;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    if (!generation_stats_history.empty())
    {
      float global_avg_fitness = 0.0f;
      float global_avg_penalty = 0.0f;

      for (const auto& stats : generation_stats_history)
      {
        global_avg_fitness += stats.avg_fitness;
        global_avg_penalty += stats.avg_penalty;
      }

      global_avg_fitness /= generation_stats_history.size();
      global_avg_penalty /= generation_stats_history.size();

      std::cout << "\n=== Final Statistics ===" << std::endl;
      std::cout << "Average fitness across all generations: "
                << global_avg_fitness << std::endl;
      std::cout << "Average penalty across all generations: "
                << global_avg_penalty << std::endl;
      std::cout << "Total generations: " << generation_stats_history.size()
                << std::endl;
    }
    std::cout << "==============================\n" << std::endl;
  }

  Individual get_best() override { return best_individual; }

  const std::vector<float>& get_fitness_history() const override
  {
    return fitness_history;
  }
};
