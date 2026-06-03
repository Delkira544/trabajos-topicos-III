#pragma once
#include "config/constants.hpp"
#include "types.hpp"
#include <algorithm>
#include <cstddef>
#include <random>

class GeneticSolver
{
  protected:
  KnapsackInstance instance;
  std::vector<Individual> population;
  size_t population_size;
  size_t generations;
  float mutation_rate;
  float crossover_rate;
  bool verbose;
  std::mt19937 rng;

  void initialize_population()
  {
    population.clear();
    for (size_t i = 0; i < population_size; ++i)
    {
      population.push_back(CreateRandomIndividual(rng));
    }
  }

  Individual CreateRandomIndividual(std::mt19937& r)
  {
    Individual individual;
    individual.chromosome.assign(instance.items.size(), false);

    std::bernoulli_distribution d(0.25);

    std::vector<int> indices(instance.items.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), r);

    float total_weight = 0.0f;
    float total_volume = 0.0f;
    for (int idx : indices)
    {
      const Item& item = instance.items[idx];
      if (total_weight + item.weight <= instance.max_weight &&
          total_volume + item.volume <= instance.max_volume)
      {
        if (d(r))
        {
          individual.chromosome[idx] = true;
          total_weight += item.weight;
          total_volume += item.volume;
        }
      }
    }
    return individual;
  }

  // Mejor solución encontrada
  Individual best_individual;

  // Histórico de fitness por generación
  std::vector<float> fitness_history;

  public:
  GeneticSolver(KnapsackInstance& instance, int seed = 0)
  {
    this->instance        = instance;
    this->population_size = Config::GeneticAlgorithm::POPULATION_SIZE;
    this->generations     = Config::GeneticAlgorithm::GENERATIONS;
    this->mutation_rate   = Config::GeneticAlgorithm::MUTATION_RATE;
    this->crossover_rate  = Config::GeneticAlgorithm::CROSSOVER_RATE;
    rng.seed(seed);
  }
  virtual ~GeneticSolver() = default;

  virtual void run() = 0;

  /**
   * @brief Obtiene el mejor individuo encontrado
   */
  virtual Individual get_best() = 0;

  /**
   * @brief Obtiene el histórico de fitness
   */
  virtual const std::vector<float>& get_fitness_history() const = 0;
};
