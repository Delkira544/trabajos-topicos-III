#pragma once
#include "config/constants.hpp"
#include "types.hpp"
#include <algorithm>
#include <cstddef>
#include <random>
#include <numeric>

class GeneticSolver
{
  protected:
  KnapsackInstance instance;
  size_t population_size;
  size_t generations;
  float mutation_rate;
  float crossover_rate;
  bool verbose;
  std::mt19937 rng;
  int initial_seed;

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

  // Mejor solución factible encontrada (Req 5.1)
  Individual best_valid_individual;

  // Histórico de fitness por generación
  std::vector<float> fitness_history;

  public:
  GeneticSolver(KnapsackInstance& instance, 
               size_t pop_size = 0,
               size_t num_gens = 0, 
               int seed = 0)
  {
    this->instance        = instance;
    this->population_size = (pop_size > 0) ? pop_size : Config::GeneticAlgorithm::POPULATION_SIZE;
    this->generations     = (num_gens > 0) ? num_gens : Config::GeneticAlgorithm::GENERATIONS;
    this->mutation_rate   = Config::GeneticAlgorithm::MUTATION_RATE;
    this->crossover_rate  = Config::GeneticAlgorithm::CROSSOVER_RATE;
    this->initial_seed    = seed;
    rng.seed(seed);
  }
  virtual ~GeneticSolver() = default;

  virtual void run() = 0;

  virtual Individual get_best() = 0;

  virtual const std::vector<float>& get_fitness_history() const = 0;
};
