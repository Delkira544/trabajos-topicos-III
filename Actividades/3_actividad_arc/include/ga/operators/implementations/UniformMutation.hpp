#pragma once
#include "ga/operators/Mutation.hpp"
#include <random>

namespace ga::operators
{
  /**
   * @brief Implementación de mutación uniforme (bit flip)
   */
  class UniformMutation : public Mutation
  {
  private:
    float mutation_rate;

  public:
    explicit UniformMutation(float rate = 0.04f) : mutation_rate(rate) {}

    void apply(Individual& individual, std::mt19937& rng) override
    {
      std::bernoulli_distribution dist(mutation_rate);

      for (size_t i = 0; i < individual.chromosome.size(); ++i)
      {
        if (dist(rng))
        {
          individual.chromosome[i] = !individual.chromosome[i]; // Flip del bit
        }
      }
    }

    float get_mutation_rate() const override { return mutation_rate; }
  };
} // namespace ga::operators
