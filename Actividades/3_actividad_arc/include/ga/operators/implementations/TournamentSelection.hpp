#pragma once
#include "ga/operators/Selection.hpp"
#include <random>
#include <stdexcept>

namespace ga::operators
{
  /**
   * @brief Implementación de selección por torneo
   */
  class TournamentSelection : public Selection
  {
    private:
    size_t tournament_size;

    public:
    explicit TournamentSelection(size_t size = 3) : tournament_size(size) {}

    const Individual& select(const std::vector<Individual>& population,
                             std::mt19937& rng) override
    {
      if (population.empty())
      {
        throw std::runtime_error("Population cannot be empty for selection");
      }

      std::uniform_int_distribution<size_t> dist(0, population.size() - 1);

      // Seleccionar tournament_size individuos aleatorios
      size_t best_idx    = dist(rng);
      float best_fitness = population[best_idx].fitness;

      for (size_t i = 1; i < tournament_size && i < population.size(); ++i)
      {
        size_t idx = dist(rng);
        if (population[idx].fitness > best_fitness)
        {
          best_fitness = population[idx].fitness;
          best_idx     = idx;
        }
      }

      return population[best_idx];
    }

    std::pair<const Individual&, const Individual&> select_pair(
      const std::vector<Individual>& population, std::mt19937& rng) override
    {
      return {select(population, rng), select(population, rng)};
    }
  };
} // namespace ga::operators
