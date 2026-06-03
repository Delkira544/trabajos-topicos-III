#pragma once
#include "ga/operators/ConstraintValidator.hpp"
#include "types.hpp"

namespace ga::operators
{
  /**
   * @brief Implementación de validador de restricciones para el Knapsack
   */
  class KnapsackValidator : public ConstraintValidator
  {
  public:
    bool is_feasible(const Individual& individual,
                     const KnapsackInstance& instance) override
    {
      float total_weight = 0.0f;
      float total_volume = 0.0f;

      for (size_t i = 0; i < individual.chromosome.size(); ++i)
      {
        if (individual.chromosome[i])
        {
          total_weight += instance.items[i].weight;
          total_volume += instance.items[i].volume;
        }
      }

      return (total_weight <= instance.max_weight) &&
             (total_volume <= instance.max_volume);
    }

    void repair(Individual& individual,
                const KnapsackInstance& instance) override
    {
      // Estrategia simple: eliminar genes aleatorios hasta que sea factible
      while (!is_feasible(individual, instance))
      {
        // Encontrar un gen en true y ponerlo en false
        bool found = false;
        for (size_t i = individual.chromosome.size(); i > 0; --i)
        {
          if (individual.chromosome[i - 1])
          {
            individual.chromosome[i - 1] = false;
            found = true;
            break;
          }
        }
        if (!found)
          break; // Si no hay más genes, salir
      }
    }
  };
} // namespace ga::operators
