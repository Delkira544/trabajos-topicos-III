#pragma once
#include "types.hpp"
#include <vector>

namespace ga::operators
{
  /**
   * @brief Interfaz para evaluadores de fitness
   */
  class FitnessEvaluator
  {
  public:
    virtual ~FitnessEvaluator() = default;

    /**
     * @brief Evalúa el fitness de un individuo
     * @param individual Individuo a evaluar (se modifica in-place)
     * @param instance Instancia del problema
     */
    virtual void evaluate(Individual& individual,
                          const KnapsackInstance& instance) = 0;

    /**
     * @brief Evalúa toda una población
     * @param population Población a evaluar
     * @param instance Instancia del problema
     */
    virtual void evaluate_population(std::vector<Individual>& population,
                                     const KnapsackInstance& instance) = 0;

    /**
     * @brief Encuentra el mejor individuo de una población
     * @param population Población a evaluar
     * @return Mejor individuo encontrado
     */
    virtual Individual get_best(const std::vector<Individual>& population) = 0;
  };
} // namespace ga::operators
