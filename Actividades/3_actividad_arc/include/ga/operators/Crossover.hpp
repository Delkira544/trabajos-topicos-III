#pragma once
#include "types.hpp"
#include <random>

namespace ga::operators
{
  /**
   * @brief Interfaz para operadores de cruzamiento
   */
  class Crossover
  {
  public:
    virtual ~Crossover() = default;

    /**
     * @brief Realiza el cruzamiento entre dos individuos
     * @param parent1 Primer progenitor
     * @param parent2 Segundo progenitor
     * @param rng Generador de números aleatorios
     * @return Nuevo individuo resultado del cruzamiento
     */
    virtual Individual apply(const Individual& parent1,
                             const Individual& parent2,
                             std::mt19937& rng) = 0;

    /**
     * @brief Obtiene la probabilidad de cruzamiento
     */
    virtual float get_crossover_rate() const = 0;
  };
} // namespace ga::operators
