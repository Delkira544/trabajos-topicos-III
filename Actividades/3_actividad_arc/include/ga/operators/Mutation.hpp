#pragma once
#include "types.hpp"
#include <random>

namespace ga::operators
{
  /**
   * @brief Interfaz para operadores de mutación
   */
  class Mutation
  {
  public:
    virtual ~Mutation() = default;

    /**
     * @brief Aplica mutación a un individuo
     * @param individual Individuo a mutar (se modifica in-place)
     * @param rng Generador de números aleatorios
     */
    virtual void apply(Individual& individual, std::mt19937& rng) = 0;

    /**
     * @brief Obtiene la probabilidad de mutación
     */
    virtual float get_mutation_rate() const = 0;
  };
} // namespace ga::operators
