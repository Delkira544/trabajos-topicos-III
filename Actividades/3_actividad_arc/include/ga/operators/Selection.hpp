#pragma once
#include "types.hpp"
#include <random>
#include <vector>

namespace ga::operators
{
  /**
   * @brief Interfaz para operadores de selección de padres
   */
  class Selection
  {
  public:
    virtual ~Selection() = default;

    /**
     * @brief Selecciona un individuo de la población
     * @param population Población de donde seleccionar
     * @param rng Generador de números aleatorios
     * @return Referencia al individuo seleccionado
     */
    virtual const Individual& select(const std::vector<Individual>& population,
                                      std::mt19937& rng) = 0;

    /**
     * @brief Selecciona dos individuos de la población
     * @param population Población de donde seleccionar
     * @param rng Generador de números aleatorios
     * @return Par de referencias a individuos seleccionados
     */
    virtual std::pair<const Individual&, const Individual&>
    select_pair(const std::vector<Individual>& population, std::mt19937& rng) = 0;
  };
} // namespace ga::operators
