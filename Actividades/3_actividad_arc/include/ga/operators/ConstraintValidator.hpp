#pragma once
#include "types.hpp"

namespace ga::operators
{
  /**
   * @brief Interfaz para validadores de restricciones
   */
  class ConstraintValidator
  {
  public:
    virtual ~ConstraintValidator() = default;

    /**
     * @brief Valida si un individuo es factible
     * @param individual Individuo a validar
     * @param instance Instancia del problema
     * @return true si cumple todas las restricciones
     */
    virtual bool is_feasible(const Individual& individual,
                             const KnapsackInstance& instance) = 0;

    /**
     * @brief Repara un individuo infactible si es posible
     * @param individual Individuo a reparar (se modifica in-place)
     * @param instance Instancia del problema
     */
    virtual void repair(Individual& individual,
                        const KnapsackInstance& instance) = 0;
  };
} // namespace ga::operators
