#ifndef CROSSOVER_HPP
#define CROSSOVER_HPP

#include "genetic_algorithm.hpp"
#include <random>

// =============================================================================
// Crossover — operadores de recombinación genética
// =============================================================================
// Provee dos esquemas de cruce binario que producen dos hijos a partir de dos
// padres. Ambos asumen cromosomas del mismo tamaño y dejan los hijos listos
// para ser luego mutados y evaluados.
// =============================================================================

namespace Crossover {

    /**
     * @brief Cruce de un punto (single-point crossover).
     *
     * Sortea un punto de corte cp ∈ [1, n-1]. Construye:
     *   c1 = parent1[0..cp) ⊕ parent2[cp..n)
     *   c2 = parent2[0..cp) ⊕ parent1[cp..n)
     *
     * Desventaja conocida: para problemas de mochila al borde de capacidad,
     * tiende a producir hijos sobre-capacidad con mayor frecuencia que un
     * cruce uniforme con sesgo.
     *
     * @note Disponible pero NO usado por el flujo actual del AG (queda como
     *       fallback o para comparación experimental).
     */
    void SinglePoint(const Individual &parent1, const Individual &parent2,
                     Individual &child1, Individual &child2, std::mt19937 &rng);

    /**
     * @brief Cruce uniforme con sesgo de inclusión (Mejora #4).
     *
     * Para cada posición i:
     *   - Si parent1[i] == parent2[i]:  c1[i] = c2[i] = parent1[i]  (consenso)
     *   - Si difieren:  cada hijo decide independientemente; pone 1 con
     *     probabilidad `inclusion_bias`, sino pone 0.
     *
     * Con `inclusion_bias = 0.45` se introduce un leve sesgo hacia NO incluir
     * cuando los padres están en desacuerdo. Esto reduce la probabilidad de
     * que los hijos hereden demasiados ítems → mantiene la búsqueda dentro de
     * regiones cercanas a la factibilidad de capacidad.
     *
     * @param inclusion_bias  Probabilidad de poner 1 cuando los padres difieren.
     *                        0.5 = uniforme clásico; <0.5 sesga a 0; >0.5 sesga a 1.
     */
    void Uniform(const Individual &parent1, const Individual &parent2,
                 Individual &child1, Individual &child2, std::mt19937 &rng,
                 float inclusion_bias = 0.45f);

} // namespace Crossover

#endif
