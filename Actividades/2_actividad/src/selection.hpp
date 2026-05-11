#ifndef SELECTION_HPP
#define SELECTION_HPP

#include "genetic_algorithm.hpp"
#include <vector>
#include <random>

// =============================================================================
// Selection — operadores de selección de padres
// =============================================================================
// Provee la estrategia de selección por torneo, que escoge `k` individuos
// aleatorios y devuelve el mejor según el orden lexicográfico `IsBetter`
// (hard_feasible → is_valid → penalty → fitness).
// =============================================================================

namespace Selection {

    /**
     * @brief Selección por torneo de tamaño k.
     * @param population  Población de la que seleccionar.
     * @param k           Tamaño del torneo (k=5 es la elección actual del AG).
     * @param rng         Generador aleatorio (puede ser local al hilo).
     * @return  Copia del mejor individuo entre k sorteados al azar.
     * @throws  std::invalid_argument si la población está vacía o k <= 0.
     *
     * Mayor k → mayor presión selectiva (convergencia más rápida, menor diversidad).
     */
    Individual Tournament(const std::vector<Individual> &population, int k,
                          std::mt19937 &rng);

} // namespace Selection

#endif
