#ifndef CROSSOVER_HPP
#define CROSSOVER_HPP

#include "genetic_algorithm.hpp"
#include <random>

namespace Crossover {
    void SinglePoint(const Individual &parent1, const Individual &parent2,
                     Individual &child1, Individual &child2, std::mt19937 &rng);

    // Mejora #4 — Uniform crossover con sesgo: cuando los padres difieren en
    // un gen, sampleamos 1 con probabilidad inclusion_bias (típicamente 0.45,
    // sesgo a no incluir) → preserva factibilidad de capacidad mejor que
    // single-point cuando los padres están al borde de la capacidad.
    void Uniform(const Individual &parent1, const Individual &parent2,
                 Individual &child1, Individual &child2, std::mt19937 &rng,
                 float inclusion_bias = 0.45f);
}

#endif
