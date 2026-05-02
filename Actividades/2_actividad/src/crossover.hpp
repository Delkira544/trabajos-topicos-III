#ifndef CROSSOVER_HPP
#define CROSSOVER_HPP

#include "genetic_algorithm.hpp"
#include <random>

namespace Crossover {
    void SinglePoint(const Individual &parent1, const Individual &parent2,
                     Individual &child1, Individual &child2, std::mt19937 &rng);
}

#endif
