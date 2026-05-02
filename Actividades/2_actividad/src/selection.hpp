#ifndef SELECTION_HPP
#define SELECTION_HPP

#include "genetic_algorithm.hpp"
#include <vector>
#include <random>

namespace Selection {
    Individual Tournament(const std::vector<Individual> &population, int k,
                          std::mt19937 &rng);
} // namespace Selection

#endif
