#ifndef FITNESS_HPP
#define FITNESS_HPP

#include <iostream>
#include <random>
#include "genetic_algorithm.hpp"

namespace Fitness {
    void Evaluate(Individual &ind, const Instance &instance, int generation,
                  int total_generations);
    void Repair(Individual &ind, const Instance &instance, std::mt19937 &rng);
    void PrintConstraintDetails(const Individual &ind,
                                const Instance &instance);
} // namespace Fitness

#endif
