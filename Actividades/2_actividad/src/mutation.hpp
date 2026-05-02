#ifndef MUTATION_HPP
#define MUTATION_HPP

#include "genetic_algorithm.hpp"
#include <random>

namespace Mutation {
    void BitFlip(Individual &ind, float mutation_rate, std::mt19937 &rng);
}

#endif
