#ifndef FITNESS_HPP
#define FITNESS_HPP

#include <iostream>
#include "genetic_algorithm.hpp"

namespace Fitness {
    void Evaluate(Individual &ind, const Instance &instance);
    void PrintConstraintDetails(const Individual &ind,
                                const Instance &instance);
} // namespace Fitness

#endif
