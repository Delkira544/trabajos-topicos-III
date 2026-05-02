#include "mutation.hpp"

namespace Mutation {

    void BitFlip(Individual &ind, float mutation_rate, std::mt19937 &rng) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < ind.chromosome.size(); ++i) {
            if (dist(rng) < mutation_rate) {
                ind.chromosome[i] = !ind.chromosome[i];
            }
        }
    }
} // namespace Mutation
