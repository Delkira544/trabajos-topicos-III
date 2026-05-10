#include "crossover.hpp"

namespace Crossover {

    void SinglePoint(const Individual &parent1, const Individual &parent2,
                     Individual &child1, Individual &child2,
                     std::mt19937 &rng) {

        int num_genes = parent1.chromosome.size();

        child1.chromosome.resize(num_genes);
        child2.chromosome.resize(num_genes);

        std::uniform_int_distribution<int> dist(1, num_genes - 1);
        int cross_point = dist(rng);

        for (int i = 0; i < num_genes; ++i) {
            if (i < cross_point) {
                child1.chromosome[i] = parent1.chromosome[i];
                child2.chromosome[i] = parent2.chromosome[i];
            } else {
                child1.chromosome[i] = parent2.chromosome[i];
                child2.chromosome[i] = parent1.chromosome[i];
            }
        }

        child1.fitness = 0.0f;
        child2.fitness = 0.0f;
    }

} // namespace Crossover
