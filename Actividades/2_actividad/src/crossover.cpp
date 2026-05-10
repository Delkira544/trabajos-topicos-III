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

    void Uniform(const Individual &parent1, const Individual &parent2,
                 Individual &child1, Individual &child2, std::mt19937 &rng,
                 float inclusion_bias) {

        size_t num_genes = parent1.chromosome.size();
        child1.chromosome.resize(num_genes);
        child2.chromosome.resize(num_genes);

        // Cuando los padres difieren, decidimos cada bit con la probabilidad
        // inclusion_bias (default 0.45 → leve sesgo a no incluir).
        std::bernoulli_distribution include(inclusion_bias);

        for (size_t i = 0; i < num_genes; ++i) {
            bool g1 = parent1.chromosome[i];
            bool g2 = parent2.chromosome[i];
            if (g1 == g2) {
                // Padres concuerdan: heredar tal cual
                child1.chromosome[i] = g1;
                child2.chromosome[i] = g1;
            } else {
                // Padres difieren: muestreo independiente con sesgo
                child1.chromosome[i] = include(rng);
                child2.chromosome[i] = include(rng);
            }
        }

        child1.fitness = 0.0f;
        child2.fitness = 0.0f;
    }

} // namespace Crossover
