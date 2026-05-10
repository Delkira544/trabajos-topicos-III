#include "mutation.hpp"
#include <algorithm>

namespace Mutation {

    void BitFlip(Individual &ind, float mutation_rate, std::mt19937 &rng) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < ind.chromosome.size(); ++i) {
            if (dist(rng) < mutation_rate) {
                ind.chromosome[i] = !ind.chromosome[i];
            }
        }
    }

    void BitFlipAsymmetric(Individual &ind, float mutation_rate,
                           const Instance &instance, std::mt19937 &rng) {
        // 1) Calcular si el individuo está sobre capacidad
        float total_w = 0.0f, total_v = 0.0f;
        for (size_t i = 0; i < ind.chromosome.size(); ++i) {
            if (ind.chromosome[i]) {
                total_w += instance.items[i].weight;
                total_v += instance.items[i].volume;
            }
        }
        bool over = (total_w > instance.knapsack.max_weight) ||
                    (total_v > instance.knapsack.max_volume);

        // 2) Si excede capacidad: P(1→0) × 3 y P(0→1) × 0.25; si no, simétrico.
        float rate_remove = over ? std::min(1.0f, mutation_rate * 3.0f)
                                 : mutation_rate;
        float rate_add    = over ? mutation_rate * 0.25f
                                 : mutation_rate;

        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < ind.chromosome.size(); ++i) {
            if (ind.chromosome[i]) {
                if (dist(rng) < rate_remove) ind.chromosome[i] = false;
            } else {
                if (dist(rng) < rate_add) ind.chromosome[i] = true;
            }
        }
    }
} // namespace Mutation
