#include "selection.hpp"
#include <stdexcept>

namespace Selection {

    Individual Tournament(const std::vector<Individual> &population, int k,
                          std::mt19937 &rng) {
        // Protección de seguridad básica
        if (population.empty() || k <= 0) {
            throw std::invalid_argument(
                "La población está vacía o el tamaño del torneo es inválido.");
        }
        std::uniform_int_distribution<int> dist(0, population.size() - 1);

        int best_idx = dist(rng);
        float best_fitness = population[best_idx].fitness;

        for (int i = 1; i < k; ++i) {
            int rand_idx = dist(rng);

            const Individual &current = population[best_idx];
            const Individual &candidate = population[rand_idx];

            bool candidate_better = false;

            if (candidate.is_valid && current.is_valid) {
                candidate_better = candidate.fitness > best_fitness;
            } else if (!candidate.is_valid == !current.is_valid) {
                candidate_better = candidate.fitness > best_fitness;
            } else if (candidate.is_valid && !current.is_valid) {
                candidate_better = true;
            }

            if (candidate_better) {
                best_idx = rand_idx;
                best_fitness = candidate.fitness;
            }
        }

        return population[best_idx];
    }

} // namespace Selection
