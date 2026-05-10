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

        for (int i = 1; i < k; ++i) {
            int rand_idx = dist(rng);
            if (IsBetter(population[rand_idx], population[best_idx]))
                best_idx = rand_idx;
        }

        return population[best_idx];
    }

} // namespace Selection
