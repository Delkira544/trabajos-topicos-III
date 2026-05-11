#include "selection.hpp"
#include <stdexcept>

// =============================================================================
// Implementación de Selection::Tournament.
//
// Algoritmo:
//   1. Tomar un índice al azar como "campeón provisional".
//   2. Sortear k-1 índices más; reemplazar el campeón si alguno es mejor
//      según IsBetter (orden lexicográfico definido en genetic_algorithm.hpp).
//   3. Retornar una copia del campeón final.
//
// Complejidad: O(k) por llamada.
// =============================================================================

namespace Selection {

    Individual Tournament(const std::vector<Individual> &population, int k,
                          std::mt19937 &rng) {
        // Validación defensiva: población vacía o k <= 0 dispararía UB en la
        // distribución uniforme de abajo.
        if (population.empty() || k <= 0) {
            throw std::invalid_argument(
                "La población está vacía o el tamaño del torneo es inválido.");
        }

        // Distribución de índices uniforme [0, N-1].
        std::uniform_int_distribution<int> dist(0, population.size() - 1);

        // Campeón inicial: un índice al azar.
        int best_idx = dist(rng);

        // Comparar contra k-1 retadores y quedarse con el mejor.
        for (int i = 1; i < k; ++i) {
            int rand_idx = dist(rng);
            if (IsBetter(population[rand_idx], population[best_idx]))
                best_idx = rand_idx;
        }

        // Retornamos COPIA — el llamador suele modificarla (crossover/mutación).
        return population[best_idx];
    }

} // namespace Selection
