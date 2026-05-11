#include "crossover.hpp"

// =============================================================================
// Implementación de los operadores de cruce.
// Ambos operadores dejan `child1.fitness = child2.fitness = 0` porque la
// evaluación real se delega a Fitness::Evaluate después de la mutación.
// =============================================================================

namespace Crossover {

    // -------------------------------------------------------------------------
    // SinglePoint — cruce clásico de un punto.
    // -------------------------------------------------------------------------
    void SinglePoint(const Individual &parent1, const Individual &parent2,
                     Individual &child1, Individual &child2,
                     std::mt19937 &rng) {

        int num_genes = parent1.chromosome.size();

        child1.chromosome.resize(num_genes);
        child2.chromosome.resize(num_genes);

        // Punto de corte estricto en [1, n-1] para garantizar que ambos hijos
        // hereden material genético de los dos padres (cp=0 o cp=n sería degenerado).
        std::uniform_int_distribution<int> dist(1, num_genes - 1);
        int cross_point = dist(rng);

        // Recombinación: antes del punto → herencia "recta", después → cruzada.
        for (int i = 0; i < num_genes; ++i) {
            if (i < cross_point) {
                child1.chromosome[i] = parent1.chromosome[i];
                child2.chromosome[i] = parent2.chromosome[i];
            } else {
                child1.chromosome[i] = parent2.chromosome[i];
                child2.chromosome[i] = parent1.chromosome[i];
            }
        }

        // Fitness se calcula recién en Fitness::Evaluate post-mutación.
        child1.fitness = 0.0f;
        child2.fitness = 0.0f;
    }

    // -------------------------------------------------------------------------
    // Uniform — cruce uniforme con sesgo de inclusión (Mejora #4).
    // -------------------------------------------------------------------------
    // Estrategia:
    //   - Genes consenso (padres iguales): se heredan directamente, sin azar.
    //   - Genes en disputa (padres difieren): se muestrea cada hijo de forma
    //     independiente con sesgo `inclusion_bias` (típicamente 0.45).
    //
    // Por qué sesgar a 0.45:
    //   Si ambos padres están al borde de la capacidad, un sampleo uniforme
    //   (0.5) tiende a producir hijos sobre-capacidad. Bajar a 0.45 desplaza
    //   levemente la masa hacia "no incluir", manteniendo a los hijos en la
    //   región factible o cerca de ella.
    // -------------------------------------------------------------------------
    void Uniform(const Individual &parent1, const Individual &parent2,
                 Individual &child1, Individual &child2, std::mt19937 &rng,
                 float inclusion_bias) {

        size_t num_genes = parent1.chromosome.size();
        child1.chromosome.resize(num_genes);
        child2.chromosome.resize(num_genes);

        // Distribución reusable para todos los genes en disputa.
        std::bernoulli_distribution include(inclusion_bias);

        for (size_t i = 0; i < num_genes; ++i) {
            bool g1 = parent1.chromosome[i];
            bool g2 = parent2.chromosome[i];
            if (g1 == g2) {
                // Caso 1: los padres concuerdan → herencia directa.
                child1.chromosome[i] = g1;
                child2.chromosome[i] = g1;
            } else {
                // Caso 2: padres difieren → cada hijo decide independientemente
                // (no se "reparte" — ambos hijos pueden quedar igual por azar).
                child1.chromosome[i] = include(rng);
                child2.chromosome[i] = include(rng);
            }
        }

        // Fitness pendiente de re-evaluación post-mutación.
        child1.fitness = 0.0f;
        child2.fitness = 0.0f;
    }

} // namespace Crossover
