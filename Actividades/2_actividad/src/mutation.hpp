#ifndef MUTATION_HPP
#define MUTATION_HPP

#include "genetic_algorithm.hpp"
#include <random>

namespace Mutation {
    // Bit-flip simétrico clásico: cada gen voltea con probabilidad mutation_rate.
    void BitFlip(Individual &ind, float mutation_rate, std::mt19937 &rng);

    // Mejora #5 — Bit-flip asimétrico consciente de la capacidad:
    // si el individuo excede peso/volumen, sesga fuertemente hacia eliminar
    // (1→0) y reduce la prob. de añadir (0→1). Si está bajo capacidad,
    // se comporta como BitFlip clásico.
    void BitFlipAsymmetric(Individual &ind, float mutation_rate,
                           const Instance &instance, std::mt19937 &rng);

    // Mutación dirigida a violaciones soft: con probabilidad call_prob por
    // individuo, recopila las violaciones activas (incompatibilidades y
    // dependencias) y aplica UNA reparación al azar a UNA violación al azar.
    // No es greedy (no fixea todo), no es determinista — sólo da al GA un
    // sesgo hacia los genes específicamente involucrados en las violaciones.
    void TargetedFix(Individual &ind, const Instance &instance,
                     std::mt19937 &rng, float call_prob = 0.5f);
}

#endif
