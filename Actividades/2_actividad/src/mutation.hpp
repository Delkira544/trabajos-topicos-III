#ifndef MUTATION_HPP
#define MUTATION_HPP

#include "genetic_algorithm.hpp"
#include <random>

// =============================================================================
// Mutation — operadores de mutación en tres capas
// =============================================================================
// El AG actual combina dos mutaciones por cada hijo:
//   1) BitFlipAsymmetric — toca cada bit con sesgo según el estado de capacidad.
//   2) TargetedFix       — con probabilidad p, fixea UNA violación soft al azar.
//
// El BitFlip clásico queda disponible como fallback/baseline experimental.
// =============================================================================

namespace Mutation {

    /**
     * @brief Bit-flip simétrico clásico (baseline).
     *
     * Recorre el cromosoma y voltea cada bit con probabilidad `mutation_rate`.
     * Trata 1→0 y 0→1 con la misma probabilidad → puede empeorar individuos
     * que ya excedan capacidad. No se usa en el flujo principal.
     */
    void BitFlip(Individual &ind, float mutation_rate, std::mt19937 &rng);

    /**
     * @brief Bit-flip asimétrico consciente de capacidad (Mejora #5).
     *
     * Antes de mutar, calcula el peso/volumen actual del individuo:
     *   - Si EXCEDE capacidad: P(1→0) = mutation_rate · 3 (cap 1.0),
     *     P(0→1) = mutation_rate · 0.25 → sesga fuertemente a eliminar.
     *   - Si está dentro de capacidad: simétrico (igual a BitFlip clásico).
     *
     * Reemplaza al operador de reparación greedy, presionando hacia factibilidad
     * de capacidad de forma probabilística (no determinista).
     */
    void BitFlipAsymmetric(Individual &ind, float mutation_rate,
                           const Instance &instance, std::mt19937 &rng);

    /**
     * @brief Mutación dirigida a violaciones blandas (Mejora #11).
     *
     * Con probabilidad `call_prob`, el operador se "activa". Si se activa:
     *   1. Recolecta todas las violaciones soft ACTUALMENTE activas
     *      (incompatibilidades con ambos miembros seleccionados,
     *       dependencias rotas).
     *   2. Elige UNA violación al azar.
     *   3. La fixea de forma aleatoria:
     *      - Incompat: deselecciona uno de los dos ítems (50/50).
     *      - Dep: o agrega el requerido, o quita el dependiente (50/50).
     *
     * @note NO es Repair (no es greedy ni resuelve todas las violaciones).
     *       Es un sesgo probabilístico que ayuda al GA a tocar genes
     *       específicamente involucrados en violaciones, en vez de confiar
     *       sólo en mutación aleatoria que difícilmente acierta el bit correcto
     *       cuando quedan pocas violaciones residuales (problema del plateau).
     */
    void TargetedFix(Individual &ind, const Instance &instance,
                     std::mt19937 &rng, float call_prob = 0.5f);

} // namespace Mutation

#endif
