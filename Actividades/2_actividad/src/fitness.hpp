#ifndef FITNESS_HPP
#define FITNESS_HPP

#include <iostream>
#include <random>
#include "genetic_algorithm.hpp"

// =============================================================================
// Fitness — función de aptitud y reportes de restricciones
// =============================================================================
// Centraliza el cálculo de fitness y de las banderas auxiliares de cada
// individuo (`is_valid`, `hard_feasible`, `penalty`). También provee un
// pretty-printer para diagnosticar el cumplimiento por tipo de restricción.
// =============================================================================

namespace Fitness {

    /**
     * @brief Evalúa un individuo y rellena fitness/penalty/is_valid/hard_feasible.
     *
     * Modelo de fitness (Mejora #1):
     *   fitness = obj_weight · valor_norm − pen_weight · violacion_norm
     *
     * Donde:
     *   - valor_norm = Σ valores seleccionados / max_value ∈ [0,1].
     *   - violacion_norm = α·norm_p + β·norm_v + γ·norm_cat + δ·norm_inc + ε·norm_dep
     *   - norm_p, norm_v son CUADRÁTICAS amplificadas: min(1, (exceso/cap)²·100).
     *   - norm_cat/inc/dep son lineales (#violaciones / total_posibles).
     *
     * Banderas auxiliares:
     *   - hard_feasible = (exceso_peso ≤ 0) && (exceso_volumen ≤ 0)
     *   - is_valid      = hard_feasible && sin violaciones soft.
     *
     * @param generation        Número de generación actual (reservado, sin uso).
     * @param total_generations Total de generaciones (reservado, sin uso).
     */
    void Evaluate(Individual &ind, const Instance &instance, int generation,
                  int total_generations);

    /**
     * @brief Imprime el estado detallado de cada restricción del individuo.
     *
     * Pensado para reporte final: muestra ✓/✗ peso, ✓/✗ volumen, ✓/✗ cada
     * categoría con su conteo y límites, lista las incompat violadas y las
     * dependencias rotas. Termina con un veredicto "válida/inválida".
     */
    void PrintConstraintDetails(const Individual &ind,
                                const Instance &instance);

} // namespace Fitness

#endif
