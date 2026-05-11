#include "mutation.hpp"
#include <algorithm>

// =============================================================================
// Implementaciones de las tres mutaciones.
// =============================================================================

namespace Mutation {

    // -------------------------------------------------------------------------
    // BitFlip — mutación simétrica clásica.
    //
    // Para cada bit, voltea con probabilidad `mutation_rate`. Trata 1→0 y 0→1
    // de forma idéntica → puede empeorar a individuos que ya excedan capacidad.
    // -------------------------------------------------------------------------
    void BitFlip(Individual &ind, float mutation_rate, std::mt19937 &rng) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < ind.chromosome.size(); ++i) {
            if (dist(rng) < mutation_rate) {
                ind.chromosome[i] = !ind.chromosome[i];
            }
        }
    }

    // -------------------------------------------------------------------------
    // TargetedFix — mutación dirigida a violaciones blandas (Mejora #11).
    //
    // Inspira el comportamiento sin caer en un operador greedy:
    //   - Se activa con prob `call_prob` (≈50%) → no toca todos los individuos.
    //   - Cuando se activa, sólo fixea UNA violación al azar (no todas).
    //   - El fix elegido es estocástico (50/50 entre las dos resoluciones).
    //
    // Resultado: la presión hacia factibilidad blanda no destruye material
    // genético en bloque, pero sí ayuda a salir del "plateau" donde quedan
    // 1-3 violaciones residuales que mutación aleatoria difícilmente fixea
    // por azar puro (necesitaría acertar exactamente el bit involucrado).
    // -------------------------------------------------------------------------
    void TargetedFix(Individual &ind, const Instance &instance,
                     std::mt19937 &rng, float call_prob) {
        // (a) Decidir si "tocar" este individuo. Si no, salir sin hacer nada.
        std::bernoulli_distribution call(call_prob);
        if (!call(rng)) return;

        // (b) Recolectar todas las violaciones soft ACTUALMENTE activas.
        //     - incompatibilidad activa: ambos ítems del par seleccionados.
        //     - dependencia rota: ítem seleccionado, su requerido NO.
        struct Violation {
            int kind;  // 0 = incompatibilidad, 1 = dependencia
            int a, b;  // genes involucrados
        };
        std::vector<Violation> violations;
        violations.reserve(16); // heurística: pocas violaciones en estado tardío

        for (const auto &inc : instance.incompatibilities) {
            if (ind.chromosome[inc.id_a] && ind.chromosome[inc.id_b]) {
                violations.push_back({0, inc.id_a, inc.id_b});
            }
        }
        for (const auto &[item_id, req_id] : instance.dependencies) {
            if (ind.chromosome[item_id] && !ind.chromosome[req_id]) {
                violations.push_back({1, item_id, req_id});
            }
        }

        // Nada que fixear → salir.
        if (violations.empty()) return;

        // (c) Elegir UNA violación al azar.
        std::uniform_int_distribution<size_t> pick(0, violations.size() - 1);
        const Violation &v = violations[pick(rng)];

        // (d) Resolver la violación elegida con un coin-flip.
        std::bernoulli_distribution coin(0.5f);
        if (v.kind == 0) {
            // Incompatibilidad: eliminar UNO de los dos ítems al azar.
            int drop = coin(rng) ? v.a : v.b;
            ind.chromosome[drop] = false;
        } else {
            // Dependencia: añadir el requerido O quitar el dependiente.
            // Si añadir rompe capacidad, BitFlipAsymmetric en la prox. gen
            // se encargará de balancear vía sesgo a eliminar.
            if (coin(rng)) {
                ind.chromosome[v.b] = true;   // v.b == required_id
            } else {
                ind.chromosome[v.a] = false;  // v.a == item_id
            }
        }
    }

    // -------------------------------------------------------------------------
    // BitFlipAsymmetric — mutación consciente de capacidad (Mejora #5).
    //
    // Reemplaza al operador de Repair eliminado. La idea:
    //   - Computar el estado de capacidad del individuo (peso/vol totales).
    //   - Si EXCEDE → triplicar la prob. de eliminar (1→0), cuartar la de añadir.
    //   - Si NO excede → simétrico (igual a BitFlip clásico).
    //
    // Esto presiona probabilísticamente hacia factibilidad sin recurrir a una
    // proyección determinista (que destruiría diversidad genética).
    // -------------------------------------------------------------------------
    void BitFlipAsymmetric(Individual &ind, float mutation_rate,
                           const Instance &instance, std::mt19937 &rng) {
        // (a) Computar peso y volumen actuales del individuo.
        float total_w = 0.0f, total_v = 0.0f;
        for (size_t i = 0; i < ind.chromosome.size(); ++i) {
            if (ind.chromosome[i]) {
                total_w += instance.items[i].weight;
                total_v += instance.items[i].volume;
            }
        }

        // (b) ¿Excede alguna capacidad? Determina las tasas asimétricas.
        bool over = (total_w > instance.knapsack.max_weight) ||
                    (total_v > instance.knapsack.max_volume);

        // Cap a 1.0 evita que rate_remove sobrepase la probabilidad (1.0).
        float rate_remove = over ? std::min(1.0f, mutation_rate * 3.0f)
                                 : mutation_rate;
        float rate_add    = over ? mutation_rate * 0.25f
                                 : mutation_rate;

        // (c) Aplicar el bit-flip con la tasa correspondiente según el bit actual.
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
