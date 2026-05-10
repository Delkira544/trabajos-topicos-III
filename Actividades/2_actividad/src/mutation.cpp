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

    void TargetedFix(Individual &ind, const Instance &instance,
                     std::mt19937 &rng, float call_prob) {
        // Decidir si "tocar" este individuo
        std::bernoulli_distribution call(call_prob);
        if (!call(rng)) return;

        // 1) Recopilar todas las violaciones soft ACTIVAS
        struct Violation {
            int kind;  // 0 = incompatibilidad, 1 = dependencia
            int a, b;  // genes involucrados
        };
        std::vector<Violation> violations;
        violations.reserve(16);

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

        if (violations.empty()) return;

        // 2) Elegir UNA violación al azar y fixearla (de forma random tampoco
        //    determinista: 50/50 entre las dos formas posibles de resolverla).
        std::uniform_int_distribution<size_t> pick(0, violations.size() - 1);
        const Violation &v = violations[pick(rng)];
        std::bernoulli_distribution coin(0.5f);

        if (v.kind == 0) {
            // Incompatibilidad: deseleccionar uno de los dos al azar
            int drop = coin(rng) ? v.a : v.b;
            ind.chromosome[drop] = false;
        } else {
            // Dependencia: agregar el requerido O quitar el dependiente,
            // a coin-flip. Si se agrega y rompe capacidad, BitFlipAsymmetric
            // de la siguiente gen lo balanceará.
            if (coin(rng)) {
                ind.chromosome[v.b] = true;   // v.b = required_id
            } else {
                ind.chromosome[v.a] = false;  // v.a = item_id
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
