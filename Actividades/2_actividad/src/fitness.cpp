#include "fitness.hpp"
#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace Fitness {

    void Evaluate(Individual &ind, const Instance &instance) {
        // 1. Variables acumuladoras
        float total_value = 0.0f;
        float total_weight = 0.0f;
        float total_volume = 0.0f;

        // Para contar cuántos ítems llevamos de cada categoría
        std::unordered_map<std::string, int> category_counts;

        // Set de IDs seleccionados para búsqueda O(1)
        std::unordered_set<int> selected_ids;

        // 2. Recorrer el cromosoma (una sola pasada)
        for (size_t i = 0; i < ind.chromosome.size(); ++i) {
            if (ind.chromosome[i]) {
                const Item &item = instance.items[i];

                total_value += item.value;
                total_weight += item.weight;
                total_volume += item.volume;
                category_counts[item.category]++;
                selected_ids.insert(item.id);
            }
        }

        // 3. Validación de Restricciones
        bool is_valid = true;

        // A. Validar Capacidades Físicas (restricciones duras)
        if (total_weight > instance.knapsack.max_weight ||
            total_volume > instance.knapsack.max_volume) {
            is_valid = false;
        }

        // B. Validar Reglas de Categoría
        if (is_valid) {
            for (const auto &[cat_name, count] : category_counts) {
                auto it = instance.category_rules.find(cat_name);
                if (it != instance.category_rules.end()) {
                    const CategoryRule &rule = it->second;
                    if (count < rule.min || count > rule.max) {
                        is_valid = false;
                        break;
                    }
                }
            }
        }

        // C. Validar Incompatibilidades — O(i) con unordered_set
        if (is_valid) {
            for (const auto &incomp : instance.incompatibilities) {
                if (selected_ids.count(incomp.id_a) &&
                    selected_ids.count(incomp.id_b)) {
                    is_valid = false;
                    break;
                }
            }
        }

        // D. Validar Dependencias — O(d) con unordered_set
        if (is_valid) {
            for (const auto &[current_id, required_id] :
                 instance.dependencies) {
                if (selected_ids.count(current_id) &&
                    !selected_ids.count(required_id)) {
                    is_valid = false;
                    break;
                }
            }
        }

        // 4. Asignación del Fitness con Penalización Gradual
        ind.is_valid = is_valid;

        if (is_valid) {
            ind.fitness = total_value;
        } else {
            float penalty = 0.0f;

            // Penalización por peso excedido
            if (total_weight > instance.knapsack.max_weight) {
                float excess_ratio =
                    (total_weight - instance.knapsack.max_weight) /
                    instance.knapsack.max_weight;
                penalty += total_value * excess_ratio *
                           instance.penalties.weight_penalty;
            }

            // Penalización por volumen excedido
            if (total_volume > instance.knapsack.max_volume) {
                float excess_ratio =
                    (total_volume - instance.knapsack.max_volume) /
                    instance.knapsack.max_volume;
                penalty += total_value * excess_ratio *
                           instance.penalties.volume_penalty;
            }

            // Penalización por violar reglas de categoría
            for (const auto &[cat_name, count] : category_counts) {
                auto it = instance.category_rules.find(cat_name);
                if (it != instance.category_rules.end()) {
                    const CategoryRule &rule = it->second;
                    if (count < rule.min) {
                        penalty +=
                            total_value * instance.penalties.category_penalty;
                    } else if (count > rule.max) {
                        penalty += total_value *
                                   instance.penalties.category_penalty *
                                   (count - rule.max);
                    }
                }
            }

            // Penalización por incompatibilidades violadas
            for (const auto &incomp : instance.incompatibilities) {
                if (selected_ids.count(incomp.id_a) &&
                    selected_ids.count(incomp.id_b)) {
                    penalty += total_value *
                               instance.penalties.incompatibility_penalty;
                }
            }

            // Penalización por dependencias no cumplidas
            for (const auto &[current_id, required_id] :
                 instance.dependencies) {
                if (selected_ids.count(current_id) &&
                    !selected_ids.count(required_id)) {
                    penalty +=
                        total_value * instance.penalties.dependency_penalty;
                }
            }

            ind.fitness = std::max(0.0f, total_value - penalty);
        }
    }
} // namespace Fitness
