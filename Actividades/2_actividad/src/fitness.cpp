#include "fitness.hpp"
#include <algorithm>
#include <numeric>
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

        // 3. Cálculo de errores numéricos
        float exceso_peso = 0.0f;
        float exceso_volumen = 0.0f;
        int errores_categoria = 0;
        int errores_incompatibilidad = 0;
        int errores_dependencia = 0;

        // 4. Evaluación de restricciones (llenar contadores)

        // Capacidad: peso y volumen
        if (total_weight > instance.knapsack.max_weight) {
            exceso_peso = total_weight - instance.knapsack.max_weight;
        }
        if (total_volume > instance.knapsack.max_volume) {
            exceso_volumen = total_volume - instance.knapsack.max_volume;
        }

        // Categorías
        for (const auto &[cat_name, count] : category_counts) {
            auto it = instance.category_rules.find(cat_name);
            if (it != instance.category_rules.end()) {
                const CategoryRule &rule = it->second;
                if (count < rule.min || count > rule.max) {
                    errores_categoria++;
                }
            }
        }

        // Incompatibilidades
        for (const auto &incomp : instance.incompatibilities) {
            if (selected_ids.count(incomp.id_a) &&
                selected_ids.count(incomp.id_b)) {
                errores_incompatibilidad++;
            }
        }

        // Dependencias
        for (const auto &[current_id, required_id] : instance.dependencies) {
            if (selected_ids.count(current_id) &&
                !selected_ids.count(required_id)) {
                errores_dependencia++;
            }
        }

        // 5. Normalización de cada componente (a [0,1]) y combinación convexa

        // Valor máximo posible (suma de todos los items)
        float max_possible_value = 0.0f;
        for (const auto &item : instance.items) {
            max_possible_value += item.value;
        }

        // Componentes normalizados (protección contra división por cero)
        float norm_value = (max_possible_value > 0.0f)
                               ? (total_value / max_possible_value)
                               : 0.0f;

        float norm_exceso_peso =
            (instance.knapsack.max_weight > 0.0f)
                ? (exceso_peso / instance.knapsack.max_weight)
                : 0.0f;

        float norm_exceso_volumen =
            (instance.knapsack.max_volume > 0.0f)
                ? (exceso_volumen / instance.knapsack.max_volume)
                : 0.0f;

        float norm_errores_categoria =
            (!instance.category_rules.empty())
                ? (static_cast<float>(errores_categoria) /
                   static_cast<float>(instance.category_rules.size()))
                : 0.0f;

        float norm_errores_incompatibilidad =
            (!instance.incompatibilities.empty())
                ? (static_cast<float>(errores_incompatibilidad) /
                   static_cast<float>(instance.incompatibilities.size()))
                : 0.0f;

        float norm_errores_dependencia =
            (!instance.dependencies.empty())
                ? (static_cast<float>(errores_dependencia) /
                   static_cast<float>(instance.dependencies.size()))
                : 0.0f;

        // Penalización total = combinación convexa (suma ponderada normalizada)
        float penalizacion_total =
            instance.penalties.alpha * norm_exceso_peso +
            instance.penalties.beta * norm_exceso_volumen +
            instance.penalties.gamma * norm_errores_categoria +
            instance.penalties.delta * norm_errores_incompatibilidad +
            instance.penalties.epsilon * norm_errores_dependencia;

        // 6. Asignación de fitness (valor normalizado menos penalización
        // convexa)
        ind.fitness = norm_value - penalizacion_total;

        // 7. Actualizar is_valid (sin errores = válido)
        ind.is_valid = (penalizacion_total == 0.0f);
    }
} // namespace Fitness
