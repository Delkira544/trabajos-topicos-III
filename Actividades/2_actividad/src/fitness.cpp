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

        // 5. Cálculo de penalización total (fórmula)
        float penalizacion_total =
            (exceso_peso * instance.penalties.alpha) +
            (exceso_volumen * instance.penalties.beta) +
            (errores_categoria * instance.penalties.gamma) +
            (errores_incompatibilidad * instance.penalties.delta) +
            (errores_dependencia * instance.penalties.epsilon);

        // 6. Asignación de fitness
        ind.fitness = total_value - penalizacion_total;

        // 7. Seguridad extra
        if (ind.fitness < 0) ind.fitness = 0.0f;

        // 8. Actualizar is_valid (sin errores = válido)
        ind.is_valid = (penalizacion_total == 0.0f);
    }
} // namespace Fitness
