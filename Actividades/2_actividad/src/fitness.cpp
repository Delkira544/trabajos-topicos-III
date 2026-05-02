#include "fitness.hpp"
#include <unordered_map>
#include <string>

namespace Fitness {

    void Evaluate(Individual &ind, const Instance &instance) {
        // 1. Variables acumuladoras
        float total_value = 0.0f;
        float total_weight = 0.0f;
        float total_volume = 0.0f;

        // Para contar cuántos ítems llevamos de cada categoría
        std::unordered_map<std::string, int> category_counts;

        // 2. Recorrer el cromosoma (Cálculo crudo)
        for (size_t i = 0; i < ind.chromosome.size(); ++i) {
            if (ind.chromosome[i]) { // Si el gen es true (el ítem está en la
                                     // mochila)
                const Item &item = instance.items[i];

                total_value += item.value;
                total_weight += item.weight;
                total_volume += item.volume;
                category_counts[item.category]++;
            }
        }

        // 3. Validación de Restricciones
        bool is_valid = true;

        // A. Validar Capacidades Físicas
        if (total_weight > instance.knapsack.max_weight ||
            total_volume > instance.knapsack.max_volume) {
            is_valid = false;
        }

        // B. Validar Reglas de Categoría
        if (is_valid) {
            for (const auto &[cat_name, count] : category_counts) {
                // Buscamos la regla para esta categoría
                auto it = instance.category_rules.find(cat_name);
                if (it != instance.category_rules.end()) {
                    const CategoryRule &rule = it->second;
                    if (count < rule.min || count > rule.max) {
                        is_valid = false;
                        break; // Si falla una, ya no es válido, dejamos de
                               // buscar
                    }
                }
            }
        }

        // C. Validar Incompatibilidades
        if (is_valid) {
            for (const auto &incomp : instance.incompatibilities) {
                // Necesitamos saber si AMBOS IDs están en la mochila.
                // *Nota de optimización abajo
                bool has_a = false;
                bool has_b = false;

                for (size_t i = 0; i < ind.chromosome.size(); ++i) {
                    if (ind.chromosome[i]) {
                        if (instance.items[i].id == incomp.id_a) has_a = true;
                        if (instance.items[i].id == incomp.id_b) has_b = true;
                    }
                }

                if (has_a && has_b) {
                    is_valid = false;
                    break;
                }
            }
        }

        if (is_valid) {
            for (size_t i = 0; i < ind.chromosome.size(); ++i) {
                if (ind.chromosome[i]) {
                    int current_id = instance.items[i].id;
                    auto it = instance.dependencies.find(current_id);

                    if (it != instance.dependencies.end()) {
                        int required_id = it->second;
                        bool has_required = false;

                        // Buscamos si el requerido está en la mochila
                        for (size_t j = 0; j < ind.chromosome.size(); ++j) {
                            if (ind.chromosome[j] &&
                                instance.items[j].id == required_id) {
                                has_required = true;
                                break;
                            }
                        }

                        if (!has_required) {
                            is_valid = false;
                            break;
                        }
                    }
                }
            }
        }

        // 4. Asignación del Fitness y Penalización
        ind.is_valid = is_valid;

        if (is_valid) {
            ind.fitness =
                total_value; // Solución perfecta, su puntaje es su valor
        } else {
            // Estrategia de penalización dura: Si rompe las reglas, no vale
            // nada. Para problemas más avanzados, podrías restar un porcentaje.
            ind.fitness = 0.0f;
        }
    }
} // namespace Fitness
