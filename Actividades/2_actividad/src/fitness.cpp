#include "fitness.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace Fitness {

    static constexpr float PENALTY_SCALE = 3.0f;

    void Evaluate(Individual &ind, const Instance &instance, int generation,
                  int total_generations) {
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
        for (const auto &[cat_name, rule] : instance.category_rules) {
            int count = 0;
            if (category_counts.count(cat_name))
                count = category_counts.at(cat_name);
            if (count < rule.min || count > rule.max) errores_categoria++;
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

        // 5. Normalización de cada componente a [0,1] usando Pᵢ_max real

        // valor_norm = Σvᵢxᵢ / Σvᵢ  (máximo teórico = suma de todos los valores)
        float norm_value = (instance.max_value > 0.0f)
                               ? (total_value / instance.max_value)
                               : 0.0f;

        // P_peso / P_peso_max  donde P_peso_max = Σpesos_todos − max_weight
        float norm_exceso_peso =
            (instance.max_excess_weight > 0.0f)
                ? std::min(1.0f, exceso_peso / instance.max_excess_weight)
                : 0.0f;

        // P_volumen / P_volumen_max  donde P_volumen_max = Σvolúmenes_todos − max_volume
        float norm_exceso_volumen =
            (instance.max_excess_volume > 0.0f)
                ? std::min(1.0f, exceso_volumen / instance.max_excess_volume)
                : 0.0f;

        // P_cat / n_categorias  (cada categoría puede violarse una vez → Pᵢ_max = n_cat)
        float norm_errores_categoria =
            (!instance.category_rules.empty())
                ? (static_cast<float>(errores_categoria) /
                   static_cast<float>(instance.category_rules.size()))
                : 0.0f;

        // P_incomp / n_incompatibilidades
        float norm_errores_incompatibilidad =
            (!instance.incompatibilities.empty())
                ? (static_cast<float>(errores_incompatibilidad) /
                   static_cast<float>(instance.incompatibilities.size()))
                : 0.0f;

        // P_dep / n_dependencias
        float norm_errores_dependencia =
            (!instance.dependencies.empty())
                ? (static_cast<float>(errores_dependencia) /
                   static_cast<float>(instance.dependencies.size()))
                : 0.0f;

        // violacion_norm = Σ wᵢ·(Pᵢ/Pᵢ_max)  con Σwᵢ = 1 → violacion_norm ∈ [0,1]
        float violacion_norm =
            instance.penalties.alpha   * norm_exceso_peso +
            instance.penalties.beta    * norm_exceso_volumen +
            instance.penalties.gamma   * norm_errores_categoria +
            instance.penalties.delta   * norm_errores_incompatibilidad +
            instance.penalties.epsilon * norm_errores_dependencia;

        // fitness = α·valor_norm − β·violacion_norm  con α+β=1
        ind.penalty  = violacion_norm;
        ind.fitness  = instance.penalties.obj_weight * norm_value
                     - instance.penalties.pen_weight * violacion_norm;
        ind.is_valid = (violacion_norm < 1e-6f);
    }

    void Repair(Individual &ind, const Instance &instance, std::mt19937 &rng) {
        (void)rng; // ya no se usa: la eliminación aleatoria fue reemplazada por greedy
        // 1. Reparar incompatibilidades: eliminar el ítem de menor valor del par violado
        for (const auto &incomp : instance.incompatibilities) {
            int a = incomp.id_a, b = incomp.id_b;
            if (ind.chromosome[a] && ind.chromosome[b]) {
                if (instance.items[a].value <= instance.items[b].value)
                    ind.chromosome[a] = false;
                else
                    ind.chromosome[b] = false;
            }
        }

        // 2. Calcular peso/volumen actuales (tras reparar incompatibilidades)
        float total_weight = 0.0f;
        float total_volume = 0.0f;
        for (size_t i = 0; i < ind.chromosome.size(); ++i) {
            if (ind.chromosome[i]) {
                total_weight += instance.items[i].weight;
                total_volume += instance.items[i].volume;
            }
        }

        // 3. Reparar dependencias: añadir requerido si cabe, sino eliminar dependiente
        for (const auto &[item_id, required_id] : instance.dependencies) {
            if (ind.chromosome[item_id] && !ind.chromosome[required_id]) {
                float w = instance.items[required_id].weight;
                float v = instance.items[required_id].volume;
                if (total_weight + w <= instance.knapsack.max_weight &&
                    total_volume + v <= instance.knapsack.max_volume) {
                    ind.chromosome[required_id] = true;
                    total_weight += w;
                    total_volume += v;
                } else {
                    ind.chromosome[item_id] = false;
                    total_weight -= instance.items[item_id].weight;
                    total_volume -= instance.items[item_id].volume;
                }
            }
        }

        // 4. Reparar capacidad: eliminación greedy por peor ratio valor/(peso+volumen)
        //    Iterar junto con dependencias hasta estabilizar (máx. 5 rondas)
        for (int pass = 0; pass < 5; ++pass) {
            // 4a. Greedy capacity: eliminar ítems con peor ratio primero
            if (total_weight > instance.knapsack.max_weight ||
                total_volume > instance.knapsack.max_volume) {

                std::vector<int> selected;
                for (size_t i = 0; i < ind.chromosome.size(); ++i) {
                    if (ind.chromosome[i]) selected.push_back(i);
                }
                // Ordenar ascendente por ratio valor/(peso+volumen+1): peores primero
                std::sort(selected.begin(), selected.end(), [&](int a, int b) {
                    float ra = instance.items[a].value /
                               (instance.items[a].weight + instance.items[a].volume + 1.0f);
                    float rb = instance.items[b].value /
                               (instance.items[b].weight + instance.items[b].volume + 1.0f);
                    return ra < rb;
                });

                for (int idx : selected) {
                    if (total_weight <= instance.knapsack.max_weight &&
                        total_volume <= instance.knapsack.max_volume) {
                        break;
                    }
                    ind.chromosome[idx] = false;
                    total_weight -= instance.items[idx].weight;
                    total_volume -= instance.items[idx].volume;
                }
            }

            // 4b. Re-reparar dependencias rotas por la eliminación anterior
            bool changed = false;
            for (const auto &[item_id, required_id] : instance.dependencies) {
                if (ind.chromosome[item_id] && !ind.chromosome[required_id]) {
                    float w = instance.items[required_id].weight;
                    float v = instance.items[required_id].volume;
                    if (total_weight + w <= instance.knapsack.max_weight &&
                        total_volume + v <= instance.knapsack.max_volume) {
                        ind.chromosome[required_id] = true;
                        total_weight += w;
                        total_volume += v;
                    } else {
                        ind.chromosome[item_id] = false;
                        total_weight -= instance.items[item_id].weight;
                        total_volume -= instance.items[item_id].volume;
                    }
                    changed = true;
                }
            }

            // Si no hubo cambios en dependencias y capacidad ok, terminar
            if (!changed &&
                total_weight <= instance.knapsack.max_weight &&
                total_volume <= instance.knapsack.max_volume) {
                break;
            }
        }
    }

    void PrintConstraintDetails(const Individual &ind,
                                const Instance &instance) {
        float total_weight = 0.0f;
        float total_volume = 0.0f;
        std::unordered_map<std::string, int> category_counts;

        for (size_t i = 0; i < ind.chromosome.size(); ++i) {
            if (ind.chromosome[i]) {
                const Item &item = instance.items[i];
                total_weight += item.weight;
                total_volume += item.volume;
                category_counts[item.category]++;
            }
        }

        std::cout << "\n--- Restricciones ---\n";

        // Peso
        if (total_weight <= instance.knapsack.max_weight) {
            std::cout << "  ✓ Peso: " << total_weight << "/"
                      << instance.knapsack.max_weight << " kg\n";
        } else {
            std::cout << "  ✗ Peso: " << total_weight << "/"
                      << instance.knapsack.max_weight << " kg (exceso: "
                      << total_weight - instance.knapsack.max_weight
                      << " kg)\n";
        }

        // Volumen
        if (total_volume <= instance.knapsack.max_volume) {
            std::cout << "  ✓ Volumen: " << total_volume << "/"
                      << instance.knapsack.max_volume << " L\n";
        } else {
            std::cout << "  ✗ Volumen: " << total_volume << "/"
                      << instance.knapsack.max_volume << " L (exceso: "
                      << total_volume - instance.knapsack.max_volume << " L)\n";
        }

        // Categorías
        for (const auto &[cat_name, count] : category_counts) {
            auto it = instance.category_rules.find(cat_name);
            if (it != instance.category_rules.end()) {
                const CategoryRule &rule = it->second;
                bool ok = (count >= rule.min && count <= rule.max);
                std::cout << "  " << (ok ? "✓" : "✗") << " Categoría '"
                          << cat_name << "': " << count
                          << " ítems (min: " << rule.min
                          << ", max: " << rule.max << ")"
                          << (ok ? "" : " — incumplida") << "\n";
            }
        }

        // Incompatibilidades
        if (instance.incompatibilities.empty()) {
            std::cout << "  ✓ Sin reglas de incompatibilidad\n";
        } else {
            int errores = 0;
            for (const auto &incomp : instance.incompatibilities) {
                bool a = false, b = false;
                for (size_t i = 0; i < ind.chromosome.size(); ++i) {
                    if (ind.chromosome[i]) {
                        if (instance.items[i].id == incomp.id_a) a = true;
                        if (instance.items[i].id == incomp.id_b) b = true;
                    }
                }
                if (a && b) {
                    std::cout
                        << "  ✗ Incompatibilidad: item " << incomp.id_a
                        << " y item " << incomp.id_b
                        << " son incompatibles y ambos están seleccionados\n";
                    errores++;
                }
            }
            if (errores == 0) {
                std::cout << "  ✓ Incompatibilidades: 0 pares violados\n";
            }
        }

        // Dependencias
        if (instance.dependencies.empty()) {
            std::cout << "  ✓ Sin reglas de dependencia\n";
        } else {
            int errores = 0;
            for (const auto &[current_id, required_id] :
                 instance.dependencies) {
                bool has_current = false, has_required = false;
                for (size_t i = 0; i < ind.chromosome.size(); ++i) {
                    if (ind.chromosome[i]) {
                        if (instance.items[i].id == current_id)
                            has_current = true;
                        if (instance.items[i].id == required_id)
                            has_required = true;
                    }
                }
                if (has_current && !has_required) {
                    std::cout << "  ✗ Dependencia: item " << current_id
                              << " requiere item " << required_id
                              << ", pero no está seleccionado\n";
                    errores++;
                }
            }
            if (errores == 0) {
                std::cout << "  ✓ Dependencias: todas satisfechas\n";
            }
        }

        std::cout
            << "  → Solución "
            << (total_weight <= instance.knapsack.max_weight &&
                        total_volume <= instance.knapsack.max_volume &&
                        [&]() {
                            for (const auto &incomp :
                                 instance.incompatibilities) {
                                bool a = false, b = false;
                                for (size_t i = 0; i < ind.chromosome.size();
                                     ++i) {
                                    if (ind.chromosome[i]) {
                                        if (instance.items[i].id == incomp.id_a)
                                            a = true;
                                        if (instance.items[i].id == incomp.id_b)
                                            b = true;
                                    }
                                }
                                if (a && b) return false;
                            }
                            for (const auto &[current_id, required_id] :
                                 instance.dependencies) {
                                bool has_current = false, has_required = false;
                                for (size_t i = 0; i < ind.chromosome.size();
                                     ++i) {
                                    if (ind.chromosome[i]) {
                                        if (instance.items[i].id == current_id)
                                            has_current = true;
                                        if (instance.items[i].id == required_id)
                                            has_required = true;
                                    }
                                }
                                if (has_current && !has_required) return false;
                            }
                            return true;
                        }()
                    ? "válida"
                    : "inválida")
            << "\n";
    }
} // namespace Fitness
