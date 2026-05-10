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

        // 5. Normalización de cada componente a [0,1]

        // valor_norm = Σvᵢxᵢ / Σvᵢ  (máximo teórico = suma de todos los valores)
        float norm_value = (instance.max_value > 0.0f)
                               ? (total_value / instance.max_value)
                               : 0.0f;

        // Mejora #1 — penalización CUADRÁTICA sobre la capacidad para hard
        // constraints. Normalizamos por la CAPACIDAD (no por el exceso máx)
        // y amplificamos: norm = min(1, ratio² · 100)
        //   1% exceso → 0.01;  2.5% → 0.0625;  5% → 0.25;  ≥10% → 1.0
        // Así un exceso pequeño ya genera presión significativa contra el
        // valor objetivo, en vez de quedar diluido entre miles de pesos.
        constexpr float HARD_AMPLIFIER = 100.0f;
        float ratio_peso = (instance.knapsack.max_weight > 0.0f)
                               ? (exceso_peso / instance.knapsack.max_weight)
                               : 0.0f;
        float norm_exceso_peso =
            std::min(1.0f, ratio_peso * ratio_peso * HARD_AMPLIFIER);

        float ratio_volumen = (instance.knapsack.max_volume > 0.0f)
                                  ? (exceso_volumen / instance.knapsack.max_volume)
                                  : 0.0f;
        float norm_exceso_volumen =
            std::min(1.0f, ratio_volumen * ratio_volumen * HARD_AMPLIFIER);

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

        // hard_feasible: peso y volumen ambos OK (sin tocar las soft).
        // is_valid: TODAS las restricciones (hard + soft) OK.
        ind.hard_feasible = (exceso_peso <= 0.0f) && (exceso_volumen <= 0.0f);
        ind.is_valid = ind.hard_feasible
                       && (errores_categoria == 0)
                       && (errores_incompatibilidad == 0)
                       && (errores_dependencia == 0);
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
