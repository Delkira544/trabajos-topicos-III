#pragma once
#include "config/constants.hpp"
#include "ga/operators/FitnessEvaluator.hpp"
#include "types.hpp"
#include <algorithm>
#include <unordered_set>
#include <vector>

namespace ga::operators
{
  /**
   * @brief Implementación de evaluador de fitness para el Knapsack
   * multidimensional
   */
  class KnapsackFitness : public FitnessEvaluator
  {

    public:
    void evaluate(Individual& individual,
                  const KnapsackInstance& instance) override
    {
      // Variables para acumular valor, peso y volumen total, y contar
      // categorías
      float total_value  = 0.0f;
      float total_weight = 0.0f;
      float total_volume = 0.0f;

      // Contadores para categorías (si se usan reglas de categorías)
      std::unordered_map<std::string, int> category_counts;

      std::unordered_set<int> selected_ids;

      // Calcular valor, peso y volumen total
      for (size_t i = 0; i < individual.chromosome.size(); ++i)
      {
        if (individual.chromosome[i])
        {
          const Item& item = instance.items[i];
          total_value += item.value;
          total_weight += item.weight;
          total_volume += item.volume;
          category_counts[item.category]++;
          selected_ids.insert(item.id);
        }
      }

      //
      float weight_excess        = 0.0f;
      float volume_excess        = 0.0f;
      int errors_category        = 0;
      int errors_incompatibility = 0;
      int errors_dependency      = 0;

      // Evaluacion de restricciones

      // Capacidad de peso y volumen
      if (total_weight > instance.max_weight)
      {
        weight_excess = total_weight - instance.max_weight;
      }

      if (total_volume > instance.max_volume)
      {
        volume_excess = total_volume - instance.max_volume;
      }

      // Reglas de categorías
      for (const auto& [category, rule] : instance.category_rules)
      {
        int count = 0;
        if (category_counts.count(category))
          count = category_counts.at(category);
        if (count < rule.min || count > rule.max)
          errors_category++;
      }

      // Reglas de incompatibilidad
      for (const auto& rule : instance.incompatibility_rules)
      {
        if (selected_ids.count(rule.item_id_a) &&
            selected_ids.count(rule.item_id_b))
        {
          errors_incompatibility++;
        }
      }

      // Reglas de dependencia
      for (const auto& rule : instance.dependency_rules)
      {
        if (selected_ids.count(rule.item_id_a) &&
            !selected_ids.count(rule.item_id_b))
        {
          errors_dependency++;
        }
      }

      // Calculo de Normaliacion de componentes
      float norm_value =
        (instance.max_value > 0.0f) ? (total_value / instance.max_value) : 0.0f;

      float ratio_weight       = (instance.max_weight > 0.0f)
                                   ? (weight_excess / instance.max_weight)
                                   : 0.0f;
      float norm_excess_weight = std::min(1.0f, ratio_weight);

      float ratio_volume       = (instance.max_volume > 0.0f)
                                   ? (volume_excess / instance.max_volume)
                                   : 0.0f;
      float norm_excess_volume = std::min(1.0f, ratio_volume);

      // P_cat / n_categorias  (cada categoría puede violarse una vez → Pᵢ_max =
      // n_cat)
      float norm_errors_category =
        (!instance.category_rules.empty())
          ? (static_cast<float>(errors_category) /
             static_cast<float>(instance.category_rules.size()))
          : 0.0f;

      // P_incomp / n_incompatibilidades
      float norm_errors_incompatilities =
        (!instance.incompatibility_rules.empty())
          ? (static_cast<float>(errors_incompatibility) /
             static_cast<float>(instance.incompatibility_rules.size()))
          : 0.0f;

      // P_dep / n_dependencias
      float norm_errors_dependencies =
        (!instance.dependency_rules.empty())
          ? (static_cast<float>(errors_dependency) /
             static_cast<float>(instance.dependency_rules.size()))
          : 0.0f;

      float violacion_norm =
        Config::Penalty::WEIGHT_EXCESS_PENALTY * norm_excess_weight +
        Config::Penalty::VOLUME_EXCESS_PENALTY * norm_excess_volume +
        Config::Penalty::CATEGORY_VIOLATION_PENALTY * norm_errors_category +
        Config::Penalty::INCOMPATIBILITY_PENALTY * norm_errors_incompatilities +
        Config::Penalty::DEPENDENCY_VIOLATION_PENALTY *
          norm_errors_dependencies;

      individual.penalty = violacion_norm;
      individual.fitness = Config::Penalty::OBJ_WEIGHT_PENALTY * norm_value -
                           Config::Penalty::PEN_WEIGHT_PENALTY * violacion_norm;

      individual.hard_feasible =
        (weight_excess == 0.0f && volume_excess == 0.0f);

      individual.is_valid =
        individual.hard_feasible && (errors_category == 0) &&
        (errors_incompatibility == 0) && (errors_dependency == 0);
    }

    void evaluate_population(std::vector<Individual>& population,
                             const KnapsackInstance& instance) override
    {
      for (auto& individual : population)
      {
        evaluate(individual, instance);
      }
    }

    Individual get_best(const std::vector<Individual>& population) override
    {
      if (population.empty())
      {
        return Individual();
      }

      return *std::max_element(population.begin(), population.end(),
                               [](const Individual& a, const Individual& b) {
                                 return a.fitness < b.fitness;
                               });
    }
  };
} // namespace ga::operators
