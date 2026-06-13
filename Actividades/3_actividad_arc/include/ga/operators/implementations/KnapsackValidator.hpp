#pragma once
#include "ga/operators/ConstraintValidator.hpp"
#include "types.hpp"
#include <limits>
#include <cmath>
#include <unordered_map>

namespace ga::operators
{
  /**
   * @brief Implementación de validador de restricciones para el Knapsack (con Reglas Complejas)
   */
  class KnapsackValidator : public ConstraintValidator
  {
  public:
    bool is_feasible(const Individual& individual,
                     const KnapsackInstance& instance) override
    {
      float total_weight = 0.0f;
      float total_volume = 0.0f;

      // 1. Validar Capacidades
      for (size_t i = 0; i < individual.chromosome.size(); ++i)
      {
        if (individual.chromosome[i])
        {
          total_weight += instance.items[i].weight;
          total_volume += instance.items[i].volume;
        }
      }

      if (total_weight > instance.max_weight || total_volume > instance.max_volume) {
          return false;
      }

      // 2. Validar Incompatibilidades
      for (const auto& incomp : instance.incompatibility_rules)
      {
        if (individual.chromosome[incomp.item_id_a] && individual.chromosome[incomp.item_id_b])
        {
          return false;
        }
      }

      // 3. Validar Dependencias (A depende de B)
      for (const auto& dep : instance.dependency_rules)
      {
        if (individual.chromosome[dep.item_id_a] && !individual.chromosome[dep.item_id_b])
        {
          return false;
        }
      }

      // CORRECCIÓN 5: Validar Reglas de Categoría
      // Contar cuántos ítems de cada categoría están presentes en el cromosoma
      // y verificar que respeten los límites min/max
      std::unordered_map<std::string, int> category_counts;
      
      for (size_t i = 0; i < individual.chromosome.size(); ++i)
      {
        if (individual.chromosome[i])
        {
          const std::string& cat = instance.items[i].category;
          category_counts[cat]++;
        }
      }

      // Verificar que cada categoría respete sus límites
      for (const auto& [category, count] : category_counts)
      {
        auto it = instance.category_rules.find(category);
        if (it != instance.category_rules.end())
        {
          const CategoryRule& rule = it->second;
          // Si el conteo está fuera de [min, max], la solución es infactible
          if (count < rule.min || count > rule.max)
          {
            return false;
          }
        }
      }

      // También verificar categorías que NO están presentes pero tienen min > 0
      for (const auto& [category, rule] : instance.category_rules)
      {
        if (category_counts.find(category) == category_counts.end() && rule.min > 0)
        {
          // La categoría no está presente pero exige al menos rule.min ítems
          return false;
        }
      }

      return true;
    }

    void repair(Individual& individual,
                const KnapsackInstance& instance) override
    {
      bool changed = true;
      int max_iters = 100; // Evitar ciclos infinitos por reglas circulares
      int iter = 0;

      // Iteramos hasta que la mochila sea factible o alcancemos el límite de intentos
      while (changed && iter < max_iters && !is_feasible(individual, instance))
      {
        changed = false;
        iter++;

        // ---------------------------------------------------------
        // FASE 1: Resolver Incompatibilidades
        // ---------------------------------------------------------
        for (const auto& incomp : instance.incompatibility_rules)
        {
          int u = incomp.item_id_a;
          int v = incomp.item_id_b;

          if (individual.chromosome[u] && individual.chromosome[v])
          {
            // Ambos están seleccionados. Apagamos el menos eficiente.
            float cost_u = instance.items[u].weight + instance.items[u].volume;
            float eff_u = (cost_u > 0) ? instance.items[u].value / cost_u : 0.0f;

            float cost_v = instance.items[v].weight + instance.items[v].volume;
            float eff_v = (cost_v > 0) ? instance.items[v].value / cost_v : 0.0f;

            if (eff_u < eff_v) {
                individual.chromosome[u] = false;
            } else {
                individual.chromosome[v] = false;
            }
            changed = true;
          }
        }

        // ---------------------------------------------------------
        // FASE 2: Resolver Dependencias (A depende de B)
        // ---------------------------------------------------------
        for (const auto& dep : instance.dependency_rules)
        {
          int u = dep.item_id_a;
          int v = dep.item_id_b;

          // Si 'u' está en la mochila pero su pre-requisito 'v' NO lo está
          if (individual.chromosome[u] && !individual.chromosome[v])
          {
            // Apagamos 'u' para no violar la dependencia (y evitamos subir peso encendiendo 'v')
            individual.chromosome[u] = false;
            changed = true;
          }
        }

        // ---------------------------------------------------------
        // FASE 3: Resolver Capacidades (Peso y Volumen)
        // ---------------------------------------------------------
        float total_weight = 0.0f;
        float total_volume = 0.0f;

        for (size_t i = 0; i < individual.chromosome.size(); ++i)
        {
          if (individual.chromosome[i])
          {
            total_weight += instance.items[i].weight;
            total_volume += instance.items[i].volume;
          }
        }

        // Mientras nos pasemos de capacidad, eliminamos el peor ítem
        while (total_weight > instance.max_weight || total_volume > instance.max_volume)
        {
          int worst_idx = -1;
          float worst_eff = std::numeric_limits<float>::max();

          for (size_t i = 0; i < individual.chromosome.size(); ++i)
          {
            if (individual.chromosome[i])
            {
              float cost = instance.items[i].weight + instance.items[i].volume;
              float eff = (cost > 0) ? instance.items[i].value / cost : 0.0f;
              if (eff < worst_eff)
              {
                worst_eff = eff;
                worst_idx = static_cast<int>(i);
              }
            }
          }

          if (worst_idx >= 0)
          {
            individual.chromosome[worst_idx] = false;
            total_weight -= instance.items[worst_idx].weight;
            total_volume -= instance.items[worst_idx].volume;
            changed = true;
          }
          else
          {
            break; // Ya no hay ítems que quitar
          }
        }

        // ---------------------------------------------------------
        // FASE 4: Resolver Reglas de Categoría
        // ---------------------------------------------------------
        std::unordered_map<std::string, int> cat_counts;
        for (size_t i = 0; i < individual.chromosome.size(); ++i)
        {
          if (individual.chromosome[i])
          {
            cat_counts[instance.items[i].category]++;
          }
        }

        // Apagar ítems si se excede el máximo de una categoría
        for (auto& [category, count] : cat_counts)
        {
          auto it = instance.category_rules.find(category);
          if (it != instance.category_rules.end())
          {
            int max_allowed = it->second.max;
            if (count > max_allowed)
            {
              // Eliminar los ítems menos eficientes de esta categoría
              std::vector<std::pair<float, int>> cat_items;
              for (size_t i = 0; i < individual.chromosome.size(); ++i)
              {
                if (individual.chromosome[i] && instance.items[i].category == category)
                {
                  float cost = instance.items[i].weight + instance.items[i].volume;
                  float eff = (cost > 0) ? instance.items[i].value / cost : 0.0f;
                  cat_items.push_back({eff, static_cast<int>(i)});
                }
              }
              // Ordenar por eficiencia (menor primero)
              std::sort(cat_items.begin(), cat_items.end());
              
              // Eliminar desde el menos eficiente hasta alcanzar max_allowed
              int to_remove = count - max_allowed;
              for (int j = 0; j < to_remove && j < static_cast<int>(cat_items.size()); ++j)
              {
                int item_id = cat_items[j].second;
                individual.chromosome[item_id] = false;
                changed = true;
              }
            }
          }
        }

        // Encender ítems si no alcanzamos el mínimo de una categoría (opcional, conservador)
        // En la mayoría de casos, apagar es más seguro que encender
      }
    }
  };
} // namespace ga::operators