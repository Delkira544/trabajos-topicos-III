#pragma once
#include "ga/operators/ConstraintValidator.hpp"
#include "types.hpp"
#include <limits>
#include <cmath>

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
      for (const auto& incomp : instance.incompatibilities)
      {
        if (individual.chromosome[incomp.first] && individual.chromosome[incomp.second])
        {
          return false;
        }
      }

      // 3. Validar Dependencias (first depende de second)
      for (const auto& dep : instance.dependencies)
      {
        if (individual.chromosome[dep.first] && !individual.chromosome[dep.second])
        {
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
        for (const auto& incomp : instance.incompatibilities)
        {
          int u = incomp.first;
          int v = incomp.second;

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
        // FASE 2: Resolver Dependencias (u depende de v)
        // ---------------------------------------------------------
        for (const auto& dep : instance.dependencies)
        {
          int u = dep.first;
          int v = dep.second;

          // Si 'u' está en la mochila pero su pre-requisito 'v' NO lo está
          if (individual.chromosome[u] && !individual.chromosome[v])
          {
            // Es más seguro apagar 'u' que encender 'v' para no violar el peso
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
      }
    }
  };
} // namespace ga::operators