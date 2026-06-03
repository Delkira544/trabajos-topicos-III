#pragma once
#include "ga/operators/Crossover.hpp"
#include <random>
#include <vector>

namespace ga::operators
{
  /**
   * @brief Implementación de cruzamiento de un punto
   */
  class SinglePointCrossover : public Crossover
  {
  private:
    float crossover_rate;

  public:
    explicit SinglePointCrossover(float rate = 0.7f) : crossover_rate(rate) {}

    Individual apply(const Individual& parent1, const Individual& parent2,
                     std::mt19937& rng) override
    {
      Individual child;
      child.chromosome.resize(parent1.chromosome.size());

      // Verificar si se realiza cruzamiento
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      if (dist(rng) > crossover_rate)
      {
        // Sin cruzamiento, copiar padre aleatorio
        child.chromosome =
          (dist(rng) < 0.5f) ? parent1.chromosome : parent2.chromosome;
        return child;
      }

      // Seleccionar punto de corte aleatorio
      std::uniform_int_distribution<size_t> point_dist(
        0, parent1.chromosome.size() - 1);
      size_t crossover_point = point_dist(rng);

      // Copiar primer segmento del padre1
      for (size_t i = 0; i <= crossover_point; ++i)
      {
        child.chromosome[i] = parent1.chromosome[i];
      }

      // Copiar segundo segmento del padre2
      for (size_t i = crossover_point + 1; i < parent1.chromosome.size(); ++i)
      {
        child.chromosome[i] = parent2.chromosome[i];
      }

      return child;
    }

    float get_crossover_rate() const override { return crossover_rate; }
  };
} // namespace ga::operators
