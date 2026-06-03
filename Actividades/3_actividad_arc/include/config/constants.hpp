#pragma once

namespace Config
{
  namespace GeneticAlgorithm
  {
    constexpr int POPULATION_SIZE         = 120;
    extern int GENERATIONS;  // Dinámico para permitir tuning rápido
    constexpr int CONVERGENCE_GENERATIONS = 50;
    constexpr float PERCENTAGE_CAPACITY   = 0.4f;
    constexpr float MUTATION_RATE         = 0.04f;
    constexpr float CROSSOVER_RATE        = 0.7f;
    constexpr float ELITISM_PERCENTAGE    = 0.05f;
  } // namespace GeneticAlgorithm
  namespace Penalty
  {
    // Penalizaciones dinámicas (no constexpr) para permitir tuning
    extern float WEIGHT_EXCESS_PENALTY;
    extern float VOLUME_EXCESS_PENALTY;
    extern float CATEGORY_VIOLATION_PENALTY;
    extern float INCOMPATIBILITY_PENALTY;
    extern float DEPENDENCY_VIOLATION_PENALTY;
    // Penalizacion correspondiente al valor y las penalizaciones
    constexpr float OBJ_WEIGHT_PENALTY = 0.4f;
    constexpr float PEN_WEIGHT_PENALTY = 0.7f;
  } // namespace Penalty
} // namespace Config
