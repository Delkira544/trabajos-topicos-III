#include "config/constants.hpp"

namespace Config
{
  namespace GeneticAlgorithm
  {
    // Inicializar generaciones con valor por defecto
    int GENERATIONS = 300;
  } // namespace GeneticAlgorithm

  namespace Penalty
  {
    // Inicializar penalizaciones con valores por defecto
    float WEIGHT_EXCESS_PENALTY        = 0.20f;
    float VOLUME_EXCESS_PENALTY        = 0.20f;
    float CATEGORY_VIOLATION_PENALTY   = 0.05f;
    float INCOMPATIBILITY_PENALTY      = 0.30f;
    float DEPENDENCY_VIOLATION_PENALTY = 0.25f;
  } // namespace Penalty
} // namespace Config
