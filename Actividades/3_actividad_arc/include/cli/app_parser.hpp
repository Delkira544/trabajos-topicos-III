#pragma once
#include "CLI11.hpp"
#include <string>

struct AppConfig
{
  std::string instance;
  std::string variant;
  int threads;
  int seed;
  int num_islands        = 4;
  int migration_interval = 5;
  float crossover_rate   = 0.7f;
  float mutation_rate    = 0.04f;
  int tournament_size    = 3;
  int block_size         = 128;   // tamaño de bloque CUDA
  int population_size    = 100;   // Tamaño de la población
  int num_generations    = 300;   // Número total de generaciones
  float penalty_weight   = 0.2f;  // Penalización por exceso de peso
  float penalty_volume   = 0.2f;  // Penalización por exceso de volumen
  float penalty_category = 0.2f;  // Penalización por violación de categoría
  float penalty_incomp   = 0.2f;  // Penalización por incompatibilidad
  float penalty_dep      = 0.2f;  // Penalización por dependencia
  bool verbose           = false;
};

void configurar_cli(CLI::App& app, AppConfig& config);