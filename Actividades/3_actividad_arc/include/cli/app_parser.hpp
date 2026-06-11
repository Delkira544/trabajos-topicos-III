#pragma once
#include "CLI11.hpp"
#include <string>

struct AppConfig
{
  std::string instance;
  std::string variant;
  short threads;
  int seed;
  int num_islands        = 4;
  int migration_interval = 5;
  float crossover_rate   = 0.7f;
  float mutation_rate    = 0.04f;
  int tournament_size    = 3;
  int block_size         = 128;   // tamaño de bloque CUDA
  bool verbose           = false;
};

void configurar_cli(CLI::App& app, AppConfig& config);