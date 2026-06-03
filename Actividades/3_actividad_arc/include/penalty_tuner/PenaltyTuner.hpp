#pragma once
#include "config/constants.hpp"
#include "data/instance_loader.hpp"
#include "ga/SolverFactory.hpp"
#include "types.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Configuración de penalidades para tuning
 */
struct PenaltyConfig
{
  float weight_excess;
  float volume_excess;
  float category_violation;
  float incompatibility;
  float dependency_violation;

  /**
   * @brief Verifica que los parámetros sumen 1.0 (con tolerancia)
   */
  bool is_valid() const
  {
    float sum = weight_excess + volume_excess + category_violation +
                incompatibility + dependency_violation;
    return std::abs(sum - 1.0f) < 0.001f;
  }

  /**
   * @brief Aplica esta configuración a las constantes globales
   */
  void apply() const
  {
    Config::Penalty::WEIGHT_EXCESS_PENALTY = weight_excess;
    Config::Penalty::VOLUME_EXCESS_PENALTY = volume_excess;
    Config::Penalty::CATEGORY_VIOLATION_PENALTY = category_violation;
    Config::Penalty::INCOMPATIBILITY_PENALTY = incompatibility;
    Config::Penalty::DEPENDENCY_VIOLATION_PENALTY = dependency_violation;
  }

  /**
   * @brief Retorna una representación en string
   */
  std::string to_string() const
  {
    char buffer[256];
    snprintf(buffer, sizeof(buffer),
             "WE=%.2f VE=%.2f CV=%.2f IC=%.2f DV=%.2f", weight_excess,
             volume_excess, category_violation, incompatibility,
             dependency_violation);
    return std::string(buffer);
  }
};

/**
 * @brief Resultado de una evaluación de configuración
 */
struct SearchResult
{
  PenaltyConfig config;
  float avg_validity_rate = 0.0f;
  float avg_best_fitness = 0.0f;
  int avg_generations_to_valid = 0;
  std::vector<float> seed_validity_rates;
  std::vector<float> seed_fitness_values;
  std::vector<int> seed_generations;

  /**
   * @brief Compara dos resultados por tasa de validez (descendente)
   */
  bool operator<(const SearchResult& other) const
  {
    if (std::abs(avg_validity_rate - other.avg_validity_rate) > 0.0001f)
    {
      return avg_validity_rate > other.avg_validity_rate;
    }
    return avg_best_fitness > other.avg_best_fitness;
  }
};

/**
 * @brief Herramienta para búsqueda de parámetros de penalización óptimos
 *
 * Implementa grid search sobre el espacio de penalizaciones con restricción
 * de suma = 1.0. Ejecuta cada configuración con múltiples semillas y
 * optimiza por tasa de soluciones válidas.
 */
class PenaltyTuner
{
  private:
  KnapsackInstance instance;
  std::vector<SearchResult> results;
  float step_size;
  std::string instance_path;
  int num_seeds;
  size_t tuning_generations = 50;  // Reducidas para tuning rápido

  public:
  /**
   * @brief Constructor
   * @param inst_path Ruta a los archivos de instancia
   * @param step Tamaño del paso para discretización (ej: 0.10)
   */
  PenaltyTuner(const std::string& inst_path, float step = 0.10f)
      : step_size(step), instance_path(inst_path), num_seeds(2)
  {
    std::cout << "[LOG] Loading instance from: " << inst_path << std::endl;
    std::cout.flush();
    instance = ga::data::InstanceLoader::load_instance(instance_path);
    std::cout << "[LOG] Instance loaded successfully" << std::endl;
    std::cout.flush();
  }

  /**
   * @brief Genera todas las combinaciones válidas de penalizaciones
   * @return Vector de configuraciones donde cada una suma 1.0
   */
  std::vector<PenaltyConfig> generate_grid();

  /**
   * @brief Evalúa una configuración con una semilla específica
   * @param config Configuración de penalidades
   * @param seed Semilla del RNG
   * @return Tasa de validez (0-1) y fitness del mejor individuo
   */
  std::pair<float, std::pair<float, int>> evaluate_config(
    const PenaltyConfig& config, int seed);

  /**
   * @brief Ejecuta búsqueda completa de grid
   * @param num_seeds Número de semillas por configuración
   */
  void run_search(int num_seeds = 2);

  /**
   * @brief Guarda resultados en archivo CSV
   * @param output_file Ruta del archivo de salida
   */
  void save_results(const std::string& output_file);

  /**
   * @brief Imprime las Top N mejores configuraciones
   * @param n Número de configuraciones a mostrar
   */
  void print_top_n(int n = 30);

  /**
   * @brief Retorna los resultados ordenados
   */
  const std::vector<SearchResult>& get_results() const { return results; }

  /**
   * @brief Establece el número de generaciones para tuning (default: 50)
   */
  void set_tuning_generations(size_t gen) { tuning_generations = gen; }

  /**
   * @brief Retorna el número de generaciones para tuning
   */
  size_t get_tuning_generations() const { return tuning_generations; }
};
