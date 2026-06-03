#include "penalty_tuner/PenaltyTuner.hpp"
#include "ga/BaseGA.hpp"
#include <cmath>
#include <iomanip>

/**
 * @brief Genera todas las combinaciones válidas de penalizaciones
 * donde la suma = 1.0
 */
std::vector<PenaltyConfig> PenaltyTuner::generate_grid()
{
  std::vector<PenaltyConfig> configs;

  std::cout << "[LOG] Starting grid generation with step size: " << step_size
            << std::endl;
  std::cout.flush();

  size_t count = 0;

  // Iterar sobre los 5 parámetros con restricción suma = 1.0
  for (float p1 = 0.0f; p1 <= 1.0f; p1 += step_size)
  {
    for (float p2 = 0.0f; p2 <= 1.0f - p1; p2 += step_size)
    {
      for (float p3 = 0.0f; p3 <= 1.0f - p1 - p2; p3 += step_size)
      {
        for (float p4 = 0.0f; p4 <= 1.0f - p1 - p2 - p3; p4 += step_size)
        {
          float p5 = 1.0f - p1 - p2 - p3 - p4;

          // Redondear p5 a múltiplos de step_size para evitar artefactos
          p5 = std::round(p5 / step_size) * step_size;

          if (std::abs((p1 + p2 + p3 + p4 + p5) - 1.0f) < 0.001f)
          {
            PenaltyConfig config;
            config.weight_excess          = p1;
            config.volume_excess          = p2;
            config.category_violation     = p3;
            config.incompatibility        = p4;
            config.dependency_violation   = p5;

            configs.push_back(config);
            count++;
          }
        }
      }
    }
  }

  std::cout << "[LOG] Grid generation completed. Total configurations: " << count
            << std::endl;
  std::cout.flush();

  return configs;
}

/**
 * @brief Evalúa una configuración con una semilla específica
 */
std::pair<float, std::pair<float, int>>
PenaltyTuner::evaluate_config(const PenaltyConfig& config, int seed)
{
  std::cout << "[LOG] Evaluating config with seed " << seed << ": "
            << config.to_string() << std::endl;
  std::cout.flush();

  // Aplicar configuración de penalidades
  config.apply();

  std::cout << "[LOG] Penalties applied successfully" << std::endl;
  std::cout.flush();

  // Guardar número original de generaciones
  int original_generations = Config::GeneticAlgorithm::GENERATIONS;

  // Establecer generaciones reducidas para tuning
  Config::GeneticAlgorithm::GENERATIONS = tuning_generations;

  std::cout << "[LOG] Set generations for tuning: " << tuning_generations
            << std::endl;
  std::cout.flush();

  // Crear solver
  SolverFactory::SolverConfig solver_config;
  solver_config.num_threads        = 1;
  solver_config.num_islands        = 4;
  solver_config.migration_interval = 5;
  solver_config.crossover_rate     = 0.7f;
  solver_config.mutation_rate      = 0.04f;
  solver_config.tournament_size    = 3;
  solver_config.seed               = seed;
  solver_config.verbose            = false; // Sin output durante tuning

  std::cout << "[LOG] Solver config created" << std::endl;
  std::cout.flush();

  auto solver =
    SolverFactory::create("sequential", instance, solver_config);

  std::cout << "[LOG] Solver created, starting run..." << std::endl;
  std::cout.flush();

  // Ejecutar solver
  solver->run();

  std::cout << "[LOG] Solver finished running" << std::endl;
  std::cout.flush();

  // Restaurar número original de generaciones
  Config::GeneticAlgorithm::GENERATIONS = original_generations;

  std::cout << "[LOG] Restored generations to: " << original_generations
            << std::endl;
  std::cout.flush();

  // Obtener resultado
  Individual best = solver->get_best();

  std::cout << "[LOG] Best individual obtained. Is valid: "
            << (best.is_valid ? "YES" : "NO") << ", Fitness: " << best.fitness
            << std::endl;
  std::cout.flush();

  // Calcular métrica: tasa de validez
  // En este caso, si llegó a solución válida = 1.0, sino = 0.0
  float validity_rate = best.is_valid ? 1.0f : 0.0f;

  // Retornar: (tasa_validez, (fitness, generaciones))
  // Nota: generaciones obtenidas del historial de fitness
  const auto& fitness_history = solver->get_fitness_history();
  int generations             = fitness_history.size();

  std::cout << "[LOG] Config evaluation complete. Validity: " << validity_rate
            << ", Generations: " << generations << std::endl;
  std::cout.flush();

  return {validity_rate, {best.fitness, generations}};
}

/**
 * @brief Ejecuta búsqueda completa de grid
 */
void PenaltyTuner::run_search(int num_seeds_param)
{
  num_seeds = num_seeds_param;

  std::cout << "\n╔════════════════════════════════════════════════════════════════╗"
            << std::endl;
  std::cout << "║     PENALTY TUNING: Grid Search with Normalization            ║"
            << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════════╝"
            << std::endl;
  std::cout.flush();

  std::cout << "\n[LOG] Starting penalty tuner..." << std::endl;
  std::cout << "[LOG] Instance path: " << instance_path << std::endl;
  std::cout << "[LOG] Instance loaded with " << instance.items.size() << " items"
            << std::endl;
  std::cout << "[LOG] Step size: " << step_size << std::endl;
  std::cout << "[LOG] Seeds per configuration: " << num_seeds << std::endl;
  std::cout.flush();

  // Generar grid
  std::cout << "\n[LOG] Generating grid search space..." << std::endl;
  std::cout.flush();

  auto configs = generate_grid();

  std::cout << "\n[LOG] Grid generation completed" << std::endl;
  std::cout << "[LOG] Total configurations to evaluate: " << configs.size()
            << std::endl;
  std::cout << "[LOG] Total solver runs: " << (configs.size() * num_seeds)
            << std::endl;
  std::cout << "[LOG] Starting configuration evaluation loop..." << std::endl;
  std::cout.flush();

  // Evaluar cada configuración
  size_t current = 0;
  for (const auto& config : configs)
  {
    current++;

    float total_validity    = 0.0f;
    float total_fitness     = 0.0f;
    int total_generations   = 0;

    SearchResult result;
    result.config = config;

    std::cout << "\n[PROGRESS] Config " << current << "/" << configs.size()
              << ": " << config.to_string() << std::endl;
    std::cout.flush();

    // Ejecutar con múltiples semillas
    for (int seed = 0; seed < num_seeds; ++seed)
    {
      std::cout << "[LOG] Seed " << (seed + 1) << "/" << num_seeds
                << " for this config..." << std::endl;
      std::cout.flush();

      auto [validity, fitness_gen] = evaluate_config(config, seed);
      auto [fitness, generations]  = fitness_gen;

      result.seed_validity_rates.push_back(validity);
      result.seed_fitness_values.push_back(fitness);
      result.seed_generations.push_back(generations);

      total_validity += validity;
      total_fitness += fitness;
      total_generations += generations;

      std::cout << "[SEED_RESULT] Seed " << (seed + 1) << ": Validity="
                << validity << ", Fitness=" << fitness
                << ", Generations=" << generations << std::endl;
      std::cout.flush();
    }

    // Calcular promedios
    result.avg_validity_rate        = total_validity / num_seeds;
    result.avg_best_fitness         = total_fitness / num_seeds;
    result.avg_generations_to_valid = total_generations / num_seeds;

    std::cout << "[CONFIG_RESULT] Config average - Validity: "
              << result.avg_validity_rate << ", Fitness: "
              << result.avg_best_fitness << ", Generations: "
              << result.avg_generations_to_valid << std::endl;
    std::cout.flush();

    results.push_back(result);
  }

  // Ordenar resultados por tasa de validez
  std::cout << "\n[LOG] Sorting results by validity rate..." << std::endl;
  std::cout.flush();

  std::sort(results.begin(), results.end());

  std::cout << "[LOG] Results sorted successfully!" << std::endl;
  std::cout << "[LOG] Search completed!" << std::endl;
  std::cout.flush();
}

/**
 * @brief Guarda resultados en CSV
 */
void PenaltyTuner::save_results(const std::string& output_file)
{
  std::cout << "\n[LOG] Opening file for writing: " << output_file << std::endl;
  std::cout.flush();

  std::ofstream file(output_file);

  if (!file.is_open())
  {
    std::cerr << "[ERROR] Could not open file " << output_file << std::endl;
    std::cout.flush();
    return;
  }

  std::cout << "[LOG] File opened successfully" << std::endl;
  std::cout.flush();

  // Header CSV
  std::cout << "[LOG] Writing CSV header..." << std::endl;
  std::cout.flush();

  file << "rank,weight_excess,volume_excess,category_violation,"
          "incompatibility,dependency_violation,avg_validity_rate,"
          "avg_best_fitness,avg_generations_to_valid";

  for (int i = 0; i < num_seeds; ++i)
  {
    file << ",seed" << i + 1 << "_validity,seed" << i + 1
         << "_fitness,seed" << i + 1 << "_generations";
  }
  file << "\n";

  std::cout << "[LOG] Header written" << std::endl;
  std::cout.flush();

  // Datos
  std::cout << "[LOG] Writing " << results.size() << " results to CSV..."
            << std::endl;
  std::cout.flush();

  for (size_t i = 0; i < results.size(); ++i)
  {
    const auto& result = results[i];
    file << i + 1 << ","
         << std::fixed << std::setprecision(2) << result.config.weight_excess
         << "," << result.config.volume_excess << ","
         << result.config.category_violation << ","
         << result.config.incompatibility << ","
         << result.config.dependency_violation << ","
         << result.avg_validity_rate << "," << result.avg_best_fitness << ","
         << result.avg_generations_to_valid;

    for (size_t j = 0; j < result.seed_validity_rates.size(); ++j)
    {
      file << "," << result.seed_validity_rates[j] << ","
           << result.seed_fitness_values[j] << ","
           << result.seed_generations[j];
    }
    file << "\n";

    if ((i + 1) % 100 == 0)
    {
      std::cout << "[LOG] Written " << (i + 1) << " results..." << std::endl;
      std::cout.flush();
    }
  }

  file.close();
  std::cout << "[LOG] File closed successfully" << std::endl;
  std::cout << "[SUCCESS] Results saved to: " << output_file << std::endl;
  std::cout.flush();
}

/**
 * @brief Imprime las Top N mejores configuraciones
 */
void PenaltyTuner::print_top_n(int n)
{
  std::cout << "\n[LOG] Preparing to display top " << n << " configurations..."
            << std::endl;
  std::cout.flush();

  std::cout << "\n╔════════════════════════════════════════════════════════════════╗"
            << std::endl;
  std::cout << "║                  TOP " << std::setw(2) << n
            << " PENALTY CONFIGURATIONS                    ║" << std::endl;
  std::cout << "║         Metric: Validity Rate (tasa de soluciones válidas)     ║"
            << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════════╝\n"
            << std::endl;
  std::cout.flush();

  // Header tabla
  std::cout << std::left << std::setw(5) << "Rank" << std::setw(6) << "WE"
            << std::setw(6) << "VE" << std::setw(6) << "CV"
            << std::setw(6) << "IC" << std::setw(6) << "DV"
            << std::setw(10) << "Valid%" << std::setw(12) << "Fitness"
            << std::setw(8) << "Gen" << std::endl;
  std::cout << std::string(68, '-') << std::endl;
  std::cout.flush();

  // Top N filas
  int limit = std::min(n, static_cast<int>(results.size()));

  std::cout << "[LOG] Displaying " << limit << " configurations..." << std::endl;
  std::cout.flush();

  for (int i = 0; i < limit; ++i)
  {
    const auto& result = results[i];
    std::cout << std::left << std::setw(5) << (i + 1);
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::setw(6) << result.config.weight_excess;
    std::cout << std::setw(6) << result.config.volume_excess;
    std::cout << std::setw(6) << result.config.category_violation;
    std::cout << std::setw(6) << result.config.incompatibility;
    std::cout << std::setw(6) << result.config.dependency_violation;
    std::cout << std::setw(9) << (result.avg_validity_rate * 100.0f) << "%";
    std::cout << std::setw(12) << result.avg_best_fitness;
    std::cout << std::setw(8) << result.avg_generations_to_valid;
    std::cout << std::endl;
  }

  std::cout << std::endl;
  std::cout << "[LOG] Top " << limit << " configurations displayed" << std::endl;
  std::cout.flush();
}
