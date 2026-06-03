#include "CLI11.hpp"
#include "penalty_tuner/PenaltyTuner.hpp"
#include <iostream>

int main(int argc, char** argv)
{
  std::cout << "[INIT] Penalty Tuner starting..." << std::endl;
  std::cout.flush();

  CLI::App app("Penalty Parameter Tuner - Grid Search with Normalization");

  std::string instance_path;
  float step_size = 0.10f;
  int num_seeds   = 2;
  int top_n       = 30;
  size_t generations = 50;  // Reducidas para tuning rápido
  std::string output_file = "results/penalty_search_results.csv";

  std::cout << "[LOG] Setting up CLI options..." << std::endl;
  std::cout.flush();

  app.add_option("-i,--instance", instance_path, "Path to problem instance")
    ->required();

  app.add_option("-s,--step", step_size,
                 "Step size for grid discretization (default: 0.10)")
    ->default_val(0.10f);

  app.add_option("-k,--seeds", num_seeds,
                 "Number of seeds per configuration (default: 2)")
    ->default_val(2);

  app.add_option("-n,--top-n", top_n,
                 "Show top N configurations (default: 30)")
    ->default_val(30);

  app.add_option("-g,--generations", generations,
                 "Generations for each AG during tuning (default: 50)")
    ->default_val(50);

  app.add_option("-o,--output", output_file,
                 "Output CSV file for results (default: results/penalty_search_results.csv)")
    ->default_val("results/penalty_search_results.csv");

  std::cout << "[LOG] Parsing command line arguments..." << std::endl;
  std::cout.flush();

  CLI11_PARSE(app, argc, argv);

  std::cout << "[LOG] CLI parsing completed" << std::endl;
  std::cout << "[LOG] Parameters:" << std::endl;
  std::cout << "  - Instance path: " << instance_path << std::endl;
  std::cout << "  - Step size: " << step_size << std::endl;
  std::cout << "  - Number of seeds: " << num_seeds << std::endl;
  std::cout << "  - Top N: " << top_n << std::endl;
  std::cout << "  - Generations per AG: " << generations << std::endl;
  std::cout << "  - Output file: " << output_file << std::endl;
  std::cout.flush();

  try
  {
    // Crear tuner
    std::cout << "\n[LOG] Creating PenaltyTuner instance..." << std::endl;
    std::cout.flush();

    PenaltyTuner tuner(instance_path, step_size);

    // Establecer generaciones para tuning
    tuner.set_tuning_generations(generations);

    std::cout << "[LOG] PenaltyTuner created successfully" << std::endl;
    std::cout << "[LOG] Tuning generations set to: " << generations << std::endl;
    std::cout.flush();

    // Ejecutar búsqueda
    std::cout << "\n[LOG] Starting grid search..." << std::endl;
    std::cout.flush();

    tuner.run_search(num_seeds);

    std::cout << "\n[LOG] Grid search completed" << std::endl;
    std::cout.flush();

    // Mostrar resultados
    std::cout << "\n[LOG] Displaying top " << top_n << " configurations..."
              << std::endl;
    std::cout.flush();

    tuner.print_top_n(top_n);

    // Guardar CSV
    std::cout << "\n[LOG] Saving results to CSV..." << std::endl;
    std::cout.flush();

    tuner.save_results(output_file);

    std::cout << "\n[SUCCESS] Penalty tuner completed successfully!" << std::endl;
    std::cout.flush();

    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << "[ERROR] Exception caught: " << e.what() << std::endl;
    std::cout.flush();
    return 1;
  }
}
