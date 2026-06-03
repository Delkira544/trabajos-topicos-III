#include "CLI11.hpp"
#include "cli/app_parser.hpp"
#include "data/instance_loader.hpp"
#include "ga/SolverFactory.hpp"
#include "types.hpp"
#include <iostream>

int
main(int argc, char** argv)
{
  CLI::App app;
  AppConfig config;

  configurar_cli(app, config);
  CLI11_PARSE(app, argc, argv);

  // Cargar instancia
  KnapsackInstance instance =
    ga::data::InstanceLoader::load_instance(config.instance);

  // Mostrar información de la instancia
  std::cout << "\n========== Problem Instance ==========" << std::endl;
  std::cout << "Instance: " << config.instance << std::endl;
  std::cout << "Items: " << instance.items.size() << std::endl;
  std::cout << "Max weight: " << instance.max_weight << std::endl;
  std::cout << "Max volume: " << instance.max_volume << std::endl;

  // Mostrar configuración del algoritmo
  std::cout << "\n========== Algorithm Configuration ==========" << std::endl;
  std::cout << "Variant: " << config.variant << std::endl;
  std::cout << "Threads: " << config.threads << std::endl;
  std::cout << "Seed: " << config.seed << std::endl;
  std::cout << "Crossover rate: " << config.crossover_rate << std::endl;
  std::cout << "Mutation rate: " << config.mutation_rate << std::endl;
  std::cout << "Tournament size: " << config.tournament_size << std::endl;

  // Si es un algoritmo basado en islas, mostrar configuración de islas
  if (config.variant == "islands_sequential" ||
      config.variant == "islands_parallel")
  {
    std::cout << "Number of islands: " << config.num_islands << std::endl;
    std::cout << "Migration interval: " << config.migration_interval
              << std::endl;
  }
  std::cout << "====================================\n" << std::endl;

  try
  {
    // Crear configuración del solver
    SolverFactory::SolverConfig solver_config;
    solver_config.num_threads        = config.threads;
    solver_config.num_islands        = config.num_islands;
    solver_config.migration_interval = config.migration_interval;
    solver_config.crossover_rate     = config.crossover_rate;
    solver_config.mutation_rate      = config.mutation_rate;
    solver_config.tournament_size    = config.tournament_size;
    solver_config.seed               = config.seed;
    solver_config.verbose            = config.verbose;

    // Crear y ejecutar solver
    auto solver =
      SolverFactory::create(config.variant, instance, solver_config);
    solver->run();

    // Mostrar resultado final
    Individual best = solver->get_best();
    std::cout << "\n========== Final Result ==========" << std::endl;
    std::cout << "Best fitness: " << best.fitness << std::endl;
    std::cout << "Feasible: " << (best.is_valid ? "Yes" : "No") << std::endl;
    std::cout << "Selected items: ";
    int count = 0;
    for (size_t i = 0; i < best.chromosome.size(); ++i)
    {
      if (best.chromosome[i])
      {
        std::cout << i << " ";
        count++;
      }
    }
    std::cout << "\nTotal items selected: " << count << std::endl;
    std::cout << "==================================\n" << std::endl;
  } catch (const std::exception& e)
  {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
