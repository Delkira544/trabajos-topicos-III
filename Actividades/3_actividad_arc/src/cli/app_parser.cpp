#include "cli/app_parser.hpp"

// Implementación de la función
void
configurar_cli(CLI::App& app, AppConfig& config)
{
  app.description(
    "Ejecucion del algoritmo genetico para resolver el problema de la mochila");

  app.add_option("-i,--instance", config.instance, "Archivos de instancia")
    ->required();

  app
    .add_option("-v,--variant", config.variant,
                "Variante del algoritmo genetico a ejecutar")
    ->required()
    ->check(CLI::IsMember(
      {"sequential", "parallel", "islands_sequential", "islands_parallel"}));

  app
    .add_option("-t,--threads", config.threads,
                "Cantidad de hilos de ejecucion")
    ->required()
    ->check(CLI::PositiveNumber);

  app.add_option("-s,--seed", config.seed, "Semilla para la aleatoriedad")
    ->required()
    ->check(CLI::NonNegativeNumber);

  app
    .add_option("-n,--num-islands", config.num_islands,
                "Número de islas (para algoritmos basados en islas)")
    ->default_val(4)
    ->check(CLI::PositiveNumber);

  app
    .add_option("-m,--migration-interval", config.migration_interval,
                "Intervalo de migración entre islas (en generaciones)")
    ->default_val(5)
    ->check(CLI::PositiveNumber);

  app
    .add_option("--crossover-rate", config.crossover_rate,
                "Tasa de cruzamiento (0.0 a 1.0)")
    ->default_val(0.7f)
    ->check(CLI::Range(0.0f, 1.0f));

  app
    .add_option("--mutation-rate", config.mutation_rate,
                "Tasa de mutación (0.0 a 1.0)")
    ->default_val(0.04f)
    ->check(CLI::Range(0.0f, 1.0f));

  app
    .add_option("--tournament-size", config.tournament_size,
                "Tamaño del torneo para selección")
    ->default_val(3)
    ->check(CLI::PositiveNumber);
  app.add_flag("--verbose", config.verbose,
               "Mostrar información detallada durante la ejecución");
}
