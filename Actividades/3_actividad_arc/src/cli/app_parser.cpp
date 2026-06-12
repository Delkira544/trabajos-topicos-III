#include "cli/app_parser.hpp"

void configurar_cli(CLI::App& app, AppConfig& config)
{
  app.description(
    "Ejecucion del algoritmo genetico para resolver el problema de la mochila");

  app.add_option("-i,--instance", config.instance, "Archivos de instancia")
    ->required();

  app
    .add_option("-v,--variant", config.variant,
                "Variante del algoritmo genetico a ejecutar")
    ->required()
    ->check(CLI::IsMember({
        "sequential",
        "parallel",
        "islands_sequential",
        "islands_parallel",
        "cuda_basic",       // Variante 2: CUDA básico
        "cuda_optimized"    // Variante 3: CUDA optimizado
      }));

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

  app
    .add_option("--block-size", config.block_size,
                "Tamaño de bloque CUDA (hilos por bloque, debe ser múltiplo de 32)")
    ->default_val(128)
    ->check(CLI::IsMember({32, 64, 128, 256, 512, 1024}));

  app
    .add_option("-p,--pop", config.population_size,
                "Tamaño de la poblacion")
    ->default_val(100)
    ->check(CLI::PositiveNumber);

  app
    .add_option("-g,--gen", config.num_generations,
                "Numero total de generaciones")
    ->default_val(300)
    ->check(CLI::PositiveNumber);

  app
    .add_option("--pen-weight", config.penalty_weight,
                "Penalizacion por exceso de peso (default: 0.2)")
    ->default_val(0.2f)
    ->check(CLI::Range(0.0f, 1.0f));

  app
    .add_option("--pen-volume", config.penalty_volume,
                "Penalizacion por exceso de volumen (default: 0.2)")
    ->default_val(0.2f)
    ->check(CLI::Range(0.0f, 1.0f));

  app
    .add_option("--pen-category", config.penalty_category,
                "Penalizacion por violacion de categoria (default: 0.2)")
    ->default_val(0.2f)
    ->check(CLI::Range(0.0f, 1.0f));

  app
    .add_option("--pen-incomp", config.penalty_incomp,
                "Penalizacion por incompatibilidad (default: 0.2)")
    ->default_val(0.2f)
    ->check(CLI::Range(0.0f, 1.0f));

  app
    .add_option("--pen-dep", config.penalty_dep,
                "Penalizacion por dependencia no satisfecha (default: 0.2)")
    ->default_val(0.2f)
    ->check(CLI::Range(0.0f, 1.0f));

  app.add_flag("--verbose", config.verbose,
               "Mostrar información detallada durante la ejecución");
}