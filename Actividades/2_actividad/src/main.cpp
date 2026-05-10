#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include "genetic_algorithm.hpp"
#include "island_model.hpp"
#include "fitness.hpp"
#include "instance_loader.hpp"
#include "benchmark.hpp"

struct Config {
    std::string instance;
    std::string variant = "standard";
    int threads = 1;
    int seed = 0;
    int population_size = 100;
    int generations = 50;
    float mutation_rate = 0.03f;
    bool verbose = false;
    float convergence_threshold = 0.00001f;
    std::string report_file;
    int num_islands = 4;
    int population_per_island = 25;
    int migration_frequency = 10;
    int num_migrants = 2;
    std::string migration_topology = "ring";
};

void print_usage(const char *program) {
    std::cout
        << "Uso: " << program << " [opciones]\n"
        << "  --instance <dir>       Directorio de la instancia (requerido)\n"
        << "  --variant <type>       Variante: standard, islands (default: "
           "standard)\n"
        << "  --threads <n>          Número de hilos (default: 1)\n"
        << "  --seed <n>             Semilla aleatoria (default: 0)\n"
        << "  --population <n>       Tamaño de población (default: 100)\n"
        << "  --generations <n>      Número de generaciones (default: 50)\n"
        << "  --mutation-rate <f>    Tasa de mutación (default: 0.01)\n"
        << "  --convergence-threshold <f> Umbral de convergencia (default: "
           "0.001)\n"
        << "  --report-file <path>   Archivo para reporte CSV (opcional)\n"
        << "  --num-islands <n>      Número de islas (default: 4)\n"
        << "  --pop-per-island <n>   Población por isla (default: 25)\n"
        << "  --migration-freq <n>   Frecuencia de migración (default: 10)\n"
        << "  --num-migrants <n>     Cantidad de migrantes (default: 2)\n"
        << "  --topology <type>      Topología de migración: ring, random "
           "(default: ring)\n"
        << "  --benchmark            Ejecutar benchmarks en lugar de una "
           "corrida simple\n"
        << "  --config <n>           Configuración de benchmark a utilizar "
           "(1-4) (default: 1)\n"
        << "  --verbose              Mostrar información detallada\n"
        << "  --help, -h             Mostrar esta ayuda\n";
}

int main(int argc, char *argv[]) {
    Config conf;
    bool is_benchmark = false;
    int bench_config_id = 1;

    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];

        if (flag == "--instance" && i + 1 < argc)
            conf.instance = argv[++i];
        else if (flag == "--variant" && i + 1 < argc)
            conf.variant = argv[++i];
        else if (flag == "--threads" && i + 1 < argc)
            conf.threads = std::stoi(argv[++i]);
        else if (flag == "--seed" && i + 1 < argc)
            conf.seed = std::stoi(argv[++i]);
        else if (flag == "--population" && i + 1 < argc)
            conf.population_size = std::stoi(argv[++i]);
        else if (flag == "--generations" && i + 1 < argc)
            conf.generations = std::stoi(argv[++i]);
        else if (flag == "--mutation-rate" && i + 1 < argc)
            conf.mutation_rate = std::stof(argv[++i]);
        else if (flag == "--convergence-threshold" && i + 1 < argc)
            conf.convergence_threshold = std::stof(argv[++i]);
        else if (flag == "--report-file" && i + 1 < argc)
            conf.report_file = argv[++i];
        else if (flag == "--num-islands" && i + 1 < argc)
            conf.num_islands = std::stoi(argv[++i]);
        else if (flag == "--pop-per-island" && i + 1 < argc)
            conf.population_per_island = std::stoi(argv[++i]);
        else if (flag == "--migration-freq" && i + 1 < argc)
            conf.migration_frequency = std::stoi(argv[++i]);
        else if (flag == "--num-migrants" && i + 1 < argc)
            conf.num_migrants = std::stoi(argv[++i]);
        else if (flag == "--topology" && i + 1 < argc)
            conf.migration_topology = argv[++i];
        else if (flag == "--benchmark")
            is_benchmark = true;
        else if (flag == "--config" && i + 1 < argc)
            bench_config_id = std::stoi(argv[++i]);
        else if (flag == "--verbose")
            conf.verbose = true;
        else if (flag == "--help" || flag == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (is_benchmark) {
        BenchmarkConfig bconf;
        bconf.config_id = bench_config_id;
        bconf.instances = {"data/small", "data/medium", "data/large"};
        bconf.threads_list = {1, 2, 4, 8};
        bconf.repetitions = 15;
        bconf.variant = conf.variant;
        bconf.population_size = conf.population_size;
        bconf.generations = conf.generations;
        bconf.mutation_rate = conf.mutation_rate;
        bconf.num_islands = conf.num_islands;
        bconf.population_per_island = conf.population_per_island;
        bconf.migration_frequency = conf.migration_frequency;
        bconf.num_migrants = conf.num_migrants;
        bconf.migration_topology = conf.migration_topology;
        bconf.verbose = conf.verbose;
        bconf.base_seed = conf.seed;

        if (conf.variant == "standard") {
            switch (bench_config_id) {
            case 1:
                bconf.population_size = 100;
                bconf.generations = 50;
                bconf.mutation_rate = 0.01f;
                break;
            case 2:
                bconf.population_size = 100;
                bconf.generations = 50;
                bconf.mutation_rate = 0.05f;
                break;
            case 3:
                bconf.population_size = 200;
                bconf.generations = 100;
                bconf.mutation_rate = 0.02f;
                break;
            case 4:
                bconf.population_size = 50;
                bconf.generations = 150;
                bconf.mutation_rate = 0.03f;
                break;
            default:
                break;
            }
        } else {
            switch (bench_config_id) {
            case 1:
                bconf.num_islands = 4;
                bconf.population_per_island = 25;
                bconf.generations = 50;
                bconf.mutation_rate = 0.01f;
                bconf.migration_frequency = 10;
                bconf.num_migrants = 2;
                bconf.migration_topology = "ring";
                break;
            case 2:
                bconf.num_islands = 4;
                bconf.population_per_island = 25;
                bconf.generations = 50;
                bconf.mutation_rate = 0.05f;
                bconf.migration_frequency = 10;
                bconf.num_migrants = 2;
                bconf.migration_topology = "random";
                break;
            case 3:
                bconf.num_islands = 8;
                bconf.population_per_island = 25;
                bconf.generations = 100;
                bconf.mutation_rate = 0.02f;
                bconf.migration_frequency = 20;
                bconf.num_migrants = 4;
                bconf.migration_topology = "ring";
                break;
            case 4:
                bconf.num_islands = 4;
                bconf.population_per_island = 25;
                bconf.generations = 150;
                bconf.mutation_rate = 0.03f;
                bconf.migration_frequency = 5;
                bconf.num_migrants = 1;
                bconf.migration_topology = "random";
                break;
            default:
                break;
            }
        }

        if (!conf.report_file.empty()) {
            bconf.report_file = conf.report_file;
        }

        Benchmark::Run(bconf);
        return 0;
    }

    if (conf.instance.empty()) {
        std::cerr << "Error: --instance es requerido.\n";
        std::cerr << "Usa --help para ver las opciones disponibles.\n";
        return 1;
    }

    if (conf.population_size <= 0) {
        std::cerr << "Error: --population debe ser mayor a 0.\n";
        return 1;
    }
    if (conf.generations <= 0) {
        std::cerr << "Error: --generations debe ser mayor a 0.\n";
        return 1;
    }
    if (conf.mutation_rate < 0.0f || conf.mutation_rate > 1.0f) {
        std::cerr << "Error: --mutation-rate debe estar entre 0 y 1.\n";
        return 1;
    }

    std::string data_dir = conf.instance;
    Instance instance = InstanceLoader::load(data_dir);

    std::cout << "Instancia: " << conf.instance << "\n";
    std::cout << "Variante: " << conf.variant << "\n";
    std::cout << "Hilos: " << conf.threads << "\n";
    std::cout << "Semilla: " << conf.seed << "\n";
    std::cout << "Items cargados: " << instance.items.size() << "\n";
    std::cout << "Reglas de categoria: " << instance.category_rules.size()
              << "\n";
    std::cout << "Incompatibilidades: " << instance.incompatibilities.size()
              << "\n";
    std::cout << "Dependencias: " << instance.dependencies.size() << "\n";
    std::cout << "Capacidad mochila: peso=" << instance.knapsack.max_weight
              << ", volumen=" << instance.knapsack.max_volume << "\n";
    std::cout << "Población: " << conf.population_size << "\n";
    std::cout << "Generaciones: " << conf.generations << "\n";
    std::cout << "Tasa de mutación: " << conf.mutation_rate << "\n";
    std::cout << "Umbral convergencia: " << conf.convergence_threshold << "\n";

    std::chrono::duration<double> elapsed;
    std::vector<GenerationStats> stats;
    Individual mejor;

    if (conf.variant == "islands") {
        auto start = std::chrono::high_resolution_clock::now();
        IslandModel ga(instance, conf.num_islands, conf.population_per_island,
                       conf.generations, conf.mutation_rate,
                       conf.migration_frequency, conf.num_migrants,
                       conf.migration_topology, conf.seed);
        ga.SetConvergenceThreshold(conf.convergence_threshold);

        if (conf.threads > 1) {
            ga.RunParallel(conf.threads);
        } else {
            ga.Run();
        }
        auto end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;

        stats = ga.GetStats();
        if (conf.verbose) ga.View_Population();
        mejor = ga.GetBestSolution();
    } else {
        auto start = std::chrono::high_resolution_clock::now();
        GeneticAlgorithm ga(instance, conf.population_size, conf.generations,
                            conf.mutation_rate, conf.seed);
        ga.SetConvergenceThreshold(conf.convergence_threshold);

        if (conf.threads > 1) {
            ga.RunParallel(conf.threads);
        } else {
            ga.Run();
        }
        auto end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;

        stats = ga.GetStats();
        if (conf.verbose) ga.View_Population();
        mejor = ga.GetBestSolution();
    }

    std::cout << "\n=== Reporte por Generación ===\n";
    std::cout << "Gen\tBest\tAvg\tWorst\tValid\tConverg\n";
    for (const auto &s : stats) {
        std::cout << s.generation << "\t" << s.best_fitness << "\t"
                  << s.avg_fitness << "\t" << s.worst_fitness << "\t"
                  << s.valid_count << "/" << conf.population_size << "\t"
                  << s.convergence_delta << "\n";
    }

    if (!conf.report_file.empty()) {
        std::ofstream file(conf.report_file);
        file << "generation,best_fitness,avg_fitness,worst_fitness,valid_count,"
                "convergence_delta\n";
        for (const auto &s : stats) {
            file << s.generation << "," << s.best_fitness << ","
                 << s.avg_fitness << "," << s.worst_fitness << ","
                 << s.valid_count << "," << s.convergence_delta << "\n";
        }
        std::cout << "Reporte guardado en: " << conf.report_file << "\n";
    }

    std::cout << "\n=== Resultados ===\n";
    std::cout << "Mejor Fitness: " << mejor.fitness << "\n";
    std::cout << "Solución válida: " << (mejor.is_valid ? "Sí" : "No") << "\n";

    int selected = 0;
    float total_weight = 0;
    float total_volume = 0;
    float total_value = 0;
    for (size_t i = 0; i < mejor.chromosome.size(); ++i) {
        if (mejor.chromosome[i]) {
            selected++;
            total_weight += instance.items[i].weight;
            total_volume += instance.items[i].volume;
            total_value += instance.items[i].value;
        }
    }
    std::cout << "Ítems seleccionados: " << selected << "\n";
    std::cout << "Peso total: " << total_weight << " / "
              << instance.knapsack.max_weight << "\n";
    std::cout << "Volumen total: " << total_volume << " / "
              << instance.knapsack.max_volume << "\n";
    std::cout << "Valor total: " << total_value << "\n";

    Fitness::PrintConstraintDetails(mejor, instance);

    std::cout << "\nTiempo de ejecución: " << elapsed.count() << "s\n";

    return 0;
}
