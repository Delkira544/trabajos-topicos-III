#include <chrono>
#include <iostream>
#include <string>
#include "genetic_algorithm.hpp"
#include "instance_loader.hpp"

struct Config {
    std::string instance;
    std::string variant = "standard";
    int threads = 1;
    int seed = 0;
    int population_size = 100;
    int generations = 50;
    float mutation_rate = 0.01f;
    bool verbose = false;
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
        << "  --verbose              Mostrar información detallada\n"
        << "  --help, -h             Mostrar esta ayuda\n";
}

int main(int argc, char *argv[]) {
    Config conf;

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
        else if (flag == "--verbose")
            conf.verbose = true;
        else if (flag == "--help" || flag == "-h") {
            print_usage(argv[0]);
            return 0;
        }
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

    if (conf.variant == "islands") {
        std::cerr << "Advertencia: variante 'islands' aún no implementada. "
                  << "Usando algoritmo estándar.\n";
    }

    auto start = std::chrono::high_resolution_clock::now();

    GeneticAlgorithm ga(instance, conf.population_size, conf.generations,
                        conf.mutation_rate, conf.seed);

    if (conf.threads > 1) {
        ga.RunParallel(conf.threads);
    } else {
        ga.Run();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if (conf.verbose) {
        ga.View_Population();
    }

    Individual mejor = ga.GetBestSolution();
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
    std::cout << "\nTiempo de ejecución: " << elapsed.count() << "s\n";

    return 0;
}
