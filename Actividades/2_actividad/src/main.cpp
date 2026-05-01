#include <iostream>
#include <string>
#include "instance_loader.hpp"

struct Config {
    std::string instance;
    std::string variant;
    int threads = 1;
    int seed = 0;
};

int main(int argc, char* argv[]) {
    Config conf;

    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];

        if (flag == "--instance") conf.instance = argv[++i];
        else if (flag == "--variant") conf.variant = argv[++i];
        else if (flag == "--threads") conf.threads = std::stoi(argv[++i]);
        else if (flag == "--seed") conf.seed = std::stoi(argv[++i]);
    }

    std::string data_dir = conf.instance;
    Instance instance = InstanceLoader::load(data_dir);

    std::cout << "Instancia: " << conf.instance << "\n";
    std::cout << "Variante: " << conf.variant << "\n";
    std::cout << "Hilos: " << conf.threads << "\n";
    std::cout << "Semilla: " << conf.seed << "\n";
    std::cout << "Items cargados: " << instance.items.size() << "\n";
    std::cout << "Reglas de categoria: " << instance.category_rules.size() << "\n";
    std::cout << "Incompatibilidades: " << instance.incompatibilities.size() << "\n";
    std::cout << "Dependencias: " << instance.dependencies.size() << "\n";

    for (const auto& item : instance.items) {
        std::cout << "  Item[" << item.id << "] val=" << item.value
                  << " peso=" << item.weight << " vol=" << item.volume
                  << " cat=" << item.category << "\n";
    }

    return 0;
}
