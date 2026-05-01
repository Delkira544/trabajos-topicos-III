#ifndef GENETIC_ALGORITHM_HPP
#define GENETIC_ALGORITHM_HPP

#include <vector>
#include <string>
#include <unordered_map>

struct Item {
    int id;
    float value;
    float weight;
    float volume;
    std::string category;
};

struct CategoryRule {
    int min;
    int max;
};

// Mapa: "Electrónica" -> {min: 1, max: 3}
using CategoryMap = std::unordered_map<std::string, CategoryRule>;


// Incompatibilidades: Si tengo A, no puedo tener B
// Se puede usar un vector de pares o una matriz de adyacencia
struct Incompatibility {
    int id_a;
    int id_b;
};

// Dependencias: Si tengo A, necesito obligatoriamente B
// Mapa: id_item -> id_requerido
using DependencyMap = std::unordered_map<int, int>;

struct KnapsackConfig {
    float max_weight;
    float max_volume;
};

struct Instance {
    std::vector<Item> items;
    CategoryMap category_rules;
    std::vector<Incompatibility> incompatibilities;
    DependencyMap dependencies;
    KnapsackConfig knapsack;
};
#endif
