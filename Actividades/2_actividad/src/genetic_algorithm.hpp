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

struct Individual {
    std::vector<bool> chromosome; // Representación binaria de la solución
    float fitness; // Valor de la función objetivo
    bool is_valid; // Indica si la solución cumple con las restricciones

    Individual(): fitness(0.0), is_valid(true) {}
};

// Clase para el algoritmo genético
class GeneticAlgorithm {
private:
    Instance instance;
    int population_size;
    int generations;
    float mutation_rate;

    std::vector<Individual> population;

public:
    GeneticAlgorithm(const Instance& instance, int population_size, int generations, float mutation_rate);
    void Initialize_Population();
    void View_Population();

};


#endif
