#include "genetic_algorithm.hpp"
#include <iostream>
#include <algorithm>
#include <random>

GeneticAlgorithm::GeneticAlgorithm(const Instance& instance, int population_size, int generations, float mutation_rate)
    : instance(instance), population_size(population_size), generations(generations), mutation_rate(mutation_rate) {
    Initialize_Population();
}

void GeneticAlgorithm::Initialize_Population() {
    population.clear();
    population.reserve(population_size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution d(0.5); // Probabilidad de 0.5 para incluir o no cada ítem

    std::vector<int> indices(instance.items.size());
    std::iota(indices.begin(), indices.end(), 0);

    for (int i = 0; i < population_size; ++i) {
        Individual individual;
        individual.chromosome.assign(instance.items.size(), false); // Inicializa el cromosoma con todos los ítems no incluidos

        float total_weight = 0.0;
        float total_volume = 0.0;

        std::shuffle(indices.begin(), indices.end(), gen);

        for (int idx : indices) {
            const Item& item = instance.items[idx];

            if (total_weight + item.weight <= instance.knapsack.max_weight &&
                total_volume + item.volume <= instance.knapsack.max_volume) {
                individual.chromosome[idx] = d(gen); // Decide aleatoriamente si incluir el ítem
                if (individual.chromosome[idx]) {
                    total_weight += item.weight;
                    total_volume += item.volume;
                }
            } else {
                break; // Si se excede el peso o volumen, no se incluyen más ítems
            }
        }
        population.push_back(individual);
    }
}


void GeneticAlgorithm::View_Population() {
    for (const auto& individual : population) {
        std::cout << "Individual: ";
        for (bool gene : individual.chromosome) {
            std::cout << gene << " ";
        }
        std::cout << "\n";
    }
}
