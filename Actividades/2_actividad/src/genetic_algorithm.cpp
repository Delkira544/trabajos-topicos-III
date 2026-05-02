#include "genetic_algorithm.hpp"
#include "fitness.hpp"
#include <algorithm>
#include <iostream>
#include <random>

#include "crossover.hpp"
#include "mutation.hpp"
#include "selection.hpp"

GeneticAlgorithm::GeneticAlgorithm(const Instance &instance,
                                   int population_size, int generations,
                                   float mutation_rate, int seed)
    : instance(instance), population_size(population_size),
      generations(generations), mutation_rate(mutation_rate) {
    rng.seed(seed);
}

void GeneticAlgorithm::Initialize_Population() {
    population.clear();
    population.reserve(population_size);

    std::bernoulli_distribution d(
        0.5); // Probabilidad de 0.5 para incluir o no cada ítem

    std::vector<int> indices(instance.items.size());
    std::iota(indices.begin(), indices.end(), 0);

    for (int i = 0; i < population_size; ++i) {
        Individual individual;
        individual.chromosome.assign(
            instance.items.size(),
            false); // Inicializa el cromosoma con todos los ítems no incluidos

        float total_weight = 0.0;
        float total_volume = 0.0;

        std::shuffle(
            indices.begin(), indices.end(),
            rng); // Mezcla los índices para generar soluciones variadas

        for (int idx : indices) {
            const Item &item = instance.items[idx];

            if (total_weight + item.weight <= instance.knapsack.max_weight &&
                total_volume + item.volume <= instance.knapsack.max_volume) {
                individual.chromosome[idx] =
                    d(rng); // Decide aleatoriamente si incluir el ítem
                if (individual.chromosome[idx]) {
                    total_weight += item.weight;
                    total_volume += item.volume;
                }
            } else {
                break; // Si se excede el peso o volumen, no se incluyen más
                       // ítems
            }
        }
        population.push_back(individual);
    }
}

void GeneticAlgorithm::View_Population() {
    for (const auto &individual : population) {
        std::cout << "Individual: ";
        for (bool gene : individual.chromosome) {
            std::cout << gene;
        }
        std::cout << "\n";
        std::cout << "\t| Fitness: " << individual.fitness
                  << " | Valid: " << (individual.is_valid ? "Yes" : "No");
        std::cout << "\n";
    }
}

void GeneticAlgorithm::Run() {
    // Bucle generacional
    Initialize_Population();
    for (int gen = 0; gen < generations; ++gen) {

        std::vector<Individual> new_population;
        new_population.reserve(population_size);

        while (new_population.size() < population_size) {

            int tournament_size = 3;
            Individual p1 =
                Selection::Tournament(population, tournament_size, rng);
            Individual p2 =
                Selection::Tournament(population, tournament_size, rng);

            Individual c1, c2;

            Crossover::SinglePoint(p1, p2, c1, c2, rng);

            Mutation::BitFlip(c1, mutation_rate, rng);
            Mutation::BitFlip(c2, mutation_rate, rng);

            Fitness::Evaluate(c1, instance);
            Fitness::Evaluate(c2, instance);

            new_population.push_back(c1);
            if (new_population.size() < population_size) {
                new_population.push_back(c2);
            }
        }

        population = new_population;
    }
}

Individual GeneticAlgorithm::GetBestSolution() const {
    if (population.empty()) {
        throw std::runtime_error(
            "La población está vacía. Ejecuta Run() primero.");
    }

    int best_idx = 0;
    float best_fitness = population[0].fitness;

    for (size_t i = 1; i < population.size(); ++i) {
        if (population[i].fitness > best_fitness) {
            best_fitness = population[i].fitness;
            best_idx = i;
        }
    }
    return population[best_idx];
}
