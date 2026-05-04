#include "genetic_algorithm.hpp"
#include "fitness.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
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

    std::bernoulli_distribution d(0.5);

    std::vector<int> indices(instance.items.size());
    std::iota(indices.begin(), indices.end(), 0);

    for (int i = 0; i < population_size; ++i) {
        Individual individual;
        individual.chromosome.assign(instance.items.size(), false);

        float total_weight = 0.0;
        float total_volume = 0.0;

        std::shuffle(indices.begin(), indices.end(), rng);

        for (int idx : indices) {
            const Item &item = instance.items[idx];

            if (total_weight + item.weight <= instance.knapsack.max_weight &&
                total_volume + item.volume <= instance.knapsack.max_volume) {
                individual.chromosome[idx] = d(rng);
                if (individual.chromosome[idx]) {
                    total_weight += item.weight;
                    total_volume += item.volume;
                }
            } else {
                break;
            }
        }
        population.push_back(individual);
    }
}

void GeneticAlgorithm::View_Population() {
    std::cout << "\n=== Población (" << population.size()
              << " individuos) ===\n";

    float best = population[0].fitness;
    float worst = population[0].fitness;
    float sum = 0;
    int valid_count = 0;

    for (const auto &ind : population) {
        best = std::max(best, ind.fitness);
        worst = std::min(worst, ind.fitness);
        sum += ind.fitness;
        if (ind.is_valid) valid_count++;
    }

    std::cout << "Mejor: " << best << "\n";
    std::cout << "Peor: " << worst << "\n";
    std::cout << "Promedio: " << (sum / population.size()) << "\n";
    std::cout << "Válidos: " << valid_count << "/" << population.size() << "\n";

    Individual mejor = GetBestSolution();
    std::cout << "\nMejor individuo:\n";
    std::cout << "  Fitness: " << mejor.fitness << "\n";
    std::cout << "  Válido: " << (mejor.is_valid ? "Sí" : "No") << "\n";
    int selected =
        std::count(mejor.chromosome.begin(), mejor.chromosome.end(), true);
    std::cout << "  Ítems seleccionados: " << selected << "\n";
}

Individual
GeneticAlgorithm::FindBest(const std::vector<Individual> &pop) const {
    Individual best = pop[0];
    for (size_t i = 1; i < pop.size(); ++i) {
        if (pop[i].fitness > best.fitness) {
            best = pop[i];
        }
    }
    return best;
}

void GeneticAlgorithm::Run() {
    Initialize_Population();

    // Evaluar población inicial
    for (auto &ind : population) {
        Fitness::Evaluate(ind, instance);
    }

    Individual best_ever = FindBest(population);

    for (int gen = 0; gen < generations; ++gen) {
        std::vector<Individual> new_population;
        new_population.reserve(population_size);

        // Elitismo: preservar el mejor individuo
        new_population.push_back(best_ever);

        while (static_cast<int>(new_population.size()) < population_size) {
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
            if (static_cast<int>(new_population.size()) < population_size) {
                new_population.push_back(c2);
            }
        }

        population = std::move(new_population);

        // Actualizar mejor histórico
        Individual current_best = FindBest(population);
        if (current_best.fitness > best_ever.fitness) {
            best_ever = current_best;
        }
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
