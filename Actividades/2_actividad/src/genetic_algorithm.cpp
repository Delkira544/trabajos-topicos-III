#include "genetic_algorithm.hpp"
#include "fitness.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <omp.h>

#include "crossover.hpp"
#include "mutation.hpp"
#include "selection.hpp"

GeneticAlgorithm::GeneticAlgorithm(const Instance &instance,
                                   int population_size, int generations,
                                   float mutation_rate, int seed)
    : instance(instance), population_size(population_size),
      generations(generations), mutation_rate(mutation_rate), seed(seed),
      previous_best_fitness_(0.0f), convergence_threshold_(0.001f) {
    rng.seed(seed);
}

void GeneticAlgorithm::SetConvergenceThreshold(float threshold) {
    convergence_threshold_ = threshold;
}

const std::vector<GenerationStats> &GeneticAlgorithm::GetStats() const {
    return stats_;
}

void GeneticAlgorithm::RecordStats(int gen) {
    GenerationStats gen_stats;
    gen_stats.generation = gen;

    Individual current_best = FindBest(population);
    gen_stats.best_fitness = current_best.fitness;
    gen_stats.best_is_valid = current_best.is_valid;

    float sum = 0;
    float worst = population[0].fitness;
    int valid = 0;

    for (const auto &ind : population) {
        sum += ind.fitness;
        worst = std::min(worst, ind.fitness);
        if (ind.is_valid) valid++;
    }

    gen_stats.avg_fitness = sum / population.size();
    gen_stats.worst_fitness = worst;
    gen_stats.valid_count = valid;

    if (gen > 0) {
        gen_stats.convergence_delta =
            std::abs(gen_stats.best_fitness - previous_best_fitness_);
    } else {
        gen_stats.convergence_delta = gen_stats.best_fitness;
    }
    previous_best_fitness_ = gen_stats.best_fitness;

    stats_.push_back(gen_stats);
}

bool GeneticAlgorithm::HasConverged() const {
    if (stats_.size() < 10) return false;

    size_t window = 10;
    float delta_sum = 0;
    for (size_t i = stats_.size() - window; i < stats_.size(); ++i) {
        delta_sum += stats_[i].convergence_delta;
    }
    float avg_delta = delta_sum / window;
    return avg_delta < convergence_threshold_;
}

std::mt19937 GeneticAlgorithm::get_rng_for_thread(int thread_id) const {
    std::mt19937 thread_rng;
    thread_rng.seed(seed + thread_id * 1000);
    return thread_rng;
}

void GeneticAlgorithm::RunParallel(int num_threads) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    Initialize_Population();

#pragma omp parallel for schedule(static)
    for (int i = 0; i < population_size; ++i) {
        Fitness::Evaluate(population[i], instance, 0, generations);
    }

    Individual best_ever = FindBest(population);

    for (int gen = 0; gen < generations; ++gen) {
        std::vector<Individual> new_population;
        new_population.reserve(population_size);

        new_population.push_back(best_ever);
        Fitness::Evaluate(new_population.back(), instance, gen, generations);

        int children_needed =
            population_size - static_cast<int>(new_population.size());
        std::vector<Individual> children;
        children.reserve(children_needed);

        int actual_threads = num_threads;
        if (actual_threads <= 0) {
            actual_threads = 1;
        }

        std::vector<std::vector<Individual>> all_children(actual_threads);

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::mt19937 thread_rng = get_rng_for_thread(tid);
            int tournament_size = 3;

#pragma omp for schedule(static)
            for (int i = 0; i < children_needed; i += 2) {
                Individual p1 = Selection::Tournament(
                    population, tournament_size, thread_rng);
                Individual p2 = Selection::Tournament(
                    population, tournament_size, thread_rng);

                Individual c1, c2;
                Crossover::SinglePoint(p1, p2, c1, c2, thread_rng);

                Mutation::BitFlip(c1, mutation_rate, thread_rng);
                Mutation::BitFlip(c2, mutation_rate, thread_rng);

                Fitness::Repair(c1, instance, thread_rng);
                Fitness::Repair(c2, instance, thread_rng);

                all_children[tid].push_back(c1);
                if (i + 1 < children_needed) {
                    all_children[tid].push_back(c2);
                }
            }
        }

        for (int t = 0; t < actual_threads; ++t) {
            children.insert(children.end(), all_children[t].begin(),
                            all_children[t].end());
        }

        for (int i = 0; i < children_needed; ++i) {
            new_population.push_back(children[i]);
        }

#pragma omp parallel for schedule(static)
        for (int i = 1; i < population_size; ++i) {
            Fitness::Evaluate(new_population[i], instance, gen, generations);
        }

        population = std::move(new_population);

        Individual current_best = FindBest(population);
        if (IsBetter(current_best, best_ever)) {
            best_ever = current_best;
        }

        RecordStats(gen);

        if (HasConverged() && best_ever.is_valid) {
            std::cout << "Convergencia detectada en generacion " << gen << "\n";
            break;
        }
    }
}

void GeneticAlgorithm::Initialize_Population() {
    population.clear();
    population.reserve(population_size);

    std::bernoulli_distribution d(0.35);

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
                continue;
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
        if (IsBetter(pop[i], best)) {
            best = pop[i];
        }
    }
    return best;
}

void GeneticAlgorithm::Run() {
    Initialize_Population();

    // Evaluar población inicial
    for (auto &ind : population) {
        Fitness::Evaluate(ind, instance, 0, generations);
    }

    Individual best_ever = FindBest(population);

    for (int gen = 0; gen < generations; ++gen) {
        std::vector<Individual> new_population;
        new_population.reserve(population_size);

        new_population.push_back(best_ever);
        Fitness::Evaluate(new_population.back(), instance, gen, generations);

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

            Fitness::Repair(c1, instance, rng);
            Fitness::Repair(c2, instance, rng);

            Fitness::Evaluate(c1, instance, gen, generations);
            Fitness::Evaluate(c2, instance, gen, generations);

            new_population.push_back(c1);
            if (static_cast<int>(new_population.size()) < population_size) {
                new_population.push_back(c2);
            }
        }

        population = std::move(new_population);

        Individual current_best = FindBest(population);
        if (IsBetter(current_best, best_ever)) {
            best_ever = current_best;
        }

        RecordStats(gen);

        if (HasConverged() && best_ever.is_valid) {
            std::cout << "Convergencia detectada en generacion " << gen << "\n";
            break;
        }
    }
}

Individual GeneticAlgorithm::GetBestSolution() const {
    if (population.empty()) {
        throw std::runtime_error(
            "La población está vacía. Ejecuta Run() primero.");
    }
    return FindBest(population);
}
