#include "genetic_algorithm.hpp"
#include "fitness.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

    // Evaluación paralela de la población inicial.
    // shared: population (lectura/escritura por índices disjuntos), instance (RO).
    // private: i (índice de iteración).
#pragma omp parallel for schedule(static) \
    default(none) shared(population, instance) firstprivate(generations)
    for (int i = 0; i < population_size; ++i) {
        Fitness::Evaluate(population[i], instance, 0, generations);
    }

    best_ever_ = FindBest(population);
    gens_without_improvement_ = 0;
    gens_no_improve_total_ = 0;
    first_feasible_gen_ = best_ever_.is_valid ? 0 : -1;

    // Mejora #7 — preservar el top elite_fraction_ (10%) en vez de un único
    // elite.
    int n_elite =
        std::max(1, static_cast<int>(population_size * elite_fraction_));

    for (int gen = 0; gen < generations; ++gen) {
        // 1) Elitismo múltiple: ordenar y copiar top-N a la nueva generación.
        std::vector<Individual> sorted_pop = population;
        std::sort(sorted_pop.begin(), sorted_pop.end(),
                  [](const Individual &a, const Individual &b) {
                      return IsBetter(a, b);
                  });

        std::vector<Individual> new_population;
        new_population.reserve(population_size);
        for (int i = 0; i < n_elite; ++i) {
            new_population.push_back(sorted_pop[i]);
        }

        int children_needed =
            population_size - static_cast<int>(new_population.size());

        // RNG por par para reproducibilidad independiente del nº de hilos.
        int num_pairs = (children_needed + 1) / 2;
        std::vector<Individual> children(children_needed);

        // Cada iteración escribe en children[i] y children[i+1] (índices
        // disjuntos por par), lee population/instance, y crea su propio RNG
        // local → sin race conditions.
        // shared: population, instance, children (escritura por índice).
        // firstprivate: contadores y constantes de la iteración.
#pragma omp parallel for schedule(static) \
    default(none) \
    shared(population, instance, children) \
    firstprivate(seed, gen, num_pairs, children_needed, mutation_rate)
        for (int pair = 0; pair < num_pairs; ++pair) {
            std::mt19937 pair_rng(static_cast<uint32_t>(seed) +
                                  static_cast<uint32_t>(gen) * 10000u +
                                  static_cast<uint32_t>(pair));
            // Mejora #8 — tournament k=5 (más presión selectiva).
            int tournament_size = 5;
            int i = pair * 2;

            Individual p1 =
                Selection::Tournament(population, tournament_size, pair_rng);
            Individual p2 =
                Selection::Tournament(population, tournament_size, pair_rng);

            Individual c1, c2;
            // Mejora #4 — Uniform crossover con sesgo a no incluir.
            Crossover::Uniform(p1, p2, c1, c2, pair_rng, 0.45f);

            // Mejora #5 — mutación asimétrica consciente de la capacidad.
            Mutation::BitFlipAsymmetric(c1, mutation_rate, instance, pair_rng);
            Mutation::BitFlipAsymmetric(c2, mutation_rate, instance, pair_rng);

            // Mutación DIRIGIDA a violaciones soft: 50% prob por hijo,
            // y si se activa, fixea UNA violación al azar (no greedy).
            Mutation::TargetedFix(c1, instance, pair_rng, 0.5f);
            Mutation::TargetedFix(c2, instance, pair_rng, 0.5f);

            children[i] = std::move(c1);
            if (i + 1 < children_needed) {
                children[i + 1] = std::move(c2);
            }
        }

        for (auto &child : children) {
            new_population.push_back(std::move(child));
        }
        if (static_cast<int>(new_population.size()) > population_size) {
            new_population.resize(population_size);
        }

        // Solo evaluamos los hijos (los elites ya están evaluados).
        // shared: new_population (escritura por índice disjunto), instance (RO).
#pragma omp parallel for schedule(static) \
    default(none) \
    shared(new_population, instance) \
    firstprivate(n_elite, population_size, gen, generations)
        for (int i = n_elite; i < population_size; ++i) {
            Fitness::Evaluate(new_population[i], instance, gen, generations);
        }

        population = std::move(new_population);

        Individual current_best = FindBest(population);
        if (IsBetter(current_best, best_ever_)) {
            best_ever_ = current_best;
            gens_without_improvement_ = 0;
            gens_no_improve_total_ = 0;
        } else {
            gens_without_improvement_++;
            gens_no_improve_total_++;
        }

        // Registrar la primera generación en que el mejor se volvió factible.
        if (first_feasible_gen_ < 0 && best_ever_.is_valid) {
            first_feasible_gen_ = gen;
            std::cout << "[FEASIBLE] gen " << gen
                      << ": primera solución factible encontrada — "
                      << gens_after_feasible_limit_
                      << " gens más para refinar antes de salir\n";
        }

        // Mejora #6 — anti-estancamiento (resetea sólo gens_without_improvement_).
        if (gens_without_improvement_ >= stagnation_limit_) {
            std::cout << "[STAGNATION] gen " << gen
                      << ": inyectando diversidad ("
                      << static_cast<int>(diversity_inject_fraction_ * 100)
                      << "% reemplazado)\n";
            InjectDiversity(rng);
            gens_without_improvement_ = 0;
        }

        RecordStats(gen);

        // Early-stop por factibilidad + ventana de refinamiento.
        // Cuando best_ever_ se vuelve válido y han pasado N gens, salimos.
        if (first_feasible_gen_ >= 0 &&
            (gen - first_feasible_gen_) >= gens_after_feasible_limit_) {
            std::cout << "[EARLY-STOP] gen " << gen
                      << ": factible desde gen " << first_feasible_gen_
                      << " (+" << gens_after_feasible_limit_
                      << " gens de refinamiento) — finalizando\n";
            break;
        }

        // Early-stop por convergencia clásica (fallback)
        if (HasConverged() && best_ever_.is_valid) {
            std::cout << "Convergencia detectada en generacion " << gen << "\n";
            break;
        }
    }
}

Individual GeneticAlgorithm::CreateRandomIndividual(std::mt19937 &r) const {
    // Mejora #10 — Bernoulli(0.25) en vez de 0.35: empezar bajo capacidad
    // para dejar margen al crossover/mutación sin entrar de inmediato en
    // territorio infactible.
    Individual individual;
    individual.chromosome.assign(instance.items.size(), false);

    std::bernoulli_distribution d(0.25);

    std::vector<int> indices(instance.items.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), r);

    float total_weight = 0.0f;
    float total_volume = 0.0f;
    for (int idx : indices) {
        const Item &item = instance.items[idx];
        if (total_weight + item.weight <= instance.knapsack.max_weight &&
            total_volume + item.volume <= instance.knapsack.max_volume) {
            if (d(r)) {
                individual.chromosome[idx] = true;
                total_weight += item.weight;
                total_volume += item.volume;
            }
        }
    }
    return individual;
}

void GeneticAlgorithm::Initialize_Population() {
    population.clear();
    population.reserve(population_size);
    for (int i = 0; i < population_size; ++i) {
        population.push_back(CreateRandomIndividual(rng));
    }
}

void GeneticAlgorithm::InjectDiversity(std::mt19937 &r) {
    // Mejora #6 — Reemplazar el peor `diversity_inject_fraction_` con
    // individuos nuevos (re-inicializados) para escapar de óptimos locales.
    std::sort(population.begin(), population.end(),
              [](const Individual &a, const Individual &b) {
                  return IsBetter(a, b);
              });
    int n_replace =
        static_cast<int>(population_size * diversity_inject_fraction_);
    int start = population_size - n_replace;
    for (int i = start; i < population_size; ++i) {
        population[i] = CreateRandomIndividual(r);
        Fitness::Evaluate(population[i], instance, 0, generations);
    }
}

int GeneticAlgorithm::CountTotalViolations(const Individual &ind) const {
    int total = 0;

    float total_weight = 0.0f, total_volume = 0.0f;
    std::unordered_map<std::string, int> category_counts;
    std::unordered_set<int> selected_ids;

    for (size_t i = 0; i < ind.chromosome.size(); ++i) {
        if (ind.chromosome[i]) {
            const Item &item = instance.items[i];
            total_weight  += item.weight;
            total_volume  += item.volume;
            category_counts[item.category]++;
            selected_ids.insert(item.id);
        }
    }

    if (total_weight > instance.knapsack.max_weight) total++;
    if (total_volume > instance.knapsack.max_volume) total++;

    for (const auto &[cat_name, rule] : instance.category_rules) {
        int count = category_counts.count(cat_name)
                        ? category_counts.at(cat_name) : 0;
        if (count < rule.min || count > rule.max) total++;
    }

    for (const auto &inc : instance.incompatibilities) {
        if (selected_ids.count(inc.id_a) && selected_ids.count(inc.id_b))
            total++;
    }

    for (const auto &[item_id, req_id] : instance.dependencies) {
        if (selected_ids.count(item_id) && !selected_ids.count(req_id))
            total++;
    }

    return total;
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

    for (auto &ind : population) {
        Fitness::Evaluate(ind, instance, 0, generations);
    }

    best_ever_ = FindBest(population);
    gens_without_improvement_ = 0;
    gens_no_improve_total_ = 0;
    first_feasible_gen_ = best_ever_.is_valid ? 0 : -1;

    int n_elite =
        std::max(1, static_cast<int>(population_size * elite_fraction_));

    for (int gen = 0; gen < generations; ++gen) {
        // Mejora #7 — multi-elite: top-N preservado.
        std::vector<Individual> sorted_pop = population;
        std::sort(sorted_pop.begin(), sorted_pop.end(),
                  [](const Individual &a, const Individual &b) {
                      return IsBetter(a, b);
                  });

        std::vector<Individual> new_population;
        new_population.reserve(population_size);
        for (int i = 0; i < n_elite; ++i) {
            new_population.push_back(sorted_pop[i]);
        }

        int tournament_size = 10; // mejora #8
        while (static_cast<int>(new_population.size()) < population_size) {
            Individual p1 =
                Selection::Tournament(population, tournament_size, rng);
            Individual p2 =
                Selection::Tournament(population, tournament_size, rng);

            Individual c1, c2;
            Crossover::Uniform(p1, p2, c1, c2, rng, 0.45f);                // #4
            Mutation::BitFlipAsymmetric(c1, mutation_rate, instance, rng); // #5
            Mutation::BitFlipAsymmetric(c2, mutation_rate, instance, rng);
            // Mutación dirigida a violaciones soft (probabilística, parcial)
            Mutation::TargetedFix(c1, instance, rng, 0.5f);
            Mutation::TargetedFix(c2, instance, rng, 0.5f);

            Fitness::Evaluate(c1, instance, gen, generations);
            Fitness::Evaluate(c2, instance, gen, generations);

            new_population.push_back(c1);
            if (static_cast<int>(new_population.size()) < population_size) {
                new_population.push_back(c2);
            }
        }

        population = std::move(new_population);

        Individual current_best = FindBest(population);
        if (IsBetter(current_best, best_ever_)) {
            best_ever_ = current_best;
            gens_without_improvement_ = 0;
            gens_no_improve_total_ = 0;
        } else {
            gens_without_improvement_++;
            gens_no_improve_total_++;
        }

        // Registrar la primera generación factible.
        if (first_feasible_gen_ < 0 && best_ever_.is_valid) {
            first_feasible_gen_ = gen;
            std::cout << "[FEASIBLE] gen " << gen
                      << ": primera solución factible encontrada — "
                      << gens_after_feasible_limit_
                      << " gens más para refinar antes de salir\n";
        }

        // Mejora #6 — anti-estancamiento (resetea sólo gens_without_improvement_).
        if (gens_without_improvement_ >= stagnation_limit_) {
            std::cout << "[STAGNATION] gen " << gen
                      << ": inyectando diversidad\n";
            InjectDiversity(rng);
            gens_without_improvement_ = 0;
        }

        RecordStats(gen);

        // Early-stop por factibilidad + ventana de refinamiento.
        if (first_feasible_gen_ >= 0 &&
            (gen - first_feasible_gen_) >= gens_after_feasible_limit_) {
            std::cout << "[EARLY-STOP] gen " << gen
                      << ": factible desde gen " << first_feasible_gen_
                      << " (+" << gens_after_feasible_limit_
                      << " gens de refinamiento) — finalizando\n";
            break;
        }

        // Convergencia clásica (fallback)
        if (HasConverged() && best_ever_.is_valid) {
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
