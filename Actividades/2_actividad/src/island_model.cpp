#include "island_model.hpp"
#include "fitness.hpp"
#include "selection.hpp"
#include "crossover.hpp"
#include "mutation.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <omp.h>

/**
 * @brief Constructor del modelo de islas.
 *
 * Inicializa parámetros de configuración (nº de islas, población por isla,
 * generaciones, mutación, frecuencia de migración, topología, semilla) e
 * invoca InitializeIslands para crear y evaluar la población inicial.
 */
IslandModel::IslandModel(const Instance &inst, int islands_count,
                         int pop_per_island, int gens, float mut_rate,
                         int mig_freq, int n_migrants, const std::string &top,
                         int s)
    : instance(inst), num_islands(islands_count),
      population_per_island(pop_per_island), generations(gens),
      mutation_rate(mut_rate), migration_frequency(mig_freq),
      num_migrants(n_migrants), topology(top), seed(s),
      convergence_threshold_(0.001f) {
    InitializeIslands();
}

/**
 * @brief Crea un individuo aleatorio respetando capacidad (Bernoulli 0.25 +
 * corte greedy). Mismo patrón que GeneticAlgorithm::CreateRandomIndividual
 * para que islas y estándar partan de distribuciones equivalentes.
 */
Individual IslandModel::CreateRandomIndividual(std::mt19937 &r) const {
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

/**
 * @brief Inicializa las islas con población respetando capacidad y RNG
 * independiente por isla (semilla = seed + i).
 */
void IslandModel::InitializeIslands() {
    islands.assign(num_islands, std::vector<Individual>{});
    island_rngs.assign(num_islands, std::mt19937{});

    for (int i = 0; i < num_islands; ++i) {
        island_rngs[i].seed(seed + i);
        islands[i].reserve(population_per_island);
        for (int p = 0; p < population_per_island; ++p) {
            Individual ind = CreateRandomIndividual(island_rngs[i]);
            Fitness::Evaluate(ind, instance, 0, generations);
            islands[i].push_back(std::move(ind));
        }
    }
}

Individual IslandModel::FindBestInIsland(int island_idx) const {
    Individual best = islands[island_idx][0];
    for (const auto &ind : islands[island_idx]) {
        if (IsBetter(ind, best)) best = ind;
    }
    return best;
}

Individual IslandModel::FindBestOverall() const {
    Individual best = islands[0][0];
    for (int i = 0; i < num_islands; ++i) {
        for (const auto &ind : islands[i]) {
            if (IsBetter(ind, best)) best = ind;
        }
    }
    return best;
}

/**
 * @brief Cuenta violaciones totales de un individuo (peso, volumen, cada
 * categoría fuera de rango, cada par incompat activo, cada dependencia rota).
 * Idéntica semántica a GeneticAlgorithm::CountTotalViolations.
 */
int IslandModel::CountTotalViolations(const Individual &ind) const {
    int total = 0;
    float total_weight = 0.0f, total_volume = 0.0f;
    std::unordered_map<std::string, int> category_counts;
    std::unordered_set<int> selected_ids;

    for (size_t i = 0; i < ind.chromosome.size(); ++i) {
        if (ind.chromosome[i]) {
            const Item &item = instance.items[i];
            total_weight += item.weight;
            total_volume += item.volume;
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

void IslandModel::RecordStats(int gen) {
    GenerationStats stat;
    stat.generation = gen;
    stat.valid_count = 0;

    int total_pop = num_islands * population_per_island;
    float total_fitness = 0.0f;

    Individual best_overall;
    bool first = true;
    float worst = 0.0f;

    for (int i = 0; i < num_islands; ++i) {
        for (const auto &ind : islands[i]) {
            if (first) {
                best_overall = ind;
                worst = ind.fitness;
                first = false;
            } else {
                if (IsBetter(ind, best_overall)) best_overall = ind;
                if (ind.fitness < worst) worst = ind.fitness;
            }
            total_fitness += ind.fitness;
            if (ind.is_valid) stat.valid_count++;
        }
    }

    stat.best_fitness = best_overall.fitness;
    stat.worst_fitness = worst;
    stat.avg_fitness = total_fitness / total_pop;
    stat.best_is_valid = best_overall.is_valid;

    if (stats_.empty()) {
        stat.convergence_delta = 0.0f;
    } else {
        stat.convergence_delta =
            std::abs(stat.best_fitness - stats_.back().best_fitness);
    }

    stats_.push_back(stat);
}

bool IslandModel::HasConverged() const {
    if (stats_.size() < 10) return false;

    size_t window = 10;
    float delta_sum = 0;
    for (size_t i = stats_.size() - window; i < stats_.size(); ++i) {
        delta_sum += stats_[i].convergence_delta;
    }
    float avg_delta = delta_sum / window;
    return avg_delta < convergence_threshold_;
}

/**
 * @brief Intercambio periódico de migrantes entre islas.
 *
 * Cada isla aporta sus `num_migrants` mejores (orden lexicográfico vía
 * IsBetter). El destino se decide por topología:
 *   - "ring":  ((i - 1 + N) % N)
 *   - "random": elección aleatoria ≠ origen
 * Los migrantes reemplazan a los peores de la isla destino, manteniendo
 * el tamaño de población constante.
 */
void IslandModel::Migrate() {
    std::vector<std::vector<Individual>> emigrants(num_islands);
    for (int i = 0; i < num_islands; ++i) {
        std::vector<Individual> sorted_island = islands[i];
        std::sort(sorted_island.begin(), sorted_island.end(),
                  [](const Individual &a, const Individual &b) {
                      return IsBetter(a, b);
                  });
        for (int m = 0; m < num_migrants; ++m) {
            emigrants[i].push_back(sorted_island[m]);
        }
    }

    for (int i = 0; i < num_islands; ++i) {
        int source_idx = 0;
        if (topology == "ring") {
            source_idx = (i - 1 + num_islands) % num_islands;
        } else { // random
            std::uniform_int_distribution<> dis(0, num_islands - 1);
            source_idx = dis(island_rngs[i]);
            while (source_idx == i && num_islands > 1) {
                source_idx = dis(island_rngs[i]);
            }
        }

        std::vector<Individual> &target_island = islands[i];
        std::sort(target_island.begin(), target_island.end(),
                  [](const Individual &a, const Individual &b) {
                      return IsBetter(a, b);
                  });

        for (int m = 0; m < num_migrants; ++m) {
            target_island[target_island.size() - 1 - m] = emigrants[source_idx][m];
        }
    }
}

/**
 * @brief Aplica una generación de evolución a UNA isla:
 *   1) Multi-elite del 10% (preserva top-N intacto)
 *   2) Tournament k=5 sobre la isla local
 *   3) Uniform crossover con sesgo 0.45
 *   4) BitFlipAsymmetric (consciente de capacidad)
 *   5) TargetedFix probabilístico para violaciones soft
 *   6) Evaluate
 *
 * Trabaja exclusivamente sobre `islands[idx]` e `island_rngs[idx]` → puede
 * ser invocada en paralelo desde RunParallel sin race conditions.
 */
void IslandModel::EvolveIsland(int idx, int gen) {
    auto &island = islands[idx];
    auto &local_rng = island_rngs[idx];

    int n_elite = std::max(
        1, static_cast<int>(population_per_island * elite_fraction_));

    // Ordenar de mejor a peor (IsBetter)
    std::sort(island.begin(), island.end(),
              [](const Individual &a, const Individual &b) {
                  return IsBetter(a, b);
              });

    std::vector<Individual> new_island;
    new_island.reserve(population_per_island);
    for (int e = 0; e < n_elite; ++e) {
        new_island.push_back(island[e]);
    }

    while (static_cast<int>(new_island.size()) < population_per_island) {
        Individual p1 = Selection::Tournament(island, tournament_size_, local_rng);
        Individual p2 = Selection::Tournament(island, tournament_size_, local_rng);

        Individual c1, c2;
        Crossover::Uniform(p1, p2, c1, c2, local_rng, crossover_bias_);

        Mutation::BitFlipAsymmetric(c1, mutation_rate, instance, local_rng);
        Mutation::BitFlipAsymmetric(c2, mutation_rate, instance, local_rng);

        Mutation::TargetedFix(c1, instance, local_rng, targeted_fix_prob_);
        Mutation::TargetedFix(c2, instance, local_rng, targeted_fix_prob_);

        Fitness::Evaluate(c1, instance, gen, generations);
        new_island.push_back(c1);

        if (static_cast<int>(new_island.size()) < population_per_island) {
            Fitness::Evaluate(c2, instance, gen, generations);
            new_island.push_back(c2);
        }
    }

    island = std::move(new_island);
}

/**
 * @brief Inyecta diversidad reemplazando el peor `diversity_inject_fraction_`
 * de CADA isla con individuos nuevos. Se dispara cuando el `best_ever_` no
 * mejora en `stagnation_limit_` generaciones — es un evento global de tipo
 * "catástrofe" que renueva ~30% del genoma colectivo.
 */
void IslandModel::InjectDiversityAllIslands() {
    int n_replace = static_cast<int>(population_per_island
                                     * diversity_inject_fraction_);
    if (n_replace <= 0) return;

    for (int i = 0; i < num_islands; ++i) {
        auto &island = islands[i];
        std::sort(island.begin(), island.end(),
                  [](const Individual &a, const Individual &b) {
                      return IsBetter(a, b);
                  });
        int start = population_per_island - n_replace;
        for (int p = start; p < population_per_island; ++p) {
            island[p] = CreateRandomIndividual(island_rngs[i]);
            Fitness::Evaluate(island[p], instance, 0, generations);
        }
    }
}

/**
 * @brief Bucle evolutivo secuencial. Para cada generación: evoluciona cada
 * isla, migra cada `migration_frequency`, actualiza best_ever_ y aplica
 * detección de estancamiento + early-stop.
 */
void IslandModel::Run() {
    RecordStats(0);
    best_ever_ = FindBestOverall();
    gens_without_improvement_ = 0;
    gens_no_improve_total_ = 0;
    first_feasible_gen_ = best_ever_.is_valid ? 0 : -1;

    for (int gen = 1; gen <= generations; ++gen) {
        for (int i = 0; i < num_islands; ++i) {
            EvolveIsland(i, gen);
        }

        if (gen % migration_frequency == 0) {
            Migrate();
        }

        Individual current_best = FindBestOverall();
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

        if (gens_without_improvement_ >= stagnation_limit_) {
            std::cout << "[STAGNATION] gen " << gen
                      << ": inyectando diversidad en todas las islas ("
                      << static_cast<int>(diversity_inject_fraction_ * 100)
                      << "% reemplazado por isla)\n";
            InjectDiversityAllIslands();
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

/**
 * @brief Versión paralela: cada isla se evoluciona en un hilo OpenMP. Sin
 * race conditions porque cada hilo trabaja sobre `islands[i]` e
 * `island_rngs[i]` exclusivos. La migración y los chequeos globales ocurren
 * fuera del bloque paralelo.
 */
void IslandModel::RunParallel(int num_threads) {
    if (num_threads > 0) omp_set_num_threads(num_threads);

    RecordStats(0);
    best_ever_ = FindBestOverall();
    gens_without_improvement_ = 0;
    gens_no_improve_total_ = 0;
    first_feasible_gen_ = best_ever_.is_valid ? 0 : -1;

    for (int gen = 1; gen <= generations; ++gen) {
        // Cada iteración accede sólo a islands[i] e island_rngs[i] → sin race.
        // shared: islands, island_rngs, instance (todas las direcciones).
        // firstprivate: parámetros estables durante la generación.
        #pragma omp parallel for schedule(dynamic) \
            default(none) \
            shared(islands, island_rngs, instance) \
            firstprivate(num_islands, gen)
        for (int i = 0; i < num_islands; ++i) {
            EvolveIsland(i, gen);
        }

        if (gen % migration_frequency == 0) {
            Migrate();
        }

        Individual current_best = FindBestOverall();
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

        if (gens_without_improvement_ >= stagnation_limit_) {
            std::cout << "[STAGNATION] gen " << gen
                      << ": inyectando diversidad en todas las islas ("
                      << static_cast<int>(diversity_inject_fraction_ * 100)
                      << "% reemplazado por isla)\n";
            InjectDiversityAllIslands();
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

void IslandModel::View_Population() const {
    int total_pop = num_islands * population_per_island;
    std::cout << "\n=== Población de Islas (" << total_pop
              << " individuos en " << num_islands << " islas) ===\n";

    Individual best_ind = islands[0][0];
    float worst = islands[0][0].fitness;
    float sum = 0;
    int valid_count = 0;

    for (int i = 0; i < num_islands; ++i) {
        for (const auto &ind : islands[i]) {
            if (IsBetter(ind, best_ind)) best_ind = ind;
            if (ind.fitness < worst) worst = ind.fitness;
            sum += ind.fitness;
            if (ind.is_valid) valid_count++;
        }
    }

    std::cout << "Mejor: " << best_ind.fitness << "\n";
    std::cout << "Peor: " << worst << "\n";
    std::cout << "Promedio: " << (sum / total_pop) << "\n";
    std::cout << "Válidos: " << valid_count << "/" << total_pop << "\n";

    Individual mejor = GetBestSolution();
    std::cout << "\nMejor individuo global:\n";
    std::cout << "  Fitness: " << mejor.fitness << "\n";
    std::cout << "  Válido: " << (mejor.is_valid ? "Sí" : "No") << "\n";
    int selected =
        std::count(mejor.chromosome.begin(), mejor.chromosome.end(), true);
    std::cout << "  Ítems seleccionados: " << selected << "\n";
}
