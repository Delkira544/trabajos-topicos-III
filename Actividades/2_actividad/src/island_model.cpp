#include "island_model.hpp"
#include "fitness.hpp"
#include "selection.hpp"
#include "crossover.hpp"
#include "mutation.hpp"
#include <iostream>
#include <algorithm>
#include <omp.h>

/**
 * @brief Constructor de la clase IslandModel.
 * 
 * Inicializa los parámetros de configuración del modelo de islas como el número 
 * de islas, población, generaciones, mutación, frecuencia y topología de migración.
 * Inmediatamente después, invoca InitializeIslands para crear la población inicial.
 *
 * @param inst Instancia cargada del problema de la mochila.
 * @param islands_count Cantidad de subpoblaciones (islas).
 * @param pop_per_island Tamaño de la población dentro de cada isla.
 * @param gens Número total de generaciones a evolucionar.
 * @param mut_rate Tasa de mutación a aplicar sobre los cromosomas.
 * @param mig_freq Frecuencia (en generaciones) con la que ocurre la migración.
 * @param n_migrants Cantidad de mejores individuos que migrarán de cada isla.
 * @param top Tipo de topología de interconexión ("ring" o "random").
 * @param s Semilla inicial base para la reproducibilidad estocástica.
 */
IslandModel::IslandModel(const Instance& inst, int islands_count, int pop_per_island, 
                         int gens, float mut_rate, int mig_freq, int n_migrants, 
                         const std::string& top, int s)
    : instance(inst), num_islands(islands_count), population_per_island(pop_per_island),
      generations(gens), mutation_rate(mut_rate), migration_frequency(mig_freq),
      num_migrants(n_migrants), topology(top), seed(s), convergence_threshold_(0.001f) {
    InitializeIslands();
}

/**
 * @brief Genera e inicializa los individuos de cada isla de forma estocástica.
 * 
 * Genera cromosomas aleatorios a nivel de bit (0 ó 1) y evalúa su factor 
 * de calidad (fitness) base. Adicionalmente, provee una semilla independiente a 
 * cada RNG (Random Number Generator) de cada isla para que su evolución inicial 
 * pueda divergir libremente aunque vengan de la misma semilla del problema.
 */
void IslandModel::InitializeIslands() {
    islands.resize(num_islands, std::vector<Individual>(population_per_island));
    island_rngs.resize(num_islands);
    
    for (int i = 0; i < num_islands; ++i) {
        island_rngs[i].seed(seed + i);
        for (int p = 0; p < population_per_island; ++p) {
            islands[i][p].chromosome.resize(instance.items.size());
            std::uniform_int_distribution<> dis(0, 1);
            for (size_t j = 0; j < instance.items.size(); ++j) {
                islands[i][p].chromosome[j] = dis(island_rngs[i]);
            }
            Fitness::Repair(islands[i][p], instance, island_rngs[i]);
            Fitness::Evaluate(islands[i][p], instance, 0, generations);
        }
    }
}

Individual IslandModel::FindBestInIsland(int island_idx) const {
    Individual best = islands[island_idx][0];
    for (const auto& ind : islands[island_idx]) {
        if (ind.fitness > best.fitness) {
            best = ind;
        }
    }
    return best;
}

Individual IslandModel::FindBestOverall() const {
    Individual best = islands[0][0];
    for (int i = 0; i < num_islands; ++i) {
        for (const auto& ind : islands[i]) {
            if (ind.fitness > best.fitness) {
                best = ind;
            }
        }
    }
    return best;
}

void IslandModel::RecordStats(int gen) {
    GenerationStats stat;
    stat.generation = gen;
    stat.best_fitness = -1e9f;
    stat.worst_fitness = 1e9f;
    stat.valid_count = 0;
    
    float total_fitness = 0.0f;
    int total_pop = num_islands * population_per_island;
    
    Individual best_overall;
    bool first = true;
    
    for (int i = 0; i < num_islands; ++i) {
        for (const auto& ind : islands[i]) {
            if (first || ind.fitness > stat.best_fitness) {
                stat.best_fitness = ind.fitness;
                best_overall = ind;
                first = false;
            }
            if (ind.fitness < stat.worst_fitness) stat.worst_fitness = ind.fitness;
            total_fitness += ind.fitness;
            if (ind.is_valid) stat.valid_count++;
        }
    }
    
    stat.avg_fitness = total_fitness / total_pop;
    stat.best_is_valid = best_overall.is_valid;
    
    if (stats_.empty()) stat.convergence_delta = 0.0f;
    else stat.convergence_delta = std::abs(stat.best_fitness - stats_.back().best_fitness);
    
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
 * @brief Orquesta el intercambio biológico de individuos entre ecosistemas aislados (Islas).
 * 
 * Durante determinados intervalos (definidos por migration_frequency), las islas exportan 
 * su conjunto de super élites (definido por num_migrants) hacia otras islas de la red.
 * El destino se define en base a la topología configurada de reubicación poblacional (topology).
 * 
 * Topologías soportadas:
 * - "ring": Transfiere los mejores individuos cíclicamente hacia la isla vecina de la izquierda matemática: ((i - 1 + N) % N).
 * - "random": Escoge estocásticamente un ecosistema destino de la red (que no sea el de origen) para transferir a los individuos.
 *
 * Los inmigrantes llegan a la isla de destino reemplazando permanentemente a las composiciones 
 * genéticas con el factor de calidad más bajo (peor fitness), de manera que nunca se altera o 
 * desbalancea el límite poblacional en el arreglo subyacente.
 */
void IslandModel::Migrate() {
    std::vector<std::vector<Individual>> emigrants(num_islands);
    for (int i = 0; i < num_islands; ++i) {
        std::vector<Individual> sorted_island = islands[i];
        std::sort(sorted_island.begin(), sorted_island.end(),
                  [](const Individual& a, const Individual& b) { return a.fitness > b.fitness; });
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
            while(source_idx == i && num_islands > 1) {
                source_idx = dis(island_rngs[i]);
            }
        }
        
        std::vector<Individual>& target_island = islands[i];
        std::sort(target_island.begin(), target_island.end(),
                  [](const Individual& a, const Individual& b) { return a.fitness > b.fitness; });
                  
        for (int m = 0; m < num_migrants; ++m) {
            target_island[target_island.size() - 1 - m] = emigrants[source_idx][m];
        }
    }
}

/**
 * @brief Ejecuta de manera enteramente secuencial el flujo evolutivo del Algoritmo Genético sobre las islas.
 *
 * Itérase estrictamente la cantidad de ciclos descritos por parameterizaciones establecidas ("generations").
 * En cada ciclo, el sistema operativo transita secuencialmente por cada archipiélago para evolucionar a su 
 * respectiva población. El proceso general a ciclo de vida genético local emplea:
 * 1. Preservación estricta local por elitismo natural (FindBestInIsland), asegurando que ninguna buena cepa perezca.
 * 2. Operación de torneo contra especímenes aleatorizados para el pase de material entre padres.
 * 3. Cruce del tipo "Single Point/Un Punto", seguido de probabilidad de mutación genética (BitFlip).
 * 4. Actualización del factor de aptitud ante todos los nuevos miembros descendientes (Evaluate).
 * 
 * Si la generación de turno lograra coincidir con el lapso de `migration_frequency`, la evaluación se  
 * suspende momentáneamente con el objetivo de ejecutar intercambios horizontales sincrónicos de información (Migrate).
 */
void IslandModel::Run() {
    RecordStats(0);
    
    for (int gen = 1; gen <= generations; ++gen) {
        for (int i = 0; i < num_islands; ++i) {
            std::vector<Individual> new_island;
            new_island.reserve(population_per_island);
            // Elitismo local
            new_island.push_back(FindBestInIsland(i));
            
            while ((int)new_island.size() < population_per_island) {
                Individual p1 = Selection::Tournament(islands[i], 3, island_rngs[i]);
                Individual p2 = Selection::Tournament(islands[i], 3, island_rngs[i]);
                
                Individual c1, c2;
                Crossover::SinglePoint(p1, p2, c1, c2, island_rngs[i]);
                
                Mutation::BitFlip(c1, mutation_rate, island_rngs[i]);
                Fitness::Repair(c1, instance, island_rngs[i]);
                Fitness::Evaluate(c1, instance, gen, generations);
                new_island.push_back(c1);
                
                if ((int)new_island.size() < population_per_island) {
                    Mutation::BitFlip(c2, mutation_rate, island_rngs[i]);
                    Fitness::Repair(c2, instance, island_rngs[i]);
                    Fitness::Evaluate(c2, instance, gen, generations);
                    new_island.push_back(c2);
                }
            }
            islands[i] = std::move(new_island);
        }
        
        if (gen % migration_frequency == 0) {
            Migrate();
        }
        
        RecordStats(gen);

        if (HasConverged() && stats_.back().best_is_valid) {
            std::cout << "Convergencia detectada en generacion " << gen << "\n";
            break;
        }
    }
}

/**
 * @brief Orquestador acelerado del Modelo de Islas por multiprocesamiento.
 *
 * Utiliza estructuras de OpenMP (\pragma omp parallel for schedule(dynamic)) para procesar
 * iteraciones simultáneas y verdaderamente dinámicas de cada isla en un procesador o sub-hilo
 * independiente, sin la necesidad de tener estados críticos que puedan producir condiciones de carrera 
 * durante el ciclo genético y estocástico interno (ya que cada RNG es independiente en cada isla).
 * Después de que los hilos calculan la evolución de sus ecosistemas y regresan para sincronizarse en cada ciclo genérico, el proceso 
 * de migración se ejecuta asincrónicamente mediante una detención lógica.
 */
void IslandModel::RunParallel(int num_threads) {
    if (num_threads > 0) omp_set_num_threads(num_threads);
    
    RecordStats(0);
    
    for (int gen = 1; gen <= generations; ++gen) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_islands; ++i) {
            std::vector<Individual> new_island;
            new_island.reserve(population_per_island);
            // Elitismo local
            new_island.push_back(FindBestInIsland(i));
            
            while ((int)new_island.size() < population_per_island) {
                Individual p1 = Selection::Tournament(islands[i], 3, island_rngs[i]);
                Individual p2 = Selection::Tournament(islands[i], 3, island_rngs[i]);
                
                Individual c1, c2;
                Crossover::SinglePoint(p1, p2, c1, c2, island_rngs[i]);
                
                Mutation::BitFlip(c1, mutation_rate, island_rngs[i]);
                Fitness::Repair(c1, instance, island_rngs[i]);
                Fitness::Evaluate(c1, instance, gen, generations);
                new_island.push_back(c1);
                
                if ((int)new_island.size() < population_per_island) {
                    Mutation::BitFlip(c2, mutation_rate, island_rngs[i]);
                    Fitness::Repair(c2, instance, island_rngs[i]);
                    Fitness::Evaluate(c2, instance, gen, generations);
                    new_island.push_back(c2);
                }
            }
            islands[i] = std::move(new_island);
        }
        
        if (gen % migration_frequency == 0) {
            Migrate();
        }
        
        RecordStats(gen);

        if (HasConverged() && stats_.back().best_is_valid) {
            std::cout << "Convergencia detectada en generacion " << gen << "\n";
            break;
        }
    }
}

void IslandModel::View_Population() const {
    std::cout << "Poblaciones de Islas no mostradas detalladamente aún.\n";
}