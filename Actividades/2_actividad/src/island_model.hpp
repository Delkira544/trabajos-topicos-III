#ifndef ISLAND_MODEL_HPP
#define ISLAND_MODEL_HPP

#include "genetic_algorithm.hpp"
#include <vector>
#include <string>
#include <random>

// =============================================================================
// IslandModel — variante con sub-poblaciones aisladas y migración periódica
// =============================================================================
// Mantiene K sub-poblaciones (islas) que evolucionan independientemente con
// su propio RNG. Cada `migration_frequency` generaciones, las islas envían
// sus mejores individuos a islas vecinas según una topología (ring/random).
//
// La isla i recibe migrantes de la isla `(i-1+K) mod K` (ring), o de un
// origen aleatorio distinto a i (random). Los migrantes reemplazan a los
// peores de la isla destino.
//
// Hereda todas las mejoras del AG estándar a través del helper EvolveIsland().
// =============================================================================

/**
 * @brief AG con modelo de islas para la mochila extendida.
 *
 * Variante paralela natural: cada isla puede evolucionar en un hilo OpenMP
 * independiente (`#pragma omp parallel for` sobre islands en RunParallel).
 */
class IslandModel {
private:
    // ── Configuración inmutable ──────────────────────────────────────────
    Instance instance;              ///< Datos del problema
    int num_islands;                ///< K = nº de sub-poblaciones
    int population_per_island;      ///< Tamaño de cada isla
    int generations;                ///< Techo de generaciones
    float mutation_rate;            ///< Tasa base de mutación
    int migration_frequency;        ///< Cada cuántas gens se migra
    int num_migrants;               ///< Cuántos top migran por isla
    std::string topology;           ///< "ring" o "random"
    int seed;                       ///< Semilla base (cada isla usa seed+i)
    float convergence_threshold_;   ///< Umbral para HasConverged

    // ── Estado del bucle ─────────────────────────────────────────────────
    std::vector<std::vector<Individual>> islands;  ///< Matriz K × pop_per_island
    std::vector<std::mt19937> island_rngs;         ///< RNG independiente por isla
    std::vector<GenerationStats> stats_;           ///< Log de stats por gen

    // ── Mejoras integradas del AG estándar ─────────────────────────
    // #4 Uniform crossover con sesgo a no incluir
    float crossover_bias_ = 0.45f;
    // #5 Mutación dirigida a violaciones soft (probabilidad por hijo)
    float targeted_fix_prob_ = 0.5f;
    // #7 Multi-elite del 10% dentro de cada isla
    float elite_fraction_ = 0.10f;
    // #8 Tournament k=5
    int   tournament_size_ = 5;

    // #6 Anti-estancamiento global (migración ya aporta diversidad → límite mayor)
    int   stagnation_limit_ = 100;
    float diversity_inject_fraction_ = 0.30f;
    int   gens_without_improvement_ = 0;
    Individual best_ever_;

    // #12 Early-stop por estancamiento total (NO se resetea con injection)
    int   near_feasible_max_violations_ = 0;
    int   near_feasible_stall_limit_ = 250;
    int   gens_no_improve_total_ = 0;

    // Early-stop por "factibilidad alcanzada + N gens de refinamiento":
    // cuando best_ever_ se vuelve válido por primera vez, registramos la gen.
    // 75 gens después salimos (independiente de mejora posterior).
    int   gens_after_feasible_limit_ = 75;
    int   first_feasible_gen_ = -1;

    // ── Métodos privados de gestión ───────────────────────────────────────
    void InitializeIslands();                          ///< Crea K islas iniciales
    void Migrate();                                    ///< Intercambia migrantes
    Individual FindBestOverall() const;                ///< Mejor de TODAS las islas
    Individual FindBestInIsland(int island_idx) const; ///< Mejor de UNA isla
    void RecordStats(int gen);                         ///< Append stats al log

    // ── Helpers compartidos con el AG estándar ──────────────────────────
    /// Genera un individuo con Bernoulli(0.25) + corte greedy de capacidad.
    Individual CreateRandomIndividual(std::mt19937 &r) const;
    /// Aplica UNA generación de evolución a UNA isla (puede llamarse en paralelo).
    void EvolveIsland(int idx, int gen);
    /// Reemplaza el peor 30% de TODAS las islas (anti-estancamiento global).
    void InjectDiversityAllIslands();
    /// Cuenta violaciones individuales (peso + vol + cat + incompat + dep).
    int  CountTotalViolations(const Individual &ind) const;

public:
    /// Constructor — recibe TODOS los parámetros e inicializa las islas.
    IslandModel(const Instance& inst, int islands_count, int pop_per_island,
                int gens, float mut_rate, int mig_freq, int n_migrants,
                const std::string& top, int s);

    /// Bucle evolutivo secuencial (un solo hilo).
    void Run();
    /// Bucle evolutivo paralelo (un hilo por isla, schedule dynamic).
    void RunParallel(int num_threads = 0);

    /// Ajusta el umbral de convergencia clásica.
    void SetConvergenceThreshold(float threshold) { convergence_threshold_ = threshold; }
    /// True si Δ promedio sobre últimas 10 gens es < umbral.
    bool HasConverged() const;
    /// Acceso de sólo lectura al log de stats.
    const std::vector<GenerationStats>& GetStats() const { return stats_; }
    /// Devuelve el mejor individuo de TODAS las islas.
    Individual GetBestSolution() const { return FindBestOverall(); }
    /// Imprime resumen de la mega-población (debug).
    void View_Population() const;
};

#endif
