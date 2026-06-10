#pragma once
#include "ga/BaseGA.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>

/**
 * @brief Solver CUDA Básico del Algoritmo Genético
 *
 * Paraleliza en GPU las siguientes etapas:
 *   - Evaluación de aptitud (fitness_kernel)
 *   - Selección por torneo (tournament_kernel)
 *   - Cruzamiento de un punto (crossover_kernel)
 *   - Mutación uniforme (mutation_kernel)
 *   - Búsqueda del mejor individuo (reduce_best_kernel)
 *
 * La población se mantiene en memoria global de GPU durante toda la
 * evolución. Solo se transfiere a CPU al inicio (carga) y al final
 * (recuperar mejor individuo).
 *
 * Representación lineal orientada a accesos coalescentes:
 *   genes[ind * n_items + gene]
 */
class CUDABasic : public BaseGA
{
protected:
    // ── Parámetros GPU ──────────────────────────────────────────────
    int block_size;   ///< Hilos por bloque CUDA (default: 128)

    // ── Datos de instancia aplanados (para GPU) ──────────────────────
    int   n_items;
    int   n_incomp;
    int   n_dep;
    int   n_cat_rules;

    // Ítems
    std::vector<float> h_item_values;
    std::vector<float> h_item_weights;
    std::vector<float> h_item_volumes;
    std::vector<int>   h_item_cat_ids;   ///< índice numérico de categoría

    // Reglas de incompatibilidad  [n_incomp * 2]
    std::vector<int> h_incomp_a;
    std::vector<int> h_incomp_b;

    // Reglas de dependencia  [n_dep * 2]
    std::vector<int> h_dep_a;
    std::vector<int> h_dep_b;

    // Reglas de categoría  [n_cat_rules * 3]: {cat_id, min, max}
    std::vector<int> h_cat_id;
    std::vector<int> h_cat_min;
    std::vector<int> h_cat_max;

    // Penalizaciones (escalares)
    float pen_weight;
    float pen_volume;
    float pen_category;
    float pen_incomp;
    float pen_dep;
    float obj_w;
    float pen_w;

    // ── Punteros en device (GPU) ─────────────────────────────────────
    uint8_t* d_population;   ///< [pop_size * n_items]  genes actuales
    uint8_t* d_offspring;    ///< [pop_size * n_items]  genes nueva gen
    float*   d_fitness;      ///< [pop_size]
    float*   d_penalty;      ///< [pop_size]
    uint8_t* d_hard_feas;    ///< [pop_size]  1=hard_feasible
    uint8_t* d_is_valid;     ///< [pop_size]  1=totalmente válido

    // Datos de instancia en device
    float* d_values;
    float* d_weights;
    float* d_volumes;
    int*   d_cat_ids;
    int*   d_incomp_a;
    int*   d_incomp_b;
    int*   d_dep_a;
    int*   d_dep_b;
    int*   d_cat_id;
    int*   d_cat_min;
    int*   d_cat_max;

    curandState* d_rng_states; ///< Estado cuRAND por hilo

    // ── Métodos internos ─────────────────────────────────────────────

    /** Convierte las estructuras C++ de la instancia a arrays planos */
    void flatten_instance();

    /** Reserva y copia datos de instancia a GPU (solo una vez) */
    void upload_instance();

    /** Genera población inicial en CPU y la sube a GPU */
    void initialize_population() override;

    /** Lanza fitness_kernel para evaluar toda la población en GPU */
    void evaluate_population() override;

    /** Lanza kernels de selección, cruzamiento y mutación */
    void do_reproduction() override;

    /** Libera toda la memoria GPU */
    void free_device_memory();

public:
    CUDABasic(KnapsackInstance& inst,
              std::unique_ptr<ga::operators::Crossover>           cross,
              std::unique_ptr<ga::operators::Mutation>            mut,
              std::unique_ptr<ga::operators::Selection>           sel,
              std::unique_ptr<ga::operators::FitnessEvaluator>    fit,
              std::unique_ptr<ga::operators::ConstraintValidator> val,
              bool verbose   = false,
              int  seed      = 0,
              int  block_sz  = 128);

    ~CUDABasic() override;
};
