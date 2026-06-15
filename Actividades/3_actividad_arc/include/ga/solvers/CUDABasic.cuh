#pragma once
#include "ga/BaseGA.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>

/**
 * @brief Solver CUDA Básico del Algoritmo Genético
 *
 * Paraleliza en GPU:
 *   - Evaluación de aptitud (fitness_kernel)
 *   - Selección por torneo + cruzamiento + mutación (reproduce_kernel)
 *   - Búsqueda del mejor individuo en CPU (KnapsackFitness::get_best)
 *
 * Métricas registradas por generación:
 *   - Tiempo de kernel de fitness      (cudaEvent_t)
 *   - Tiempo de kernel de reproducción (cudaEvent_t)
 *   - Tiempo de transferencia H→D      (cudaEvent_t)
 *   - Tiempo de transferencia D→H      (cudaEvent_t)
 *
 * Representación coalescente: genes[ind * n_items + gene]
 */
class CUDABasic : public BaseGA
{
protected:
    // ── Parámetros GPU ───────────────────────────────────────────────
    int block_size;

    // ── Datos de instancia aplanados ────────────────────────────────
    int n_items, n_incomp, n_dep, n_cat_rules;

    std::vector<float> h_item_values;
    std::vector<float> h_item_weights;
    std::vector<float> h_item_volumes;
    std::vector<int>   h_item_cat_ids;
    std::vector<int>   h_incomp_a, h_incomp_b;
    std::vector<int>   h_dep_a,    h_dep_b;
    std::vector<int>   h_cat_id,   h_cat_min, h_cat_max;

    // Penalizaciones
    float pen_weight, pen_volume, pen_category, pen_incomp, pen_dep;
    float obj_w, pen_w;

    // ── Punteros device ──────────────────────────────────────────────
    uint8_t* d_population;
    uint8_t* d_offspring;
    float*   d_fitness;
    float*   d_penalty;
    uint8_t* d_hard_feas;
    uint8_t* d_is_valid;
    float*   d_values;
    float*   d_weights;
    float*   d_volumes;
    int*     d_cat_ids;
    int*     d_incomp_a, *d_incomp_b;
    int*     d_dep_a,    *d_dep_b;
    int*     d_cat_id,   *d_cat_min, *d_cat_max;
    curandState* d_rng_states;

    // ── Acumuladores de métricas CUDA (en ms) ───────────────────────
    float total_kernel_fitness_ms  = 0.f;
    float total_kernel_repro_ms    = 0.f;
    float total_transfer_h2d_ms    = 0.f;
    float total_transfer_d2h_ms    = 0.f;
    long  timing_samples           = 0;

    // ── Buffers para cromosomas de mejores individuos (evita D→H cada gen) ──
    std::vector<uint8_t> h_best_chrom;
    std::vector<uint8_t> h_best_valid_chrom;

    // ── Métodos internos ─────────────────────────────────────────────
    void flatten_instance();
    void upload_instance();
    void initialize_population() override;
    void evaluate_population()   override;
    void do_reproduction()       override;
    void free_device_memory();

    float elapsed_ms(cudaEvent_t start, cudaEvent_t stop);

public:
    CUDABasic(KnapsackInstance& inst,
              std::unique_ptr<ga::operators::Crossover>           cross,
              std::unique_ptr<ga::operators::Mutation>            mut,
              std::unique_ptr<ga::operators::Selection>           sel,
              std::unique_ptr<ga::operators::FitnessEvaluator>    fit,
              std::unique_ptr<ga::operators::ConstraintValidator> val,
              bool verbose  = false,
              int  seed     = 0,
              int  block_sz = 128,
              size_t pop_size = 0,
              size_t num_gens = 0);

    ~CUDABasic() override;

    // ── Setter para pesos de penalización (desde SolverConfig) ──────
    void set_penalty_weights(float w, float v, float c, float i, float d) {
        pen_weight = w;
        pen_volume = v;
        pen_category = c;
        pen_incomp = i;
        pen_dep = d;
    }

    // ── Accesores de métricas (llamados desde main después de run()) ──
    float get_kernel_fitness_ms()  const { return total_kernel_fitness_ms; }
    float get_kernel_repro_ms()    const { return total_kernel_repro_ms;   }
    float get_transfer_h2d_ms()    const { return total_transfer_h2d_ms;   }
    float get_transfer_d2h_ms()    const { return total_transfer_d2h_ms;   }
    long  get_timing_samples()     const { return timing_samples;          }

    /** Descarga el cromosoma del mejor individuo desde GPU (lazy, solo al final) */
    Individual get_best() override;

    /** Porcentaje de soluciones válidas en la población final */
    float get_feasible_pct() const {
        if (population.empty()) return 0.f;
        int cnt = 0;
        for (const auto& ind : population) if (ind.is_valid) cnt++;
        return 100.f * cnt / (float)population.size();
    }
};
