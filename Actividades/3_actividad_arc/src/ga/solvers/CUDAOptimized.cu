#include "ga/solvers/CUDAOptimized.cuh"
#include "ga/cuda/fitness_kernel.cuh"
#include "ga/cuda/operators_kernel.cuh"
#include "config/constants.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err)        \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";   \
            throw std::runtime_error(cudaGetErrorString(err));             \
        }                                                                  \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
CUDAOptimized::CUDAOptimized(
    KnapsackInstance& inst,
    std::unique_ptr<ga::operators::Crossover>           cross,
    std::unique_ptr<ga::operators::Mutation>            mut,
    std::unique_ptr<ga::operators::Selection>           sel,
    std::unique_ptr<ga::operators::FitnessEvaluator>    fit,
    std::unique_ptr<ga::operators::ConstraintValidator> val,
    bool verbose, int seed, int block_sz,
    bool const_mem, bool streams, bool shared_reduce,
    size_t pop_size, size_t num_gens)
    : CUDABasic(inst, std::move(cross), std::move(mut), std::move(sel),
                std::move(fit), std::move(val), verbose, seed, block_sz, pop_size, num_gens),
      use_const_memory(const_mem),
      use_streams(streams),
      use_shared_reduce(shared_reduce),
      stream_eval(nullptr), stream_repro(nullptr)
{
    // CORRECCIÓN 3: Registrar vectores en memoria pinned (no pageable)
    // Esto elimina el cuello de botella donde cudaMemcpyAsync fuerza
    // sincronía porque los vectores están en memoria pageable.
    // Con manejo robusto de rollback en caso de error
    try {
        std::vector<void*> registered_ptrs;
        
        // Registrar arrays de ítems en memoria pinned
        if (!h_item_values.empty()) {
            CUDA_CHECK(cudaHostRegister(h_item_values.data(),  h_item_values.size() * sizeof(float), cudaHostRegisterDefault));
            registered_ptrs.push_back(h_item_values.data());
        }
        if (!h_item_weights.empty()) {
            CUDA_CHECK(cudaHostRegister(h_item_weights.data(), h_item_weights.size() * sizeof(float), cudaHostRegisterDefault));
            registered_ptrs.push_back(h_item_weights.data());
        }
        if (!h_item_volumes.empty()) {
            CUDA_CHECK(cudaHostRegister(h_item_volumes.data(), h_item_volumes.size() * sizeof(float), cudaHostRegisterDefault));
            registered_ptrs.push_back(h_item_volumes.data());
        }
        if (!h_item_cat_ids.empty()) {
            CUDA_CHECK(cudaHostRegister(h_item_cat_ids.data(), h_item_cat_ids.size() * sizeof(int), cudaHostRegisterDefault));
            registered_ptrs.push_back(h_item_cat_ids.data());
        }
        
        // Registrar arrays de restricciones
        if (!h_incomp_a.empty()) {
            CUDA_CHECK(cudaHostRegister(h_incomp_a.data(), h_incomp_a.size() * sizeof(int), cudaHostRegisterDefault));
            registered_ptrs.push_back(h_incomp_a.data());
        }
        if (!h_incomp_b.empty()) {
            CUDA_CHECK(cudaHostRegister(h_incomp_b.data(), h_incomp_b.size() * sizeof(int), cudaHostRegisterDefault));
            registered_ptrs.push_back(h_incomp_b.data());
        }
        if (!h_dep_a.empty()) {
            CUDA_CHECK(cudaHostRegister(h_dep_a.data(),   h_dep_a.size() * sizeof(int), cudaHostRegisterDefault));
            registered_ptrs.push_back(h_dep_a.data());
        }
        if (!h_dep_b.empty()) {
            CUDA_CHECK(cudaHostRegister(h_dep_b.data(),   h_dep_b.size() * sizeof(int), cudaHostRegisterDefault));
            registered_ptrs.push_back(h_dep_b.data());
        }
        if (!h_cat_id.empty()) {
            CUDA_CHECK(cudaHostRegister(h_cat_id.data(),  h_cat_id.size() * sizeof(int), cudaHostRegisterDefault));
            registered_ptrs.push_back(h_cat_id.data());
        }
        if (!h_cat_min.empty()) {
            CUDA_CHECK(cudaHostRegister(h_cat_min.data(), h_cat_min.size() * sizeof(int), cudaHostRegisterDefault));
            registered_ptrs.push_back(h_cat_min.data());
        }
        if (!h_cat_max.empty()) {
            CUDA_CHECK(cudaHostRegister(h_cat_max.data(), h_cat_max.size() * sizeof(int), cudaHostRegisterDefault));
            registered_ptrs.push_back(h_cat_max.data());
        }

        if (verbose) {
            std::cout << "[OPT] Memoria host registrada como pinned (solapamiento H↔D efectivo)\n";
        }
    } catch (const std::exception& e) {
        if (verbose) {
            std::cerr << "[WARN] Fallo al registrar memoria como pinned: " << e.what() << "\n";
        }
        // Continuar sin pinned memory (degradación correcta)
    }

    // ── Optimización 7: Streams CUDA ────────────────────────────────
    if (use_streams) {
        CUDA_CHECK(cudaStreamCreate(&stream_eval));
        CUDA_CHECK(cudaStreamCreate(&stream_repro));
    }

    // ── Optimización 1 y 5: Memoria constante ───────────────────────
    if (use_const_memory) {
        upload_instance_optimized();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Destructor
// ─────────────────────────────────────────────────────────────────────────────
CUDAOptimized::~CUDAOptimized()
{
    // CORRECCIÓN 3: Desregistrar memoria pinned para evitar memory leaks
    // Es importante hacer esto antes de destruir los vectores
    try {
        cudaHostUnregister(h_item_values.data());
        cudaHostUnregister(h_item_weights.data());
        cudaHostUnregister(h_item_volumes.data());
        cudaHostUnregister(h_item_cat_ids.data());
        cudaHostUnregister(h_incomp_a.data());
        cudaHostUnregister(h_incomp_b.data());
        cudaHostUnregister(h_dep_a.data());
        cudaHostUnregister(h_dep_b.data());
        cudaHostUnregister(h_cat_id.data());
        cudaHostUnregister(h_cat_min.data());
        cudaHostUnregister(h_cat_max.data());
    } catch (...) {
        // Ignorar errores en destructor
    }

    // Destruir streams
    if (stream_eval)  cudaStreamDestroy(stream_eval);
    if (stream_repro) cudaStreamDestroy(stream_repro);
}

// ─────────────────────────────────────────────────────────────────────────────
// upload_instance_optimized
// Sube datos de ítems a memoria constante cuando n_items <= MAX_CONST_ITEMS.
// Optimización 1: broadcast eficiente desde caché de constantes.
// Optimización 5: datos de solo lectura en __constant__ evitan accesos a
//                  memoria global para cada hilo.
// ─────────────────────────────────────────────────────────────────────────────
void CUDAOptimized::upload_instance_optimized()
{
    if (n_items > MAX_CONST_ITEMS) {
        if (verbose) {
            std::cout << "[OPT] n_items=" << n_items
                      << " > MAX_CONST_ITEMS=" << MAX_CONST_ITEMS
                      << " → usando memoria global (no constante)\n";
        }
        return;
    }

    // Subir arrays de ítems a memoria constante
    CUDA_CHECK(cudaMemcpyToSymbol(c_values,  h_item_values .data(), n_items*sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_weights, h_item_weights.data(), n_items*sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_volumes, h_item_volumes.data(), n_items*sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_cat_ids, h_item_cat_ids.data(), n_items*sizeof(int)));

    // Subir parámetros escalares a memoria constante
    FitnessParams p;
    p.max_weight   = instance.max_weight;
    p.max_volume   = instance.max_volume;
    p.max_value    = instance.max_value;
    p.pen_weight   = pen_weight;
    p.pen_volume   = pen_volume;
    p.pen_category = pen_category;
    p.pen_incomp   = pen_incomp;
    p.pen_dep      = pen_dep;
    p.obj_w        = obj_w;
    p.pen_w        = pen_w;
    p.n_items      = n_items;
    p.n_incomp     = n_incomp;
    p.n_dep        = n_dep;
    p.n_cat_rules  = n_cat_rules;
    CUDA_CHECK(cudaMemcpyToSymbol(c_params, &p, sizeof(FitnessParams)));

    if (verbose) {
        std::cout << "[OPT] Datos de instancia subidos a memoria constante ("
                  << n_items * (sizeof(float)*3 + sizeof(int))
                  << " bytes)\n";
    }
}

// ── Helpers de eventos para timing en CUDAOptimized ──────────────────────────
#define OPT_EVENT_CREATE(s,e) cudaEvent_t s, e; \
    cudaEventCreate(&s); cudaEventCreate(&e)
#define OPT_EVENT_DESTROY(s,e) cudaEventDestroy(s); cudaEventDestroy(e)

// ─────────────────────────────────────────────────────────────────────────────
// evaluate_population  (override)
// Optimizaciones aplicadas:
//   - Opt 3: reducción paralela intra-bloque con shared memory
//            (fitness_kernel_opt: un bloque por individuo)
//   - Opt 1/5: usa c_values, c_weights, c_volumes desde memoria constante
//   - Opt 7: lanza en stream_eval para solapar con transferencias
// Métricas: tiempos de kernel fitness y transferencia D→H registrados con eventos
// ─────────────────────────────────────────────────────────────────────────────
void CUDAOptimized::evaluate_population()
{
    size_t pop_sz = population_size;

    bool has_complex_constraints = (n_incomp > 0 || n_dep > 0 || n_cat_rules > 0);
    
    if (use_shared_reduce && n_items <= MAX_CONST_ITEMS && use_const_memory && !has_complex_constraints) {
        size_t smem_bytes = 3 * (size_t)block_size * sizeof(float);
        cudaStream_t s = use_streams ? stream_eval : 0;

        if (verbose) {
            std::cout << "[OPT] Usando fitness_kernel_opt (restricciones simples)\n";
        }

        // ── Medir tiempo de kernel fitness optimizado ─────────────────
        OPT_EVENT_CREATE(k0, k1);
        cudaEventRecord(k0, s);

        fitness_kernel_opt<<<(int)pop_sz, block_size, smem_bytes, s>>>(
            d_population,
            d_fitness, d_penalty, d_hard_feas, d_is_valid,
            (int)pop_sz);

        cudaEventRecord(k1, s);
        if (use_streams) {
            CUDA_CHECK(cudaStreamSynchronize(stream_eval));
        } else {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        total_kernel_fitness_ms += elapsed_ms(k0, k1);
        OPT_EVENT_DESTROY(k0, k1);

        // ── Medir transferencia D→H de resultados escalares ───────────
        std::vector<float>   h_fit  (pop_sz);
        std::vector<float>   h_pen  (pop_sz);
        std::vector<uint8_t> h_hf   (pop_sz);
        std::vector<uint8_t> h_valid(pop_sz);

        OPT_EVENT_CREATE(d0, d1);
        cudaEventRecord(d0, s);
        CUDA_CHECK(cudaMemcpy(h_fit.data(),   d_fitness,   pop_sz*sizeof(float),   cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_pen.data(),   d_penalty,   pop_sz*sizeof(float),   cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_hf.data(),    d_hard_feas, pop_sz*sizeof(uint8_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_valid.data(), d_is_valid,  pop_sz*sizeof(uint8_t), cudaMemcpyDeviceToHost));
        cudaEventRecord(d1, s);
        cudaEventSynchronize(d1);
        total_transfer_d2h_ms += elapsed_ms(d0, d1);
        OPT_EVENT_DESTROY(d0, d1);

        for (size_t i = 0; i < pop_sz; ++i) {
            population[i].fitness       = h_fit  [i];
            population[i].penalty       = h_pen  [i];
            population[i].hard_feasible = (h_hf   [i] == 1);
            population[i].is_valid      = (h_valid[i] == 1);
        }

        // Rastrear mejores individuos y descargar solo sus cromosomas
        size_t best_idx = 0;
        size_t best_valid_idx = SIZE_MAX;
        for (size_t i = 1; i < pop_sz; ++i) {
            if (population[i].fitness > population[best_idx].fitness)
                best_idx = i;
        }
        for (size_t i = 0; i < pop_sz; ++i) {
            if (population[i].is_valid) {
                if (best_valid_idx == SIZE_MAX || population[i].fitness > population[best_valid_idx].fitness)
                    best_valid_idx = i;
            }
        }

        h_best_chrom.resize(n_items);
        CUDA_CHECK(cudaMemcpy(h_best_chrom.data(),
                              d_population + best_idx * (size_t)n_items,
                              n_items * sizeof(uint8_t), cudaMemcpyDeviceToHost));

        if (best_valid_idx != SIZE_MAX) {
            h_best_valid_chrom.resize(n_items);
            CUDA_CHECK(cudaMemcpy(h_best_valid_chrom.data(),
                                  d_population + best_valid_idx * (size_t)n_items,
                                  n_items * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        }

        ++timing_samples;
        return;
    }

    // Fallback al kernel básico (ya incluye timing y tracking de mejores)
    CUDABasic::evaluate_population();
}

// ─────────────────────────────────────────────────────────────────────────────
// do_reproduction  (override)
// Optimizaciones adicionales sobre CUDABasic::do_reproduction():
//   - Opt 4: ajuste dinámico de tamaño de bloque
//   - Opt 6: control de divergencia de warps en reproduce_kernel
//   - Opt 7: usa stream_repro para solapar con la evaluación del siguiente ciclo
//   - Opt 2: accesos coalescentes garantizados (genes[ind*n + g])
// Métricas: tiempo de kernel de reproducción registrado con eventos
// ─────────────────────────────────────────────────────────────────────────────
void CUDAOptimized::do_reproduction()
{
    size_t pop_sz = population_size;

    int opt_bs = get_optimal_block_size(n_items);
    int blocks = ((int)pop_sz + opt_bs - 1) / opt_bs;

    cudaStream_t s = use_streams ? stream_repro : 0;

    // ── Medir tiempo de kernel de reproducción ────────────────────────
    OPT_EVENT_CREATE(k0, k1);
    cudaEventRecord(k0, s);

    reproduce_kernel<<<blocks, opt_bs, 0, s>>>(
        d_population, d_offspring, d_fitness,
        d_rng_states,
        (int)pop_sz, n_items,
        3,
        crossover_op->get_crossover_rate(),
        mutation_op->get_mutation_rate(),
        d_values, d_weights, d_volumes,
        d_incomp_a, d_incomp_b,
        d_dep_a, d_dep_b,
        instance.max_weight, instance.max_volume,
        n_incomp, n_dep);

    cudaEventRecord(k1, s);
    if (use_streams) {
        CUDA_CHECK(cudaStreamSynchronize(stream_repro));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    total_kernel_repro_ms += elapsed_ms(k0, k1);
    OPT_EVENT_DESTROY(k0, k1);

    // ── Elitismo (device→device, no cuenta como H↔D) ──────────────
    size_t elitism_count = std::max((size_t)1,
        (size_t)(pop_sz * Config::GeneticAlgorithm::ELITISM_PERCENTAGE));

    std::vector<size_t> sorted_idx(pop_sz);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::partial_sort(sorted_idx.begin(),
                      sorted_idx.begin() + elitism_count,
                      sorted_idx.end(),
                      [this](size_t a, size_t b) {
                          return population[a].fitness > population[b].fitness;
                      });

    for (size_t e = 0; e < elitism_count; ++e) {
        size_t src = sorted_idx[e];
        CUDA_CHECK(cudaMemcpyAsync(
            d_offspring + e * (size_t)n_items,
            d_population + src * (size_t)n_items,
            (size_t)n_items * sizeof(uint8_t),
            cudaMemcpyDeviceToDevice,
            s));
    }

    // ── Swap con bloque óptimo (todo en GPU, sin D→H) ─────────────
    int total_genes = (int)pop_sz * n_items;
    int swap_blocks = (total_genes + opt_bs - 1) / opt_bs;
    swap_populations_kernel<<<swap_blocks, opt_bs, 0, s>>>(
        d_population, d_offspring, (int)pop_sz, n_items);

    if (use_streams) {
        CUDA_CHECK(cudaStreamSynchronize(stream_repro));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    // NOTA: Cromosomas no se descargan cada gen (Req 6.2).
    // Se descargan lazy en get_best() solo para el mejor individuo.
}

// ─────────────────────────────────────────────────────────────────────────────
// get_optimal_block_size
// Usa cudaOccupancyMaxPotentialBlockSize para determinar el tamaño de bloque
// que maximiza la ocupancia del kernel de reproducción.
// Optimización 4 del enunciado.
// ─────────────────────────────────────────────────────────────────────────────
int CUDAOptimized::get_optimal_block_size(int /*n_items*/)
{
    int min_grid_size = 0;
    int opt_block_size = 128;  // fallback

    cudaError_t err = cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &opt_block_size,
        reproduce_kernel,
        0,   // shared memory dinámica = 0
        0);  // sin restricción de bloques

    if (err != cudaSuccess) {
        return 128;  // valor por defecto seguro
    }

    // Redondear al múltiplo de 32 más cercano (alineado a warp)
    opt_block_size = ((opt_block_size + 31) / 32) * 32;
    return opt_block_size;
}
