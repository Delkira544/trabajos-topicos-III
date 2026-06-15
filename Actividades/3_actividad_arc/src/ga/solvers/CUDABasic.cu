#include "ga/solvers/CUDABasic.cuh"
#include "ga/cuda/fitness_kernel.cuh"
#include "ga/cuda/operators_kernel.cuh"
#include "config/constants.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// ── Macro de verificación de errores CUDA ────────────────────────────────────
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            std::cerr << "[CUDA ERROR] " << cudaGetErrorString(_e)          \
                      << "  " << __FILE__ << ":" << __LINE__ << "\n";      \
            throw std::runtime_error(cudaGetErrorString(_e));               \
        }                                                                   \
    } while (0)

// ── Helpers de eventos CUDA ───────────────────────────────────────────────────
// Crean, registran y destruyen un par de eventos midiendo el intervalo en ms.
#define EVENT_CREATE(s,e)  cudaEvent_t s, e; \
                           cudaEventCreate(&s); cudaEventCreate(&e)
#define EVENT_DESTROY(s,e) cudaEventDestroy(s); cudaEventDestroy(e)

float CUDABasic::elapsed_ms(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.f;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
CUDABasic::CUDABasic(KnapsackInstance& inst,
                     std::unique_ptr<ga::operators::Crossover>           cross,
                     std::unique_ptr<ga::operators::Mutation>            mut,
                     std::unique_ptr<ga::operators::Selection>           sel,
                     std::unique_ptr<ga::operators::FitnessEvaluator>    fit,
                     std::unique_ptr<ga::operators::ConstraintValidator> val,
                     bool verbose, int seed, int block_sz,
                     size_t pop_size, size_t num_gens)
    : BaseGA(inst, std::move(cross), std::move(mut), std::move(sel),
             std::move(fit), std::move(val), verbose, seed, pop_size, num_gens),
      block_size(block_sz),
      d_population(nullptr), d_offspring(nullptr),
      d_fitness(nullptr),    d_penalty(nullptr),
      d_hard_feas(nullptr),  d_is_valid(nullptr),
      d_values(nullptr),     d_weights(nullptr),  d_volumes(nullptr),
      d_cat_ids(nullptr),
      d_incomp_a(nullptr),   d_incomp_b(nullptr),
      d_dep_a(nullptr),      d_dep_b(nullptr),
      d_cat_id(nullptr),     d_cat_min(nullptr),  d_cat_max(nullptr),
      d_rng_states(nullptr)
{
    pen_weight   = Config::Penalty::WEIGHT_EXCESS_PENALTY;
    pen_volume   = Config::Penalty::VOLUME_EXCESS_PENALTY;
    pen_category = Config::Penalty::CATEGORY_VIOLATION_PENALTY;
    pen_incomp   = Config::Penalty::INCOMPATIBILITY_PENALTY;
    pen_dep      = Config::Penalty::DEPENDENCY_VIOLATION_PENALTY;
    obj_w        = Config::Penalty::OBJ_WEIGHT_PENALTY;
    pen_w        = Config::Penalty::PEN_WEIGHT_PENALTY;

    flatten_instance();
    upload_instance();
}

CUDABasic::~CUDABasic() { free_device_memory(); }

// ─────────────────────────────────────────────────────────────────────────────
// flatten_instance  — estructuras C++ → arrays planos GPU-friendly
// ─────────────────────────────────────────────────────────────────────────────
void CUDABasic::flatten_instance()
{
    n_items = (int)instance.items.size();

    std::unordered_map<std::string, int> cat_map;
    int next_cat = 0;

    h_item_values .resize(n_items);
    h_item_weights.resize(n_items);
    h_item_volumes.resize(n_items);
    h_item_cat_ids.resize(n_items);

    for (int i = 0; i < n_items; ++i) {
        const Item& it = instance.items[i];
        h_item_values [i] = it.value;
        h_item_weights[i] = it.weight;
        h_item_volumes[i] = it.volume;
        auto jt = cat_map.find(it.category);
        if (jt == cat_map.end()) { cat_map[it.category] = next_cat; h_item_cat_ids[i] = next_cat++; }
        else                     { h_item_cat_ids[i] = jt->second; }
    }

    std::unordered_map<int,int> id2idx;
    for (int i = 0; i < n_items; ++i) id2idx[instance.items[i].id] = i;

    n_incomp = (int)instance.incompatibility_rules.size();
    h_incomp_a.resize(n_incomp); h_incomp_b.resize(n_incomp);
    for (int r = 0; r < n_incomp; ++r) {
        if (!id2idx.count(instance.incompatibility_rules[r].item_id_a)) {
            throw std::runtime_error(
                "Incompatibility rule references non-existent item_id_a: " +
                std::to_string(instance.incompatibility_rules[r].item_id_a)
            );
        }
        if (!id2idx.count(instance.incompatibility_rules[r].item_id_b)) {
            throw std::runtime_error(
                "Incompatibility rule references non-existent item_id_b: " +
                std::to_string(instance.incompatibility_rules[r].item_id_b)
            );
        }
        h_incomp_a[r] = id2idx[instance.incompatibility_rules[r].item_id_a];
        h_incomp_b[r] = id2idx[instance.incompatibility_rules[r].item_id_b];
    }

    n_dep = (int)instance.dependency_rules.size();
    h_dep_a.resize(n_dep); h_dep_b.resize(n_dep);
    for (int r = 0; r < n_dep; ++r) {
        if (!id2idx.count(instance.dependency_rules[r].item_id_a)) {
            throw std::runtime_error(
                "Dependency rule references non-existent item_id_a: " +
                std::to_string(instance.dependency_rules[r].item_id_a)
            );
        }
        if (!id2idx.count(instance.dependency_rules[r].item_id_b)) {
            throw std::runtime_error(
                "Dependency rule references non-existent item_id_b: " +
                std::to_string(instance.dependency_rules[r].item_id_b)
            );
        }
        h_dep_a[r] = id2idx[instance.dependency_rules[r].item_id_a];
        h_dep_b[r] = id2idx[instance.dependency_rules[r].item_id_b];
    }

    n_cat_rules = (int)instance.category_rules.size();
    h_cat_id.resize(n_cat_rules); h_cat_min.resize(n_cat_rules); h_cat_max.resize(n_cat_rules);
    int r = 0;
    for (const auto& [name, rule] : instance.category_rules) {
        h_cat_id [r] = cat_map.count(name) ? cat_map[name] : -1;
        h_cat_min[r] = rule.min;
        h_cat_max[r] = rule.max;
        ++r;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// upload_instance  — copia datos de instancia a GPU (una sola vez)
// Mide tiempo de transferencia H→D y lo acumula en total_transfer_h2d_ms
// ─────────────────────────────────────────────────────────────────────────────
void CUDABasic::upload_instance()
{
    size_t pop_sz = population_size;
    size_t n      = (size_t)n_items;

    EVENT_CREATE(ev0, ev1);
    cudaEventRecord(ev0);

    CUDA_CHECK(cudaMalloc(&d_values,  n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_volumes, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cat_ids, n*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_values,  h_item_values .data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_item_weights.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_volumes, h_item_volumes.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cat_ids, h_item_cat_ids.data(), n*sizeof(int),   cudaMemcpyHostToDevice));

    if (n_incomp > 0) {
        CUDA_CHECK(cudaMalloc(&d_incomp_a, n_incomp*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_incomp_b, n_incomp*sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_incomp_a, h_incomp_a.data(), n_incomp*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_incomp_b, h_incomp_b.data(), n_incomp*sizeof(int), cudaMemcpyHostToDevice));
    }
    if (n_dep > 0) {
        CUDA_CHECK(cudaMalloc(&d_dep_a, n_dep*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_dep_b, n_dep*sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_dep_a, h_dep_a.data(), n_dep*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dep_b, h_dep_b.data(), n_dep*sizeof(int), cudaMemcpyHostToDevice));
    }
    if (n_cat_rules > 0) {
        CUDA_CHECK(cudaMalloc(&d_cat_id,  n_cat_rules*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cat_min, n_cat_rules*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cat_max, n_cat_rules*sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_cat_id,  h_cat_id .data(), n_cat_rules*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cat_min, h_cat_min.data(), n_cat_rules*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cat_max, h_cat_max.data(), n_cat_rules*sizeof(int), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMalloc(&d_population, pop_sz*n*sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_offspring,  pop_sz*n*sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_fitness,    pop_sz*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_penalty,    pop_sz*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hard_feas,  pop_sz*sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_is_valid,   pop_sz*sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_rng_states, pop_sz*sizeof(curandState)));

    cudaEventRecord(ev1);
    total_transfer_h2d_ms += elapsed_ms(ev0, ev1);
    EVENT_DESTROY(ev0, ev1);

    // Inicializar cuRAND
    int blocks = ((int)pop_sz + block_size - 1) / block_size;
    init_rng_kernel<<<blocks, block_size>>>(
        d_rng_states, (unsigned long long)rng(), (int)pop_sz);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ─────────────────────────────────────────────────────────────────────────────
// initialize_population  — CPU genera población, luego H→D
// ─────────────────────────────────────────────────────────────────────────────
void CUDABasic::initialize_population()
{
    BaseGA::initialize_population();

    size_t pop_sz = population_size;
    std::vector<uint8_t> h_genes(pop_sz * (size_t)n_items);
    for (size_t i = 0; i < pop_sz; ++i)
        for (int g = 0; g < n_items; ++g)
            h_genes[i*(size_t)n_items + g] = population[i].chromosome[g] ? 1u : 0u;

    // Medir transferencia H→D de la población inicial
    EVENT_CREATE(ev0, ev1);
    cudaEventRecord(ev0);
    CUDA_CHECK(cudaMemcpy(d_population, h_genes.data(),
                          pop_sz*(size_t)n_items*sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    cudaEventRecord(ev1);
    total_transfer_h2d_ms += elapsed_ms(ev0, ev1);
    EVENT_DESTROY(ev0, ev1);
}

// ─────────────────────────────────────────────────────────────────────────────
// evaluate_population  — fitness_kernel + D→H de resultados
// ─────────────────────────────────────────────────────────────────────────────
void CUDABasic::evaluate_population()
{
    size_t pop_sz = population_size;
    int    blocks = ((int)pop_sz + block_size - 1) / block_size;

    FitnessParams p;
    p.max_weight   = instance.max_weight;  p.max_volume   = instance.max_volume;
    p.max_value    = instance.max_value;   p.pen_weight   = pen_weight;
    p.pen_volume   = pen_volume;           p.pen_category = pen_category;
    p.pen_incomp   = pen_incomp;           p.pen_dep      = pen_dep;
    p.obj_w        = obj_w;                p.pen_w        = pen_w;
    p.n_items      = n_items;              p.n_incomp     = n_incomp;
    p.n_dep        = n_dep;                p.n_cat_rules  = n_cat_rules;

    // ── Medir tiempo de kernel fitness ───────────────────────────────
    EVENT_CREATE(k0, k1);
    cudaEventRecord(k0);
    fitness_kernel<<<blocks, block_size>>>(
        d_population,
        d_values, d_weights, d_volumes, d_cat_ids,
        d_incomp_a, d_incomp_b, d_dep_a, d_dep_b,
        d_cat_id, d_cat_min, d_cat_max,
        d_fitness, d_penalty, d_hard_feas, d_is_valid,
        (int)pop_sz, p);
    cudaEventRecord(k1);
    total_kernel_fitness_ms += elapsed_ms(k0, k1);
    EVENT_DESTROY(k0, k1);

    // ── Medir transferencia D→H de resultados escalares ──────────────
    std::vector<float>   h_fit  (pop_sz);
    std::vector<float>   h_pen  (pop_sz);
    std::vector<uint8_t> h_hf   (pop_sz);
    std::vector<uint8_t> h_valid(pop_sz);

    EVENT_CREATE(d0, d1);
    cudaEventRecord(d0);
    CUDA_CHECK(cudaMemcpy(h_fit.data(),   d_fitness,   pop_sz*sizeof(float),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pen.data(),   d_penalty,   pop_sz*sizeof(float),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_hf.data(),    d_hard_feas, pop_sz*sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_valid.data(), d_is_valid,  pop_sz*sizeof(uint8_t), cudaMemcpyDeviceToHost));
    cudaEventRecord(d1);
    total_transfer_d2h_ms += elapsed_ms(d0, d1);
    EVENT_DESTROY(d0, d1);

    for (size_t i = 0; i < pop_sz; ++i) {
        population[i].fitness       = h_fit  [i];
        population[i].penalty       = h_pen  [i];
        population[i].hard_feasible = (h_hf   [i] == 1);
        population[i].is_valid      = (h_valid[i] == 1);
    }

    // Rastrear mejores individuos y descargar solo sus cromosomas (unos pocos KB)
    // en vez de toda la población (MB). Req 6.2: evitar transferencias masivas.
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

    // Descargar solo el cromosoma del mejor individuo (n_items bytes, no pop_sz*n_items)
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
}

// ─────────────────────────────────────────────────────────────────────────────
// do_reproduction  — reproduce_kernel + elitismo + swap + D→H cromosomas
// ─────────────────────────────────────────────────────────────────────────────
void CUDABasic::do_reproduction()
{
    size_t pop_sz = population_size;
    int    blocks = ((int)pop_sz + block_size - 1) / block_size;

    // ── Medir tiempo de kernel de reproducción ────────────────────────
    EVENT_CREATE(k0, k1);
    cudaEventRecord(k0);
    reproduce_kernel<<<blocks, block_size>>>(
        d_population, d_offspring, d_fitness, d_rng_states,
        (int)pop_sz, n_items, 3,
        crossover_op->get_crossover_rate(),
        mutation_op->get_mutation_rate(),
        // ─── Punteros adicionales para reparación ──────
        d_values, d_weights, d_volumes,
        d_incomp_a, d_incomp_b,
        d_dep_a, d_dep_b,
        instance.max_weight, instance.max_volume,
        n_incomp, n_dep);
    cudaEventRecord(k1);
    total_kernel_repro_ms += elapsed_ms(k0, k1);
    EVENT_DESTROY(k0, k1);

    // ── Elitismo (device→device, no cuenta como H↔D) ─────────────────
    size_t elitism_count = std::max((size_t)1,
        (size_t)(pop_sz * Config::GeneticAlgorithm::ELITISM_PERCENTAGE));
    std::vector<size_t> sorted_idx(pop_sz);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::partial_sort(sorted_idx.begin(), sorted_idx.begin() + elitism_count,
                      sorted_idx.end(), [this](size_t a, size_t b){
                          return population[a].fitness > population[b].fitness; });
    for (size_t e = 0; e < elitism_count; ++e) {
        size_t src = sorted_idx[e];
        CUDA_CHECK(cudaMemcpy(
            d_offspring  + e   * (size_t)n_items,
            d_population + src * (size_t)n_items,
            (size_t)n_items*sizeof(uint8_t), cudaMemcpyDeviceToDevice));
    }

    // ── Swap offspring → population (todo en GPU, sin D→H) ──────────
    int total = (int)pop_sz * n_items;
    swap_populations_kernel<<<(total+block_size-1)/block_size, block_size>>>(
        d_population, d_offspring, (int)pop_sz, n_items);
    CUDA_CHECK(cudaDeviceSynchronize());
    // NOTA: No se descargan cromosomas a CPU cada generación (Req 6.2).
    // Los cromosomas del mejor individuo se descargan lazy en get_best().
    // Solo se transfieren fitness/penalty/flags escalares (~16 bytes/ind).
}

// ─────────────────────────────────────────────────────────────────────────────
// free_device_memory
// ─────────────────────────────────────────────────────────────────────────────
void CUDABasic::free_device_memory()
{
    auto sf = [](void* p){ if (p) cudaFree(p); };
    sf(d_population); sf(d_offspring);
    sf(d_fitness);    sf(d_penalty);
    sf(d_hard_feas);  sf(d_is_valid);
    sf(d_values);     sf(d_weights);   sf(d_volumes);  sf(d_cat_ids);
    sf(d_incomp_a);   sf(d_incomp_b);
    sf(d_dep_a);      sf(d_dep_b);
    sf(d_cat_id);     sf(d_cat_min);   sf(d_cat_max);
    sf(d_rng_states);
}

// ─────────────────────────────────────────────────────────────────────────────
// get_best  — descarga lazy del cromosoma del mejor individuo desde GPU
// Solo se ejecuta UNA vez, al final de la ejecución, cuando main.cpp lo solicita.
// Esto evita transferir la población completa D→H cada generación.
// ─────────────────────────────────────────────────────────────────────────────
Individual CUDABasic::get_best()
{
    Individual best = BaseGA::get_best();

    // Si ya tiene cromosoma válido, no re-descargar
    if (best.chromosome.size() == (size_t)n_items)
        return best;

    // Determinar qué cromosoma usar: mejor fitness o mejor factible
    const std::vector<uint8_t>& src =
        (!best.is_valid && found_valid_solution && !h_best_valid_chrom.empty())
            ? h_best_valid_chrom : h_best_chrom;

    if (src.empty())
        return best;

    best.chromosome.resize(n_items);
    for (int g = 0; g < n_items; ++g)
        best.chromosome[g] = (src[g] != 0);

    return best;
}
