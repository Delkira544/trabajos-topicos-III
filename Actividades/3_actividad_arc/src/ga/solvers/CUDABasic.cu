#include "ga/solvers/CUDABasic.cuh"
#include "ga/cuda/fitness_kernel.cuh"
#include "ga/cuda/operators_kernel.cuh"
#include "ga/cuda/reduction_kernel.cuh"
#include "config/constants.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Macro para verificar errores CUDA
// ─────────────────────────────────────────────────────────────────────────────
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
CUDABasic::CUDABasic(KnapsackInstance& inst,
                     std::unique_ptr<ga::operators::Crossover>           cross,
                     std::unique_ptr<ga::operators::Mutation>            mut,
                     std::unique_ptr<ga::operators::Selection>           sel,
                     std::unique_ptr<ga::operators::FitnessEvaluator>    fit,
                     std::unique_ptr<ga::operators::ConstraintValidator> val,
                     bool verbose, int seed, int block_sz)
    : BaseGA(inst, std::move(cross), std::move(mut), std::move(sel),
             std::move(fit), std::move(val), verbose, seed),
      block_size(block_sz),
      d_population(nullptr), d_offspring(nullptr),
      d_fitness(nullptr), d_penalty(nullptr),
      d_hard_feas(nullptr), d_is_valid(nullptr),
      d_values(nullptr), d_weights(nullptr), d_volumes(nullptr),
      d_cat_ids(nullptr),
      d_incomp_a(nullptr), d_incomp_b(nullptr),
      d_dep_a(nullptr), d_dep_b(nullptr),
      d_cat_id(nullptr), d_cat_min(nullptr), d_cat_max(nullptr),
      d_rng_states(nullptr)
{
    // Leer penalizaciones y pesos de Config
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

// ─────────────────────────────────────────────────────────────────────────────
// Destructor
// ─────────────────────────────────────────────────────────────────────────────
CUDABasic::~CUDABasic()
{
    free_device_memory();
}

// ─────────────────────────────────────────────────────────────────────────────
// flatten_instance
// Convierte las estructuras C++ con strings/maps en arrays planos aptos
// para la GPU. Los nombres de categoría se convierten a índices enteros.
// ─────────────────────────────────────────────────────────────────────────────
void CUDABasic::flatten_instance()
{
    n_items = (int)instance.items.size();

    // Asignar índices numéricos a categorías
    std::unordered_map<std::string, int> cat_index_map;
    int next_cat_id = 0;

    h_item_values .resize(n_items);
    h_item_weights.resize(n_items);
    h_item_volumes.resize(n_items);
    h_item_cat_ids.resize(n_items);

    for (int i = 0; i < n_items; ++i) {
        const Item& it = instance.items[i];
        h_item_values [i] = it.value;
        h_item_weights[i] = it.weight;
        h_item_volumes[i] = it.volume;

        auto it2 = cat_index_map.find(it.category);
        if (it2 == cat_index_map.end()) {
            cat_index_map[it.category] = next_cat_id;
            h_item_cat_ids[i] = next_cat_id++;
        } else {
            h_item_cat_ids[i] = it2->second;
        }
    }

    // Incompatibilidades: mapear item.id → índice en array
    std::unordered_map<int, int> id_to_idx;
    for (int i = 0; i < n_items; ++i) id_to_idx[instance.items[i].id] = i;

    n_incomp = (int)instance.incompatibility_rules.size();
    h_incomp_a.resize(n_incomp);
    h_incomp_b.resize(n_incomp);
    for (int r = 0; r < n_incomp; ++r) {
        h_incomp_a[r] = id_to_idx.count(instance.incompatibility_rules[r].item_id_a)
                        ? id_to_idx[instance.incompatibility_rules[r].item_id_a] : 0;
        h_incomp_b[r] = id_to_idx.count(instance.incompatibility_rules[r].item_id_b)
                        ? id_to_idx[instance.incompatibility_rules[r].item_id_b] : 0;
    }

    // Dependencias
    n_dep = (int)instance.dependency_rules.size();
    h_dep_a.resize(n_dep);
    h_dep_b.resize(n_dep);
    for (int r = 0; r < n_dep; ++r) {
        h_dep_a[r] = id_to_idx.count(instance.dependency_rules[r].item_id_a)
                     ? id_to_idx[instance.dependency_rules[r].item_id_a] : 0;
        h_dep_b[r] = id_to_idx.count(instance.dependency_rules[r].item_id_b)
                     ? id_to_idx[instance.dependency_rules[r].item_id_b] : 0;
    }

    // Reglas de categoría
    n_cat_rules = (int)instance.category_rules.size();
    h_cat_id .resize(n_cat_rules);
    h_cat_min.resize(n_cat_rules);
    h_cat_max.resize(n_cat_rules);
    int r = 0;
    for (const auto& [cat_name, rule] : instance.category_rules) {
        h_cat_id [r] = cat_index_map.count(cat_name) ? cat_index_map[cat_name] : -1;
        h_cat_min[r] = rule.min;
        h_cat_max[r] = rule.max;
        ++r;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// upload_instance
// Sube los datos de instancia a memoria global de GPU (solo una vez).
// La población NO se sube aquí; se sube en initialize_population().
// ─────────────────────────────────────────────────────────────────────────────
void CUDABasic::upload_instance()
{
    size_t pop_sz  = population_size;
    size_t n       = (size_t)n_items;

    // ── Arrays de ítems ──────────────────────────────────────────────
    CUDA_CHECK(cudaMalloc(&d_values,  n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_volumes, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cat_ids, n * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_values,  h_item_values .data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_item_weights.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_volumes, h_item_volumes.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cat_ids, h_item_cat_ids.data(), n*sizeof(int),   cudaMemcpyHostToDevice));

    // ── Reglas de incompatibilidad ───────────────────────────────────
    if (n_incomp > 0) {
        CUDA_CHECK(cudaMalloc(&d_incomp_a, n_incomp * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_incomp_b, n_incomp * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_incomp_a, h_incomp_a.data(), n_incomp*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_incomp_b, h_incomp_b.data(), n_incomp*sizeof(int), cudaMemcpyHostToDevice));
    }

    // ── Reglas de dependencia ────────────────────────────────────────
    if (n_dep > 0) {
        CUDA_CHECK(cudaMalloc(&d_dep_a, n_dep * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_dep_b, n_dep * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_dep_a, h_dep_a.data(), n_dep*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dep_b, h_dep_b.data(), n_dep*sizeof(int), cudaMemcpyHostToDevice));
    }

    // ── Reglas de categoría ──────────────────────────────────────────
    if (n_cat_rules > 0) {
        CUDA_CHECK(cudaMalloc(&d_cat_id,  n_cat_rules * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cat_min, n_cat_rules * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cat_max, n_cat_rules * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_cat_id,  h_cat_id .data(), n_cat_rules*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cat_min, h_cat_min.data(), n_cat_rules*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cat_max, h_cat_max.data(), n_cat_rules*sizeof(int), cudaMemcpyHostToDevice));
    }

    // ── Buffers de población y fitness ──────────────────────────────
    size_t pop_bytes = pop_sz * n * sizeof(uint8_t);
    CUDA_CHECK(cudaMalloc(&d_population, pop_bytes));
    CUDA_CHECK(cudaMalloc(&d_offspring,  pop_bytes));
    CUDA_CHECK(cudaMalloc(&d_fitness,    pop_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_penalty,    pop_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hard_feas,  pop_sz * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_is_valid,   pop_sz * sizeof(uint8_t)));

    // ── Estados cuRAND (un estado por individuo) ─────────────────────
    CUDA_CHECK(cudaMalloc(&d_rng_states, pop_sz * sizeof(curandState)));

    // Inicializar cuRAND
    int blocks = ((int)pop_sz + block_size - 1) / block_size;
    init_rng_kernel<<<blocks, block_size>>>(
        d_rng_states, (unsigned long long)rng(), (int)pop_sz);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ─────────────────────────────────────────────────────────────────────────────
// initialize_population
// Genera la población inicial en CPU (igual que GeneticSolver) y la
// transfiere a GPU en un solo cudaMemcpy.
// ─────────────────────────────────────────────────────────────────────────────
void CUDABasic::initialize_population()
{
    // Reutilizar la inicialización CPU de GeneticSolver
    BaseGA::initialize_population();

    // Aplanar el vector<Individual> a un buffer contiguo
    size_t pop_sz = population_size;
    std::vector<uint8_t> h_genes(pop_sz * (size_t)n_items);

    for (size_t i = 0; i < pop_sz; ++i) {
        for (int g = 0; g < n_items; ++g) {
            h_genes[i * (size_t)n_items + g] =
                population[i].chromosome[g] ? 1u : 0u;
        }
    }

    CUDA_CHECK(cudaMemcpy(d_population, h_genes.data(),
                          pop_sz * (size_t)n_items * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
}

// ─────────────────────────────────────────────────────────────────────────────
// evaluate_population
// Lanza fitness_kernel y luego actualiza population[] en CPU a partir de
// los resultados de GPU (solo fitness, hard_feas, is_valid, penalty).
// ─────────────────────────────────────────────────────────────────────────────
void CUDABasic::evaluate_population()
{
    size_t pop_sz = population_size;
    int blocks = ((int)pop_sz + block_size - 1) / block_size;

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

    fitness_kernel<<<blocks, block_size>>>(
        d_population,
        d_values, d_weights, d_volumes, d_cat_ids,
        d_incomp_a, d_incomp_b,
        d_dep_a,    d_dep_b,
        d_cat_id,   d_cat_min, d_cat_max,
        d_fitness, d_penalty, d_hard_feas, d_is_valid,
        (int)pop_sz, p);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Bajar resultados a CPU (solo escalares, no genes)
    std::vector<float>   h_fit  (pop_sz);
    std::vector<float>   h_pen  (pop_sz);
    std::vector<uint8_t> h_hf   (pop_sz);
    std::vector<uint8_t> h_valid(pop_sz);

    CUDA_CHECK(cudaMemcpy(h_fit.data(),   d_fitness,   pop_sz*sizeof(float),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pen.data(),   d_penalty,   pop_sz*sizeof(float),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_hf.data(),    d_hard_feas, pop_sz*sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_valid.data(), d_is_valid,  pop_sz*sizeof(uint8_t), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < pop_sz; ++i) {
        population[i].fitness       = h_fit  [i];
        population[i].penalty       = h_pen  [i];
        population[i].hard_feasible = (h_hf   [i] == 1);
        population[i].is_valid      = (h_valid[i] == 1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// do_reproduction
// 1. Lanza reproduce_kernel (selección + cruzamiento + mutación en GPU)
// 2. Aplica elitismo: copia los mejores k de d_population a d_offspring
// 3. Intercambia d_population ↔ d_offspring
// ─────────────────────────────────────────────────────────────────────────────
void CUDABasic::do_reproduction()
{
    size_t pop_sz = population_size;
    int blocks = ((int)pop_sz + block_size - 1) / block_size;

    // ── Paso 1: reproducción en GPU ──────────────────────────────────
    reproduce_kernel<<<blocks, block_size>>>(
        d_population, d_offspring, d_fitness,
        d_rng_states,
        (int)pop_sz, n_items,
        (int)3,   // tournament_size (fijo, igual que TournamentSelection)
        crossover_op->get_crossover_rate(),
        mutation_op->get_mutation_rate());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Paso 2: elitismo en GPU ──────────────────────────────────────
    size_t elitism_count = std::max((size_t)1,
        (size_t)(pop_sz * Config::GeneticAlgorithm::ELITISM_PERCENTAGE));

    // Encontrar índices de los mejores (en CPU, usando fitness ya bajado)
    std::vector<size_t> sorted_idx(pop_sz);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::partial_sort(sorted_idx.begin(),
                      sorted_idx.begin() + elitism_count,
                      sorted_idx.end(),
                      [this](size_t a, size_t b) {
                          return population[a].fitness > population[b].fitness;
                      });

    // Encontrar los peores índices en offspring (fitness ya no es válido
    // para offspring, así que simplemente reemplazamos los primeros k)
    // Estrategia simple: reemplazar posiciones 0..k-1 del offspring
    // (esto es equivalente al apply_elitism de BaseGA)
    for (size_t e = 0; e < elitism_count; ++e) {
        size_t src = sorted_idx[e];
        // Copiar genes del élite desde d_population a d_offspring[e]
        CUDA_CHECK(cudaMemcpy(
            d_offspring + e * (size_t)n_items,
            d_population + src * (size_t)n_items,
            (size_t)n_items * sizeof(uint8_t),
            cudaMemcpyDeviceToDevice));
    }

    // ── Paso 3: swap d_population ↔ d_offspring ──────────────────────
    int total_genes = (int)pop_sz * n_items;
    int swap_blocks = (total_genes + block_size - 1) / block_size;
    swap_populations_kernel<<<swap_blocks, block_size>>>(
        d_population, d_offspring, (int)pop_sz, n_items);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Actualizar estructura population[] en CPU con nuevos cromosomas
    // (necesario para que BaseGA::log_generation_stats funcione)
    std::vector<uint8_t> h_genes(pop_sz * (size_t)n_items);
    CUDA_CHECK(cudaMemcpy(h_genes.data(), d_population,
                          pop_sz*(size_t)n_items*sizeof(uint8_t),
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < pop_sz; ++i) {
        population[i].chromosome.resize(n_items);
        for (int g = 0; g < n_items; ++g) {
            population[i].chromosome[g] = (h_genes[i*(size_t)n_items+g] != 0);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// free_device_memory
// ─────────────────────────────────────────────────────────────────────────────
void CUDABasic::free_device_memory()
{
    auto safe_free = [](void* ptr) { if (ptr) cudaFree(ptr); };
    safe_free(d_population); safe_free(d_offspring);
    safe_free(d_fitness);    safe_free(d_penalty);
    safe_free(d_hard_feas);  safe_free(d_is_valid);
    safe_free(d_values);     safe_free(d_weights);
    safe_free(d_volumes);    safe_free(d_cat_ids);
    safe_free(d_incomp_a);   safe_free(d_incomp_b);
    safe_free(d_dep_a);      safe_free(d_dep_b);
    safe_free(d_cat_id);     safe_free(d_cat_min);
    safe_free(d_cat_max);    safe_free(d_rng_states);
}
