#include "ga/cuda/operators_kernel.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// init_rng_kernel
// Cada hilo inicializa su estado cuRAND con (seed + ind) como offset,
// garantizando secuencias independientes entre hilos.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void init_rng_kernel(
    curandState* states,
    unsigned long long seed,
    int pop_size)
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= pop_size) return;

    // sequence=ind garantiza que cada hilo tenga secuencia independiente
    curand_init(seed, (unsigned long long)ind, 0, &states[ind]);
}

// ─────────────────────────────────────────────────────────────────────────────
// repair_chromosome_gpu 
// Repara incompatibilidades, dependencias y capacidades
// ─────────────────────────────────────────────────────────────────────────────
__device__ void repair_chromosome_gpu(
    uint8_t* child_genes,
    const float* weights,
    const float* volumes,
    const float* values,
    const int* incomp_a,
    const int* incomp_b,
    const int* dep_a,
    const int* dep_b,
    int n_items,
    int num_incomp,
    int num_dep,
    float max_weight,
    float max_volume)
{
    bool changed = true;
    int max_iters = 100;
    int iter = 0;

    while (changed && iter < max_iters)
    {
        changed = false;
        iter++;
        bool is_valid = true;

        // FASE 1: Incompatibilidades
        for (int i = 0; i < num_incomp; ++i) {
            int u = incomp_a[i];
            int v = incomp_b[i];
            if (child_genes[u] && child_genes[v]) {
                float cost_u = weights[u] + volumes[u];
                float eff_u = (cost_u > 0) ? values[u] / cost_u : 0.0f;
                float cost_v = weights[v] + volumes[v];
                float eff_v = (cost_v > 0) ? values[v] / cost_v : 0.0f;
                
                if (eff_u < eff_v) child_genes[u] = 0;
                else child_genes[v] = 0;
                
                changed = true;
                is_valid = false;
            }
        }

        // FASE 2: Dependencias
        for (int i = 0; i < num_dep; ++i) {
            int u = dep_a[i];
            int v = dep_b[i];
            if (child_genes[u] && !child_genes[v]) {
                child_genes[u] = 0;
                changed = true;
                is_valid = false;
            }
        }

        // FASE 3: Capacidades (Peso y Volumen)
        float current_w = 0.0f, current_v = 0.0f;
        for (int i = 0; i < n_items; ++i) {
            if (child_genes[i]) {
                current_w += weights[i];
                current_v += volumes[i];
            }
        }

        if (current_w > max_weight || current_v > max_volume) {
            is_valid = false;
            int worst_idx = -1;
            float worst_eff = -1.0f;  // Inicializar a -1 (mejor que cualquier eficiencia)

            for (int i = 0; i < n_items; ++i) {
                if (child_genes[i]) {
                    float cost = weights[i] + volumes[i];
                    float eff = (cost > 0) ? values[i] / cost : 0.0f;
                    if (worst_idx == -1 || eff < worst_eff) {
                        worst_eff = eff;
                        worst_idx = i;
                    }
                }
            }

            // Solo remover si encontramos un item válido
            if (worst_idx >= 0) {
                child_genes[worst_idx] = 0;
            }

            if (worst_idx >= 0) {
                child_genes[worst_idx] = 0;
                changed = true;
            } else {
                break;
            }
        }

        if (is_valid) break;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// reproduce_kernel
// Fusión de selección por torneo + cruzamiento + mutación en un solo kernel.
// Un hilo por individuo del offspring.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void reproduce_kernel(
    const uint8_t* __restrict__ population,
    uint8_t* offspring,
    const float* __restrict__ fitness,
    curandState* rng_states,
    int   pop_size,
    int   n_items,
    int   tournament_size,
    float crossover_rate,
    float mutation_rate,
    // ─── Punteros adicionales para reparación ──────
    const float* __restrict__ values,
    const float* __restrict__ weights,
    const float* __restrict__ volumes,
    const int* __restrict__ incomp_a,
    const int* __restrict__ incomp_b,
    const int* __restrict__ dep_a,
    const int* __restrict__ dep_b,
    int   max_weight,
    int   max_volume,
    int   n_incomp,
    int   n_dep)
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= pop_size) return;

    // Cargar estado cuRAND local (en registros)
    curandState local_state = rng_states[ind];

    // ── Selección por torneo – padre 1 ────────────────────────────────
    int best1 = (int)(curand_uniform(&local_state) * pop_size) % pop_size;
    for (int t = 1; t < tournament_size; ++t) {
        int cand = (int)(curand_uniform(&local_state) * pop_size) % pop_size;
        if (fitness[cand] > fitness[best1]) best1 = cand;
    }

    // ── Selección por torneo – padre 2 ────────────────────────────────
    int best2 = (int)(curand_uniform(&local_state) * pop_size) % pop_size;
    for (int t = 1; t < tournament_size; ++t) {
        int cand = (int)(curand_uniform(&local_state) * pop_size) % pop_size;
        if (fitness[cand] > fitness[best2]) best2 = cand;
    }

    const uint8_t* p1 = population + (long long)best1 * n_items;
    const uint8_t* p2 = population + (long long)best2 * n_items;
    uint8_t* ch = offspring   + (long long)ind   * n_items;

    // ── Cruzamiento de un punto ────────────────────────────────────────
    float r_cross = curand_uniform(&local_state);
    if (r_cross <= crossover_rate) {
        int cut = (int)(curand_uniform(&local_state) * (n_items - 1));
        for (int g = 0; g < n_items; ++g) {
            ch[g] = (g <= cut) ? p1[g] : p2[g];
        }
    } else {
        for (int g = 0; g < n_items; ++g) {
            ch[g] = p1[g];
        }
    }

    // ── Mutación uniforme (bit-flip) ──────────────────────────────────
    for (int g = 0; g < n_items; ++g) {
        float r_mut = curand_uniform(&local_state);
        ch[g] ^= (r_mut < mutation_rate) ? 1 : 0;
    }

    // ── Reparación POST-mutación ──────────────────────────────────────
    repair_chromosome_gpu(
        ch, 
        weights, volumes, values, 
        incomp_a, incomp_b, dep_a, dep_b, 
        n_items, n_incomp, n_dep, 
        (float)max_weight, (float)max_volume
    );

    // Guardar estado cuRAND actualizado
    rng_states[ind] = local_state;
}

// ─────────────────────────────────────────────────────────────────────────────
// elitism_kernel
// ─────────────────────────────────────────────────────────────────────────────
__global__ void elitism_kernel(
    const uint8_t* __restrict__ elite_genes,
    uint8_t* offspring,
    const int* __restrict__ worst_idx,
    int k,
    int n_items)
{
    int elite_ind = blockIdx.x;   // qué individuo de élite
    int gene      = threadIdx.x + blockIdx.y * blockDim.x;  // qué gen

    if (elite_ind >= k || gene >= n_items) return;

    int dst_ind = worst_idx[elite_ind];
    offspring[(long long)dst_ind * n_items + gene] =
        elite_genes[(long long)elite_ind * n_items + gene];
}

// ─────────────────────────────────────────────────────────────────────────────
// swap_populations_kernel
// ─────────────────────────────────────────────────────────────────────────────
__global__ void swap_populations_kernel(
    uint8_t* population,
    const uint8_t* __restrict__ offspring,
    int pop_size,
    int n_items)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = pop_size * n_items;
    if (idx < total) {
        population[idx] = offspring[idx];
    }
}