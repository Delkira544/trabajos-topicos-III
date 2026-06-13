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
// reproduce_kernel
// Fusión de selección por torneo + cruzamiento + mutación en un solo kernel.
// Un hilo por individuo del offspring.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void reproduce_kernel(
    const uint8_t* __restrict__ population,
    uint8_t*       offspring,
    const float*   __restrict__ fitness,
    curandState*   rng_states,
    int   pop_size,
    int   n_items,
    int   tournament_size,
    float crossover_rate,
    float mutation_rate,
    // ─── Punteros adicionales para reparación ──────
    const float*   __restrict__ values,
    const float*   __restrict__ weights,
    const float*   __restrict__ volumes,
    const int*     __restrict__ incomp_a,
    const int*     __restrict__ incomp_b,
    const int*     __restrict__ dep_a,
    const int*     __restrict__ dep_b,
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
    uint8_t*       ch = offspring   + (long long)ind   * n_items;

    // ── Cruzamiento de un punto ────────────────────────────────────────
    float r_cross = curand_uniform(&local_state);
    if (r_cross <= crossover_rate) {
        // Punto de corte aleatorio en [0, n_items-1]
        int cut = (int)(curand_uniform(&local_state) * (n_items - 1));
        // Control de divergencia: usamos operador condicional sin branch
        // para evitar divergencia de warps en el loop de genes.
        for (int g = 0; g < n_items; ++g) {
            ch[g] = (g <= cut) ? p1[g] : p2[g];
        }
    } else {
        // Sin cruzamiento: copiar padre1 directamente
        // (acceso coalescente: hilos consecutivos acceden a genes consecutivos)
        for (int g = 0; g < n_items; ++g) {
            ch[g] = p1[g];
        }
    }

    // ── Mutación uniforme (bit-flip) ──────────────────────────────────
    for (int g = 0; g < n_items; ++g) {
        float r_mut = curand_uniform(&local_state);
        // Sin branch: usamos operación aritmética
        // ch[g] = ch[g] XOR (r_mut < mutation_rate ? 1 : 0)
        ch[g] ^= (r_mut < mutation_rate) ? 1 : 0;
    }

    // ── Reparación POST-mutación ──────────────────────────────────────
    __syncthreads();  // Esperar a todos los hilos antes de reparación
    repair_chromosome_gpu(
        ch, n_items,
        values, weights, volumes,
        incomp_a, incomp_b, n_incomp,
        dep_a, dep_b, n_dep,
        max_weight, max_volume);
    __syncthreads();

    // Guardar estado cuRAND actualizado
    rng_states[ind] = local_state;
}

// ─────────────────────────────────────────────────────────────────────────────
// elitism_kernel
// Sobreescribe los peores k individuos del offspring con los mejores k de elite.
// Un hilo por gen por individuo de élite.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void elitism_kernel(
    const uint8_t* __restrict__ elite_genes,
    uint8_t*       offspring,
    const int*     __restrict__ worst_idx,
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
// Copia offspring → population para la siguiente generación.
// Accesos coalescentes: hilo i accede a byte i.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void swap_populations_kernel(
    uint8_t*       population,
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
