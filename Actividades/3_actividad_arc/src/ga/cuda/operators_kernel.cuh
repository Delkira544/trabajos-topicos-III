#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>

// ─────────────────────────────────────────────────────────────────────────────
// Kernel de inicialización de estados cuRAND
// Un hilo por individuo; cada hilo tiene su propio estado independiente.
// Esto garantiza aleatoriedad reproducible y sin condiciones de carrera.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void init_rng_kernel(
    curandState* states,
    unsigned long long seed,
    int pop_size
);

// ─────────────────────────────────────────────────────────────────────────────
// Kernel de selección por torneo + cruzamiento + mutación
// Se fusionan los tres operadores en un solo kernel para:
//   - Reducir lanzamientos de kernel
//   - Mantener datos calientes en registros entre etapas
//
// Para cada individuo i en offspring:
//   1. Torneo entre tournament_size candidatos aleatorios → padre1
//   2. Torneo entre tournament_size candidatos aleatorios → padre2
//   3. Cruzamiento de un punto (si rand < crossover_rate)
//   4. Mutación uniforme bit-flip (por gen, si rand < mutation_rate)
//   5. Reparación de restricciones (incompatibilidades, dependencias, capacidades)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void reproduce_kernel(
    const uint8_t* __restrict__ population,  // [pop_size * n_items] entrada
    uint8_t*       offspring,                // [pop_size * n_items] salida
    const float*   __restrict__ fitness,     // [pop_size]
    curandState*   rng_states,               // [pop_size] un estado por hilo
    int   pop_size,
    int   n_items,
    int   tournament_size,
    float crossover_rate,
    float mutation_rate,
    // ─── Punteros adicionales para reparación ──────
    const float*   __restrict__ values,      // [n_items]
    const float*   __restrict__ weights,     // [n_items]
    const float*   __restrict__ volumes,     // [n_items]
    const int*     __restrict__ incomp_a,    // [n_incomp]
    const int*     __restrict__ incomp_b,    // [n_incomp]
    const int*     __restrict__ dep_a,       // [n_dep]
    const int*     __restrict__ dep_b,       // [n_dep]
    int   max_weight,
    int   max_volume,
    int   n_incomp,
    int   n_dep
);

// ─────────────────────────────────────────────────────────────────────────────
// Kernel de elitismo
// Copia los top-k individuos de 'population' (ya ordenados por fitness)
// sobre los peores k de 'offspring'.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void elitism_kernel(
    const uint8_t* __restrict__ elite_genes,  // [k * n_items]
    uint8_t*       offspring,                 // [pop_size * n_items]
    const int*     __restrict__ worst_idx,    // [k] índices en offspring
    int   k,
    int   n_items
);

// ─────────────────────────────────────────────────────────────────────────────
// Kernel auxiliar: copia offspring → population (swap de generación)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void swap_populations_kernel(
    uint8_t*       population,
    const uint8_t* __restrict__ offspring,
    int pop_size,
    int n_items
);
