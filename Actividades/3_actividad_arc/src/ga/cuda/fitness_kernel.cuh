#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// ── Límite para memoria constante de ítems (máx 64 KB) ──────────────────────
#define MAX_CONST_ITEMS 4096

// ── Arrays en memoria constante (optimized variant) ─────────────────────────
extern __constant__ float c_values [MAX_CONST_ITEMS];
extern __constant__ float c_weights[MAX_CONST_ITEMS];
extern __constant__ float c_volumes[MAX_CONST_ITEMS];
extern __constant__ int   c_cat_ids[MAX_CONST_ITEMS];

// ── Parámetros escalares en memoria constante ────────────────────────────────
struct FitnessParams {
    float max_weight;
    float max_volume;
    float max_value;
    float pen_weight;
    float pen_volume;
    float pen_category;
    float pen_incomp;
    float pen_dep;
    float obj_w;
    float pen_w;
    int   n_items;
    int   n_incomp;
    int   n_dep;
    int   n_cat_rules;
};
extern __constant__ FitnessParams c_params;

// ─────────────────────────────────────────────────────────────────────────────
// Kernel básico: un hilo por individuo
// genes layout: genes[ind * n_items + gene]
// ─────────────────────────────────────────────────────────────────────────────
__global__ void fitness_kernel(
    const uint8_t* __restrict__ genes,    // [pop_size * n_items]
    const float*   __restrict__ values,
    const float*   __restrict__ weights,
    const float*   __restrict__ volumes,
    const int*     __restrict__ cat_ids,
    const int*     __restrict__ incomp_a,
    const int*     __restrict__ incomp_b,
    const int*     __restrict__ dep_a,
    const int*     __restrict__ dep_b,
    const int*     __restrict__ cat_rule_id,
    const int*     __restrict__ cat_rule_min,
    const int*     __restrict__ cat_rule_max,
    float*         fitness,               // [pop_size]  output
    float*         penalty,              // [pop_size]  output
    uint8_t*       hard_feas,            // [pop_size]  output
    uint8_t*       is_valid,             // [pop_size]  output
    int pop_size,
    FitnessParams  p
);

// ─────────────────────────────────────────────────────────────────────────────
// Kernel optimizado: reducción intra-bloque con shared memory
// Un bloque por individuo; los hilos del bloque colaboran para
// acumular peso, volumen y valor del individuo.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void fitness_kernel_opt(
    const uint8_t* __restrict__ genes,
    float*   fitness,
    float*   penalty,
    uint8_t* hard_feas,
    uint8_t* is_valid,
    int pop_size
    // usa c_values, c_weights, c_volumes, c_cat_ids, c_params
);

// ─────────────────────────────────────────────────────────────────────────────
// Kernel de reducción: encuentra índice del mejor individuo
// ─────────────────────────────────────────────────────────────────────────────
__global__ void reduce_best_kernel(
    const float*   __restrict__ fitness,
    const uint8_t* __restrict__ is_valid,
    int*   best_idx_out,   // [1]
    float* best_fit_out,   // [1]
    int pop_size
);

// ─────────────────────────────────────────────────────────────────────────────
// Helpers en device (inline, usados por ambos kernels)
// ─────────────────────────────────────────────────────────────────────────────
__device__ inline float clamp01(float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); }
