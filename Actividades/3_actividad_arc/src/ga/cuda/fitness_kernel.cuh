#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// ── Límite para memoria constante de ítems ───────────────────────────────────
// Cálculo: 3 arrays float + 1 array int = 4 × 4 bytes × N
// FitnessParams ocupa ~56 bytes adicionales
// Total disponible: 65536 bytes
// 65536 - 56 (FitnessParams) = 65480 / 16 bytes por ítem = 4092 → usamos 3000
// para tener margen y evitar el error ptxas
#define MAX_CONST_ITEMS 3000

// ── Arrays en memoria constante (optimized variant) ─────────────────────────
// Solo se usan en CUDAOptimized cuando n_items <= MAX_CONST_ITEMS
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
// genes layout: genes[ind * n_items + gene]  (acceso coalescente)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void fitness_kernel(
    const uint8_t* __restrict__ genes,
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
    float*         fitness,
    float*         penalty,
    uint8_t*       hard_feas,
    uint8_t*       is_valid,
    int pop_size,
    FitnessParams  p
);

// ─────────────────────────────────────────────────────────────────────────────
// Kernel optimizado: reducción intra-bloque con shared memory
// Un bloque por individuo; hilos colaboran para acumular peso/volumen/valor
// ─────────────────────────────────────────────────────────────────────────────
__global__ void fitness_kernel_opt(
    const uint8_t* __restrict__ genes,
    float*   fitness,
    float*   penalty,
    uint8_t* hard_feas,
    uint8_t* is_valid,
    int pop_size
);

// ── Helper device ─────────────────────────────────────────────────────────────
__device__ inline float clamp01(float v) {
    return v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
}
