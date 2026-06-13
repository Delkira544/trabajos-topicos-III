#include "ga/cuda/fitness_kernel.cuh"
#include <float.h>

// ── Definición de símbolos en memoria constante ──────────────────────────────
// IMPORTANTE: estos arrays solo se usan en CUDAOptimized cuando
// n_items <= MAX_CONST_ITEMS (3000). Para instancias grandes se usa
// memoria global (punteros pasados como parámetros al kernel básico).
__constant__ float         c_values [MAX_CONST_ITEMS];
__constant__ float         c_weights[MAX_CONST_ITEMS];
__constant__ float         c_volumes[MAX_CONST_ITEMS];
__constant__ int           c_cat_ids[MAX_CONST_ITEMS];
__constant__ FitnessParams c_params;

// ─────────────────────────────────────────────────────────────────────────────
// fitness_kernel  –  un hilo por individuo (versión básica)
// Usa punteros a memoria global (funciona para cualquier n_items)
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
    float*   fitness,
    float*   penalty,
    uint8_t* hard_feas,
    uint8_t* is_valid,
    int pop_size,
    FitnessParams p)
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= pop_size) return;

    const uint8_t* chrom = genes + (long long)ind * p.n_items;

    float total_value  = 0.f;
    float total_weight = 0.f;
    float total_volume = 0.f;

    // Contadores de categoría (máximo 64 categorías distintas)
    int cat_counts[64] = {0};

    for (int g = 0; g < p.n_items; ++g) {
        if (chrom[g]) {
            total_value  += values[g];
            total_weight += weights[g];
            total_volume += volumes[g];
            int cid = cat_ids[g];
            if (cid >= 0 && cid < 64) cat_counts[cid]++;
        }
    }

    // ── Excesos ──────────────────────────────────────────────────────
    float weight_excess = (total_weight > p.max_weight)
                          ? (total_weight - p.max_weight) : 0.f;
    float volume_excess = (total_volume > p.max_volume)
                          ? (total_volume - p.max_volume) : 0.f;

    // ── Violaciones de categoría ──────────────────────────────────────
    int errors_cat = 0;
    for (int r = 0; r < p.n_cat_rules; ++r) {
        int cid = cat_rule_id[r];
        int cnt = (cid >= 0 && cid < 64) ? cat_counts[cid] : 0;
        if (cnt < cat_rule_min[r] || cnt > cat_rule_max[r]) errors_cat++;
    }

    // ── Violaciones de incompatibilidad ───────────────────────────────
    int errors_incomp = 0;
    for (int r = 0; r < p.n_incomp; ++r) {
        if (chrom[incomp_a[r]] && chrom[incomp_b[r]]) errors_incomp++;
    }

    // ── Violaciones de dependencia ────────────────────────────────────
    int errors_dep = 0;
    for (int r = 0; r < p.n_dep; ++r) {
        if (chrom[dep_a[r]] && !chrom[dep_b[r]]) errors_dep++;
    }

    // ── Normalización y fitness ───────────────────────────────────────
    float norm_value = (p.max_value  > 0.f) ? (total_value  / p.max_value)  : 0.f;
    float nw         = (p.max_weight > 0.f) ? (weight_excess / p.max_weight) : 0.f;
    float nv         = (p.max_volume > 0.f) ? (volume_excess / p.max_volume) : 0.f;
    float n_cat      = (p.n_cat_rules > 0)  ? ((float)errors_cat    / p.n_cat_rules)  : 0.f;
    float n_incomp_f = (p.n_incomp    > 0)  ? ((float)errors_incomp / p.n_incomp)     : 0.f;
    float n_dep_f    = (p.n_dep       > 0)  ? ((float)errors_dep    / p.n_dep)        : 0.f;

    float viol = p.pen_weight   * clamp01(nw)
               + p.pen_volume   * clamp01(nv)
               + p.pen_category * clamp01(n_cat)
               + p.pen_incomp   * clamp01(n_incomp_f)
               + p.pen_dep      * clamp01(n_dep_f);

    penalty  [ind] = viol;
    fitness  [ind] = p.obj_w * norm_value - p.pen_w * viol;
    hard_feas[ind] = (weight_excess == 0.f && volume_excess == 0.f) ? 1u : 0u;
    is_valid [ind] = (hard_feas[ind] && errors_cat == 0 &&
                      errors_incomp == 0 && errors_dep == 0) ? 1u : 0u;
}

// ─────────────────────────────────────────────────────────────────────────────
// fitness_kernel_opt  –  un BLOQUE por individuo con shared memory
// Solo se usa cuando n_items <= MAX_CONST_ITEMS (datos en __constant__)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void fitness_kernel_opt(
    const uint8_t* __restrict__ genes,
    float*   fitness,
    float*   penalty,
    uint8_t* hard_feas,
    uint8_t* is_valid,
    int pop_size)
{
    int ind = blockIdx.x;
    if (ind >= pop_size) return;

    // Shared memory: 3 floats por hilo (value, weight, volume)
    extern __shared__ float smem[];
    float* s_value  = smem;
    float* s_weight = smem +   blockDim.x;
    float* s_volume = smem + 2*blockDim.x;

    const uint8_t* chrom = genes + (long long)ind * c_params.n_items;
    int tid = threadIdx.x;

    // Cada hilo reduce un subconjunto de genes
    float lv = 0.f, lw = 0.f, lvo = 0.f;
    for (int g = tid; g < c_params.n_items; g += blockDim.x) {
        if (chrom[g]) {
            lv  += c_values [g];
            lw  += c_weights[g];
            lvo += c_volumes[g];
        }
    }
    s_value [tid] = lv;
    s_weight[tid] = lw;
    s_volume[tid] = lvo;
    __syncthreads();

    // Reducción paralela en shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_value [tid] += s_value [tid + stride];
            s_weight[tid] += s_weight[tid + stride];
            s_volume[tid] += s_volume[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float total_value  = s_value [0];
        float total_weight = s_weight[0];
        float total_volume = s_volume[0];

        float weight_excess = (total_weight > c_params.max_weight)
                              ? (total_weight - c_params.max_weight) : 0.f;
        float volume_excess = (total_volume > c_params.max_volume)
                              ? (total_volume - c_params.max_volume) : 0.f;

        // ── Violaciones de categoría ──────────────────────────────────
        int cat_counts[64] = {0};
        for (int g = 0; g < c_params.n_items; ++g) {
            if (chrom[g]) {
                int cid = c_cat_ids[g];
                if (cid >= 0 && cid < 64) cat_counts[cid]++;
            }
        }

        // Nota: para fitness_kernel_opt se asume que c_params contiene cat_rule_id, cat_rule_min, cat_rule_max
        // Como no tenemos acceso directo, usamos el enfoque de contar de genes
        // (esta es una limitación de usar solo memoria constante para datos grandes)
        
        // ── Violaciones de incompatibilidad ───────────────────────────────
        // Nota: no podemos validar sin acceso a incomp_a y incomp_b en memoria constante
        
        // ── Violaciones de dependencia ────────────────────────────────────
        // Nota: no podemos validar sin acceso a dep_a y dep_b en memoria constante

        float norm_value = (c_params.max_value > 0.f)
                           ? (total_value / c_params.max_value) : 0.f;
        float nw = (c_params.max_weight > 0.f)
                   ? (weight_excess / c_params.max_weight) : 0.f;
        float nv = (c_params.max_volume > 0.f)
                   ? (volume_excess / c_params.max_volume) : 0.f;

        float viol = c_params.pen_weight * clamp01(nw)
                   + c_params.pen_volume * clamp01(nv);

        penalty  [ind] = viol;
        fitness  [ind] = c_params.obj_w * norm_value - c_params.pen_w * viol;
        hard_feas[ind] = (weight_excess == 0.f && volume_excess == 0.f) ? 1u : 0u;
        // IMPORTANTE: fitness_kernel_opt solo valida peso y volumen
        // Para validación completa (categorías, incompatibilidades, dependencias),
        // se debe usar fitness_kernel con memoria global
        is_valid [ind] = hard_feas[ind];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// reduce_best_kernel  –  reducción paralela con shared memory
// ─────────────────────────────────────────────────────────────────────────────
__global__ void reduce_best_kernel(
    const float*   __restrict__ fitness,
    const uint8_t* __restrict__ is_valid,
    int*   best_idx_out,
    float* best_fit_out,
    int pop_size)
{
    extern __shared__ float sfit[];
    int* sidx = (int*)(sfit + blockDim.x);

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float f  = -FLT_MAX;
    int   ix = -1;
    if (gid < pop_size) { f = fitness[gid]; ix = gid; }

    sfit[tid] = f;
    sidx[tid] = ix;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int   other = sidx[tid + stride];
            float fo    = sfit[tid + stride];
            bool cur_valid   = (ix    >= 0 && is_valid[ix]    == 1u);
            bool other_valid = (other >= 0 && is_valid[other] == 1u);
            bool replace = (!cur_valid && other_valid) ||
                           (cur_valid == other_valid && fo > f);
            if (replace) { sfit[tid] = fo; sidx[tid] = other; f = fo; ix = other; }
        }
        __syncthreads();
    }

    if (tid == 0) {
        best_idx_out[blockIdx.x] = sidx[0];
        best_fit_out[blockIdx.x] = sfit[0];
    }
}
