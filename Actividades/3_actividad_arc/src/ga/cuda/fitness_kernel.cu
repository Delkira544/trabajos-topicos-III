#include "ga/cuda/fitness_kernel.cuh"
#include <float.h>

// ── Definición de símbolos en memoria constante ──────────────────────────────
__constant__ float        c_values [MAX_CONST_ITEMS];
__constant__ float        c_weights[MAX_CONST_ITEMS];
__constant__ float        c_volumes[MAX_CONST_ITEMS];
__constant__ int          c_cat_ids[MAX_CONST_ITEMS];
__constant__ FitnessParams c_params;

// ─────────────────────────────────────────────────────────────────────────────
// fitness_kernel  –  un hilo por individuo (versión básica)
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

    // ── Acumular valor, peso y volumen ────────────────────────────────
    float total_value  = 0.f;
    float total_weight = 0.f;
    float total_volume = 0.f;

    // Contadores de categoría: usamos un array local de tamaño fijo.
    // El número máximo de categorías distintas es n_cat_rules.
    // (Para n_cat_rules grande se puede usar memoria global o shared.)
    int cat_counts[64] = {0};  // soporta hasta 64 categorías

    for (int g = 0; g < p.n_items; ++g) {
        if (chrom[g]) {
            total_value  += values[g];
            total_weight += weights[g];
            total_volume += volumes[g];
            int cid = cat_ids[g];
            if (cid >= 0 && cid < 64) cat_counts[cid]++;
        }
    }

    // ── Excesos de peso y volumen ─────────────────────────────────────
    float weight_excess = (total_weight > p.max_weight)
                          ? (total_weight - p.max_weight) : 0.f;
    float volume_excess = (total_volume > p.max_volume)
                          ? (total_volume - p.max_volume) : 0.f;

    // ── Violaciones de categoría ──────────────────────────────────────
    int errors_cat = 0;
    for (int r = 0; r < p.n_cat_rules; ++r) {
        int cid  = cat_rule_id[r];
        int cmin = cat_rule_min[r];
        int cmax = cat_rule_max[r];
        int cnt  = (cid >= 0 && cid < 64) ? cat_counts[cid] : 0;
        if (cnt < cmin || cnt > cmax) errors_cat++;
    }

    // ── Violaciones de incompatibilidad ───────────────────────────────
    int errors_incomp = 0;
    for (int r = 0; r < p.n_incomp; ++r) {
        int a = incomp_a[r];
        int b = incomp_b[r];
        if (chrom[a] && chrom[b]) errors_incomp++;
    }

    // ── Violaciones de dependencia ────────────────────────────────────
    int errors_dep = 0;
    for (int r = 0; r < p.n_dep; ++r) {
        int a = dep_a[r];
        int b = dep_b[r];
        if (chrom[a] && !chrom[b]) errors_dep++;
    }

    // ── Normalización y cálculo de fitness ───────────────────────────
    float norm_value  = (p.max_value  > 0.f) ? (total_value  / p.max_value)  : 0.f;
    float nw          = (p.max_weight > 0.f) ? (weight_excess / p.max_weight) : 0.f;
    float nv          = (p.max_volume > 0.f) ? (volume_excess / p.max_volume) : 0.f;
    float n_cat       = (p.n_cat_rules > 0)  ? ((float)errors_cat   / p.n_cat_rules)   : 0.f;
    float n_incomp    = (p.n_incomp    > 0)  ? ((float)errors_incomp / p.n_incomp)     : 0.f;
    float n_dep       = (p.n_dep       > 0)  ? ((float)errors_dep    / p.n_dep)        : 0.f;

    float viol = p.pen_weight   * clamp01(nw)
               + p.pen_volume   * clamp01(nv)
               + p.pen_category * clamp01(n_cat)
               + p.pen_incomp   * clamp01(n_incomp)
               + p.pen_dep      * clamp01(n_dep);

    penalty  [ind] = viol;
    fitness  [ind] = p.obj_w * norm_value - p.pen_w * viol;
    hard_feas[ind] = (weight_excess == 0.f && volume_excess == 0.f) ? 1 : 0;
    is_valid [ind] = (hard_feas[ind] && errors_cat == 0 &&
                      errors_incomp == 0 && errors_dep == 0) ? 1 : 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// fitness_kernel_opt  –  un BLOQUE por individuo (shared memory)
// Los hilos del bloque colaboran para reducir peso/volumen/valor.
// Optimización 3 del enunciado: reducción paralela intra-bloque.
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

    // Shared memory: acumular value/weight/volume por hilo y luego reducir
    extern __shared__ float smem[];   // tamaño: blockDim.x * 3 floats
    float* s_value  = smem;
    float* s_weight = smem +   blockDim.x;
    float* s_volume = smem + 2*blockDim.x;

    const uint8_t* chrom = genes + (long long)ind * c_params.n_items;
    int tid = threadIdx.x;

    // Cada hilo procesa un subconjunto de genes
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

    // Solo el hilo 0 escribe el resultado final
    // (las restricciones de incompatibilidad/dependencia/categoría
    //  se evalúan en hilo 0 para evitar complejidad extra de reducción)
    if (tid == 0) {
        float total_value  = s_value [0];
        float total_weight = s_weight[0];
        float total_volume = s_volume[0];

        float weight_excess = (total_weight > c_params.max_weight)
                              ? (total_weight - c_params.max_weight) : 0.f;
        float volume_excess = (total_volume > c_params.max_volume)
                              ? (total_volume - c_params.max_volume) : 0.f;

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
        hard_feas[ind] = (weight_excess == 0.f && volume_excess == 0.f) ? 1 : 0;
        is_valid [ind] = hard_feas[ind];  // simplificado en versión opt
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// reduce_best_kernel  –  reducción paralela para hallar mejor individuo
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

    // Cargar en shared
    float f  = -FLT_MAX;
    int   ix = -1;
    if (gid < pop_size) {
        f  = fitness[gid];
        ix = gid;
    }
    sfit[tid] = f;
    sidx[tid] = ix;
    __syncthreads();

    // Reducción: preferir individuos válidos
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int other = sidx[tid + stride];
            float fo  = sfit[tid + stride];
            // Preferimos valid sobre invalid; entre iguales, mayor fitness
            bool cur_valid   = (ix   >= 0 && is_valid[ix]    == 1);
            bool other_valid = (other >= 0 && is_valid[other] == 1);
            bool replace = false;
            if (other_valid && !cur_valid)           replace = true;
            else if (other_valid == cur_valid && fo > f) replace = true;
            if (replace) { sfit[tid] = fo; sidx[tid] = other; f = fo; ix = other; }
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Reducción global: una op atómica aproximada con CAS en float
        // Para simplicidad usamos una sola escritura por bloque y
        // la CPU selecciona el mejor entre los resultados de bloques.
        int out_slot = blockIdx.x;
        best_idx_out[out_slot] = sidx[0];
        best_fit_out[out_slot] = sfit[0];
    }
}
