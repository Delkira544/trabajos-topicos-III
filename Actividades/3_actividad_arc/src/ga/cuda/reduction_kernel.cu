#include "ga/cuda/reduction_kernel.cuh"
#include <float.h>

// ─────────────────────────────────────────────────────────────────────────────
// Función device auxiliar: ¿A es mejor que B?
// Prioridad: válido > inválido; entre iguales, mayor fitness.
// ─────────────────────────────────────────────────────────────────────────────
__device__ inline bool is_better(
    float fa, int ia, uint8_t va,
    float fb, int ib, uint8_t vb)
{
    if (va && !vb) return true;
    if (!va && vb) return false;
    return fa > fb;
}

// ─────────────────────────────────────────────────────────────────────────────
// reduce_best_block  –  reducción con shared memory (versión básica)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void reduce_best_block(
    const float*   __restrict__ fitness,
    const uint8_t* __restrict__ is_valid,
    ReduceResult*  results_out,
    int pop_size)
{
    extern __shared__ char smem_raw[];
    float*  s_fit   = (float*)smem_raw;
    int*    s_idx   = (int*)  (s_fit + blockDim.x);
    uint8_t* s_valid = (uint8_t*)(s_idx + blockDim.x);

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Cargar elemento o valor neutro
    if (gid < pop_size) {
        s_fit  [tid] = fitness [gid];
        s_idx  [tid] = gid;
        s_valid[tid] = is_valid[gid];
    } else {
        s_fit  [tid] = -FLT_MAX;
        s_idx  [tid] = -1;
        s_valid[tid] = 0;
    }
    __syncthreads();

    // Reducción en shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int    ib = s_idx  [tid + stride];
            float  fb = s_fit  [tid + stride];
            uint8_t vb = s_valid[tid + stride];
            if (ib >= 0 && is_better(fb, ib, vb,
                                     s_fit[tid], s_idx[tid], s_valid[tid])) {
                s_fit  [tid] = fb;
                s_idx  [tid] = ib;
                s_valid[tid] = vb;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        results_out[blockIdx.x].best_fitness = s_fit[0];
        results_out[blockIdx.x].best_idx     = s_idx[0];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// reduce_best_warp  –  reducción optimizada con warp shuffle
// Evita shared memory para la etapa de warp → menor latencia.
//
// CORRECCIÓN 2: Implementa grid-stride loop para manejar pop_size > 1024.
// Usa blockIdx.x * blockDim.x + threadIdx.x para iterar sobre todos los 
// elementos, acumulando el mejor en cada warp. Después usa shared memory 
// para reducción entre warps.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void reduce_best_warp(
    const float*   __restrict__ fitness,
    const uint8_t* __restrict__ is_valid,
    ReduceResult*  result_out,
    int pop_size)
{
    // Inicializar con valores neutros
    float   best_fit  = -FLT_MAX;
    int     best_idx  = -1;
    uint8_t best_valid = 0;

    // GRID-STRIDE LOOP: procesar todos los elementos de la población
    // aunque sea > blockDim.x * gridDim.x
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = gid; i < pop_size; i += stride) {
        float   f  = fitness [i];
        uint8_t v  = is_valid[i];
        if (is_better(f, i, v, best_fit, best_idx, best_valid)) {
            best_fit   = f;
            best_idx   = i;
            best_valid = v;
        }
    }

    // Fase 1: Reducción dentro del warp (32 hilos) con shuffle
    int lane = threadIdx.x & 31;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float   fo  = __shfl_down_sync(0xffffffff, best_fit,  offset);
        int     ixo = __shfl_down_sync(0xffffffff, best_idx, offset);
        uint8_t vo  = (uint8_t)__shfl_down_sync(0xffffffff, (int)best_valid, offset);
        if (ixo >= 0 && is_better(fo, ixo, vo, best_fit, best_idx, best_valid)) {
            best_fit   = fo;
            best_idx   = ixo;
            best_valid = vo;
        }
    }

    // Fase 2: Reducción entre warps usando shared memory
    // Solo los lane-0 de cada warp participan
    __shared__ float   warp_fit  [32];
    __shared__ int     warp_idx  [32];
    __shared__ uint8_t warp_valid[32];

    int warp_id = threadIdx.x >> 5;

    if (lane == 0) {
        warp_fit  [warp_id] = best_fit;
        warp_idx  [warp_id] = best_idx;
        warp_valid[warp_id] = best_valid;
    }
    __syncthreads();

    // Solo primer warp finaliza la reducción
    int n_warps = (blockDim.x + 31) / 32;
    if (warp_id == 0 && lane < n_warps) {
        float   f  = warp_fit  [lane];
        int     ix = warp_idx  [lane];
        uint8_t v  = warp_valid[lane];

        for (int offset = 16; offset > 0; offset >>= 1) {
            float   fo  = __shfl_down_sync(0xffffffff, f,  offset);
            int     ixo = __shfl_down_sync(0xffffffff, ix, offset);
            uint8_t vo  = (uint8_t)__shfl_down_sync(0xffffffff, (int)v, offset);
            if (ixo >= 0 && is_better(fo, ixo, vo, f, ix, v)) {
                f   = fo;
                ix  = ixo;
                v   = vo;
            }
        }

        if (lane == 0) {
            result_out[blockIdx.x].best_fitness = f;
            result_out[blockIdx.x].best_idx     = ix;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// cpu_reduce_best  –  finaliza la reducción entre bloques en CPU
// ─────────────────────────────────────────────────────────────────────────────
ReduceResult cpu_reduce_best(const ReduceResult* partial, int n_blocks)
{
    ReduceResult best = {-FLT_MAX, -1};
    for (int i = 0; i < n_blocks; ++i) {
        if (partial[i].best_idx >= 0 &&
            partial[i].best_fitness > best.best_fitness) {
            best = partial[i];
        }
    }
    return best;
}
