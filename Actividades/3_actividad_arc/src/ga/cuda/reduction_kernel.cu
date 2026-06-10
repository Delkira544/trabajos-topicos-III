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
// ─────────────────────────────────────────────────────────────────────────────
__global__ void reduce_best_warp(
    const float*   __restrict__ fitness,
    const uint8_t* __restrict__ is_valid,
    ReduceResult*  result_out,
    int pop_size)
{
    // Fase 1: cada hilo carga su elemento
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float   f  = (gid < pop_size) ? fitness [gid] : -FLT_MAX;
    int     ix = (gid < pop_size) ? gid            : -1;
    uint8_t v  = (gid < pop_size) ? is_valid[gid]  : 0;

    // Fase 2: reducción dentro del warp (32 hilos) con shuffle
    // No necesita __syncthreads() dentro del warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        float   fo  = __shfl_down_sync(0xffffffff, f,  offset);
        int     ixo = __shfl_down_sync(0xffffffff, ix, offset);
        uint8_t vo  = (uint8_t)__shfl_down_sync(0xffffffff, (int)v, offset);
        if (ixo >= 0 && is_better(fo, ixo, vo, f, ix, v)) {
            f = fo; ix = ixo; v = vo;
        }
    }

    // Fase 3: reducción entre warps usando shared memory
    // Solo los lane-0 de cada warp participan
    __shared__ float   warp_fit  [32];
    __shared__ int     warp_idx  [32];
    __shared__ uint8_t warp_valid[32];

    int lane    = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    if (lane == 0) {
        warp_fit  [warp_id] = f;
        warp_idx  [warp_id] = ix;
        warp_valid[warp_id] = v;
    }
    __syncthreads();

    // Solo primer warp finaliza la reducción
    int n_warps = (blockDim.x + 31) / 32;
    if (warp_id == 0 && lane < n_warps) {
        f  = warp_fit  [lane];
        ix = warp_idx  [lane];
        v  = warp_valid[lane];

        for (int offset = 16; offset > 0; offset >>= 1) {
            float   fo  = __shfl_down_sync(0xffffffff, f,  offset);
            int     ixo = __shfl_down_sync(0xffffffff, ix, offset);
            uint8_t vo  = (uint8_t)__shfl_down_sync(0xffffffff, (int)v, offset);
            if (ixo >= 0 && is_better(fo, ixo, vo, f, ix, v)) {
                f = fo; ix = ixo; v = vo;
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
