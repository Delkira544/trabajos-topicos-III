#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

/**
 * @brief Kernels de reducción paralela para encontrar el mejor individuo.
 *
 * Se implementan dos estrategias:
 *
 * 1. reduce_best_block: cada bloque reduce su segmento y escribe
 *    un resultado parcial. La CPU toma el mínimo entre bloques.
 *    → Usada en CUDABasic.
 *
 * 2. reduce_best_warp: reducción completa usando instrucciones
 *    warp-shuffle (__shfl_down_sync), sin shared memory explícita.
 *    → Usada en CUDAOptimized (menor latencia).
 *
 * En ambos casos se prioriza: válido > no válido; entre iguales, mayor fitness.
 */

// ─────────────────────────────────────────────────────────────────────────────
// Resultado de reducción (un elemento por bloque)
// ─────────────────────────────────────────────────────────────────────────────
struct ReduceResult {
    float best_fitness;
    int   best_idx;
};

// ─────────────────────────────────────────────────────────────────────────────
// reduce_best_block  (versión básica con shared memory)
// Lanzar con N bloques; cada bloque recibe [pop_size/N] elementos.
// results_out debe tener tamaño gridDim.x.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void reduce_best_block(
    const float*   __restrict__ fitness,
    const uint8_t* __restrict__ is_valid,
    ReduceResult*  results_out,   // [gridDim.x]
    int pop_size
);

// ─────────────────────────────────────────────────────────────────────────────
// reduce_best_warp  (versión optimizada con warp shuffle)
// Un solo bloque de 1024 hilos reduce toda la población en una pasada.
// Requiere pop_size <= 1024 o múltiples bloques con resultado parcial.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void reduce_best_warp(
    const float*   __restrict__ fitness,
    const uint8_t* __restrict__ is_valid,
    ReduceResult*  result_out,    // [1]
    int pop_size
);

// ─────────────────────────────────────────────────────────────────────────────
// CPU helper: finaliza la reducción entre resultados de bloques
// ─────────────────────────────────────────────────────────────────────────────
ReduceResult cpu_reduce_best(const ReduceResult* partial, int n_blocks);
