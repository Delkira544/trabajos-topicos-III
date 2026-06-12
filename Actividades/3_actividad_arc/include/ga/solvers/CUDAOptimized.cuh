#pragma once
#include "ga/solvers/CUDABasic.cuh"

/**
 * @brief Solver CUDA Optimizado del Algoritmo Genético
 *
 * Extiende CUDABasic añadiendo al menos 5 optimizaciones medibles:
 *
 *  1. Memoria constante (__constant__) para parámetros de penalización
 *     y capacidades máximas (solo lectura, broadcast eficiente).
 *
 *  2. Accesos coalescentes garantizados mediante la representación
 *     genes[ind * n_items + gene] ya presente en CUDABasic.
 *
 *  3. Reducción paralela con shared memory para encontrar el mejor
 *     individuo (reduce_best_kernel_opt) y para acumular peso/volumen
 *     por individuo (fitness con reducción intra-bloque).
 *
 *  4. Ajuste dinámico del tamaño de bloque: el constructor acepta
 *     block_size y expone get_optimal_block_size() para experimentar.
 *
 *  5. Uso de memoria constante para los arrays de ítems (valores,
 *     pesos, volúmenes) cuando n_items <= MAX_CONST_ITEMS (64 KB).
 *
 *  6. Control de divergencia de warps: en los kernels de selección y
 *     mutación se minimizan los branches dependientes del hilo.
 *
 *  7. Streams CUDA: la evaluación de fitness y la preparación de la
 *     siguiente generación se solapan usando dos streams.
 *
 * Las optimizaciones que no generan mejora real se documentan en el
 * informe con sus mediciones.
 */
class CUDAOptimized : public CUDABasic
{
protected:
    // ── Streams para overlap compute/memory ─────────────────────────
    cudaStream_t stream_eval;   ///< Stream para evaluación de fitness
    cudaStream_t stream_repro;  ///< Stream para reproducción

    // ── Flags de optimización (útiles para experimentar) ────────────
    bool use_const_memory;   ///< Usar __constant__ para ítems
    bool use_streams;        ///< Usar streams para solapamiento
    bool use_shared_reduce;  ///< Usar shared memory en reducción

    // ── Override de métodos que cambian la estrategia ────────────────

    /** Evaluación usando reducción intra-bloque con shared memory */
    void evaluate_population() override;

    /** Reproducción con streams y menor divergencia de warps */
    void do_reproduction() override;

    /** Sube datos de instancia a memoria constante cuando es posible */
    void upload_instance_optimized();

public:
    CUDAOptimized(KnapsackInstance& inst,
                  std::unique_ptr<ga::operators::Crossover>           cross,
                  std::unique_ptr<ga::operators::Mutation>            mut,
                  std::unique_ptr<ga::operators::Selection>           sel,
                  std::unique_ptr<ga::operators::FitnessEvaluator>    fit,
                  std::unique_ptr<ga::operators::ConstraintValidator> val,
                  bool verbose        = false,
                  int  seed           = 0,
                  int  block_sz       = 128,
                  bool const_mem      = true,
                  bool streams        = true,
                  bool shared_reduce  = true,
                  size_t pop_size     = 0,
                  size_t num_gens     = 0);

    ~CUDAOptimized() override;

    /** Devuelve el tamaño de bloque óptimo según ocupancia del kernel */
    static int get_optimal_block_size(int n_items);
};
