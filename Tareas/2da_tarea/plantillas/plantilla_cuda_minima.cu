#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        cerr << "CUDA error en " << __FILE__ << ":"          \
             << __LINE__ << " -> "                            \
             << cudaGetErrorString(err) << endl;              \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)

// TODO: implementar kernels CUDA segun el problema.

int main() {
    // 1. Reservar memoria en CPU.
    // 2. Inicializar datos.
    // 3. Ejecutar version secuencial de referencia.
    // 4. Reservar memoria en GPU.
    // 5. Copiar datos CPU -> GPU.
    // 6. Lanzar kernel CUDA.
    // 7. Copiar resultado GPU -> CPU.
    // 8. Verificar resultado CPU vs GPU.
    // 9. Medir tiempo.
    // 10. Liberar memoria.

    return 0;
}
