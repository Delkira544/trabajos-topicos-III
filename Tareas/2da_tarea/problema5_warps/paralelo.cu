#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstdlib>
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

int contarMayoresSecuencial(const vector<int>& A, int N, int umbral) {
    int contador = 0;
    for (int i = 0; i < N; i++) {
        if (A[i] > umbral) contador++;
    }
    return contador;
}

__global__ void contarAtomicGlobal(int* A, int* contador, int N, int umbral) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N && A[idx] > umbral) {
        atomicAdd(contador, 1);
    }
}

__global__ void contarPorWarp(int* A, int* contador, int N, int umbral) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x & 31;

    bool condicion = (idx < N) && (A[idx] > umbral);
    unsigned int mask = __ballot_sync(0xffffffff, condicion);

    if (laneId == 0) {
        int cantidad = __popc(mask);
        if (cantidad > 0) {
            atomicAdd(contador, cantidad);
        }
    }
}

void probar(int N, int umbral) {
    cout << "\n========== N = " << N << " umbral=" << umbral << " ==========\n";

    mt19937 rng(12345);
    uniform_int_distribution<int> dist(0, 1000);

    vector<int> A(N);
    for (int i = 0; i < N; i++) {
        A[i] = dist(rng);
    }

    auto inicio = chrono::high_resolution_clock::now();
    int resultadoCPU = contarMayoresSecuencial(A, N, umbral);
    auto fin = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> tiempoCPU = fin - inicio;

    int *d_A, *d_contador;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_contador, sizeof(int)));

    cudaEvent_t start, stop, k_start, k_stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&k_start));
    CUDA_CHECK(cudaEventCreate(&k_stop));
    float tiempo_kernel_ms = 0.0f;
    float tiempo_total_ms = 0.0f;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int h_contador;

    // Warm-up
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_contador, 0, sizeof(int)));
    contarAtomicGlobal<<<blocks, threads>>>(d_A, d_contador, N, umbral);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Version atomicAdd global
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_contador, 0, sizeof(int)));
    CUDA_CHECK(cudaEventRecord(k_start));
    contarAtomicGlobal<<<blocks, threads>>>(d_A, d_contador, N, umbral);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(k_stop));
    CUDA_CHECK(cudaMemcpy(&h_contador, d_contador, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&tiempo_total_ms, start, stop));
    CUDA_CHECK(cudaEventSynchronize(k_stop));
    CUDA_CHECK(cudaEventElapsedTime(&tiempo_kernel_ms, k_start, k_stop));

    bool ok1 = (resultadoCPU == h_contador);
    cout << "[atomic global] blocks=" << blocks << " threads/block=" << threads
         << " tiempo kernel=" << tiempo_kernel_ms << " ms"
         << " tiempo total (con copias)=" << tiempo_total_ms << " ms"
         << " resultado=" << h_contador
         << " -> " << (ok1 ? "CORRECTO" : "INCORRECTO") << "\n";

    // Version ballot + popc por warp
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_contador, 0, sizeof(int)));
    CUDA_CHECK(cudaEventRecord(k_start));
    contarPorWarp<<<blocks, threads>>>(d_A, d_contador, N, umbral);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(k_stop));
    CUDA_CHECK(cudaMemcpy(&h_contador, d_contador, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&tiempo_total_ms, start, stop));
    CUDA_CHECK(cudaEventSynchronize(k_stop));
    CUDA_CHECK(cudaEventElapsedTime(&tiempo_kernel_ms, k_start, k_stop));

    bool ok2 = (resultadoCPU == h_contador);
    cout << "[warp ballot] blocks=" << blocks << " threads/block=" << threads
         << " tiempo kernel=" << tiempo_kernel_ms << " ms"
         << " tiempo total (con copias)=" << tiempo_total_ms << " ms"
         << " resultado=" << h_contador
         << " -> " << (ok2 ? "CORRECTO" : "INCORRECTO") << "\n";

    cout << "[CPU] resultado=" << resultadoCPU << " tiempo=" << tiempoCPU.count() << " ms\n";
    cout << "[NOTA] Se reportan por separado el tiempo de ejecucion puro del kernel y el tiempo total (incluye copias CPU<->GPU)\n";

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_contador));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(k_start));
    CUDA_CHECK(cudaEventDestroy(k_stop));
}

int main(int argc, char** argv) {
    int N_default = 1 << 26;
    int umbral = 500;
    if (argc > 2) umbral = atoi(argv[2]);
    cout << "--- Problema 5: Conteo condicional con warps ---\n";
    probar(1024, umbral);
    probar(100000, umbral);
    probar(1 << 20, umbral);
    int N = (argc > 1) ? atoi(argv[1]) : N_default;
    probar(N, umbral);
    return 0;
}
