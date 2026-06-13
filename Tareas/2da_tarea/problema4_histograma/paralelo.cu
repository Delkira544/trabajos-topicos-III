#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

const int NUM_BINS = 256;

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

void histogramaSecuencial(const vector<unsigned char>& A,
                          vector<unsigned int>& hist, int N) {
    for (int i = 0; i < NUM_BINS; i++) hist[i] = 0;
    for (int i = 0; i < N; i++) {
        hist[A[i]]++;
    }
}

__global__ void histogramaGlobal(unsigned char* A, unsigned int* hist, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(&hist[A[idx]], 1);
    }
}

__global__ void histogramaShared(unsigned char* A, unsigned int* hist, int N) {
    __shared__ unsigned int localHist[256];

    // Inicializacion paralela del histograma local (robusta para cualquier blockDim.x)
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(&localHist[A[idx]], 1);
    }
    __syncthreads();

    // Combinacion final (robusta para cualquier blockDim.x)
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        atomicAdd(&hist[i], localHist[i]);
    }
}

bool verificarHistograma(const vector<unsigned int>& hist,
                         const vector<unsigned int>& referencia) {
    for (int i = 0; i < NUM_BINS; i++) {
        if (hist[i] != referencia[i]) {
            cout << "Error en bin " << i << endl;
            cout << "Resultado: " << hist[i] << " Referencia: " << referencia[i] << endl;
            return false;
        }
    }
    return true;
}

unsigned long long sumaBins(const vector<unsigned int>& hist) {
    unsigned long long suma = 0;
    for (unsigned int v : hist) {
        suma += v;
    }
    return suma;
}

void probar(int N) {
    cout << "\n========== N = " << N << " ==========\n";

    mt19937 rng(12345);
    uniform_int_distribution<int> dist(0, NUM_BINS - 1);

    vector<unsigned char> A(N);
    vector<unsigned int> referencia(NUM_BINS, 0);
    vector<unsigned int> hist_global(NUM_BINS, 0);
    vector<unsigned int> hist_shared(NUM_BINS, 0);

    for (int i = 0; i < N; i++) {
        A[i] = static_cast<unsigned char>(dist(rng));
    }

    auto inicio = chrono::high_resolution_clock::now();
    histogramaSecuencial(A, referencia, N);
    auto fin = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> tiempoCPU = fin - inicio;

    unsigned char* d_A;
    unsigned int* d_hist;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_hist, NUM_BINS * sizeof(unsigned int)));

    cudaEvent_t start, stop, k_start, k_stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&k_start));
    CUDA_CHECK(cudaEventCreate(&k_stop));
    float tiempo_kernel_ms = 0.0f;
    float tiempo_total_ms = 0.0f;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Warm-up
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), N * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
    histogramaGlobal<<<blocks, threads>>>(d_A, d_hist, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Version atomicAdd global
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), N * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaEventRecord(k_start));
    histogramaGlobal<<<blocks, threads>>>(d_A, d_hist, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(k_stop));
    CUDA_CHECK(cudaMemcpy(hist_global.data(), d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&tiempo_total_ms, start, stop));
    CUDA_CHECK(cudaEventSynchronize(k_stop));
    CUDA_CHECK(cudaEventElapsedTime(&tiempo_kernel_ms, k_start, k_stop));

    bool ok1 = verificarHistograma(hist_global, referencia);
    cout << "[global atomic] blocks=" << blocks << " threads/block=" << threads
         << " tiempo kernel=" << tiempo_kernel_ms << " ms"
         << " tiempo total (con copias)=" << tiempo_total_ms << " ms"
         << " suma=" << sumaBins(hist_global)
         << " -> " << (ok1 ? "CORRECTO" : "INCORRECTO") << "\n";

    // Version shared memory
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), N * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaEventRecord(k_start));
    histogramaShared<<<blocks, threads>>>(d_A, d_hist, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(k_stop));
    CUDA_CHECK(cudaMemcpy(hist_shared.data(), d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&tiempo_total_ms, start, stop));
    CUDA_CHECK(cudaEventSynchronize(k_stop));
    CUDA_CHECK(cudaEventElapsedTime(&tiempo_kernel_ms, k_start, k_stop));

    bool ok2 = verificarHistograma(hist_shared, referencia);
    cout << "[shared atomic] blocks=" << blocks << " threads/block=" << threads
         << " tiempo kernel=" << tiempo_kernel_ms << " ms"
         << " tiempo total (con copias)=" << tiempo_total_ms << " ms"
         << " suma=" << sumaBins(hist_shared)
         << " -> " << (ok2 ? "CORRECTO" : "INCORRECTO") << "\n";

    cout << "[CPU] tiempo=" << tiempoCPU.count() << " ms suma=" << sumaBins(referencia) << "\n";
    cout << "[NOTA] Se reportan por separado el tiempo de ejecucion puro del kernel y el tiempo total (incluye copias CPU<->GPU)\n";

    cout << "Primeros 10 bins:" << endl;
    for (int i = 0; i < 10; i++) {
        cout << "hist[" << i << "] = " << referencia[i] << endl;
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_hist));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char** argv) {
    int N_default = 1 << 26;
    cout << "--- Problema 4: Histograma de valores 0-255 ---\n";
    probar(1024);
    probar(100000);
    probar(1 << 20);
    int N = (argc > 1) ? atoi(argv[1]) : N_default;
    probar(N);
    return 0;
}
