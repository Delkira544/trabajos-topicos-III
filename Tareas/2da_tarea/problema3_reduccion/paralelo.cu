#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
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

float sumaSecuencial(const vector<float>& A, int N) {
    float suma = 0.0f;
    for (int i = 0; i < N; i++) {
        suma += A[i];
    }
    return suma;
}

__global__ void reduccionPorBloque(float* A, float* sumasParciales, int N) {
    __shared__ float sdata[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < N) ? A[idx] : 0.0f;
    sdata[threadIdx.x] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        sumasParciales[blockIdx.x] = sdata[0];
    }
}

void probar(int N) {
    cout << "\n========== N = " << N << " ==========\n";

    vector<float> A(N);
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
    }

    auto inicio = chrono::high_resolution_clock::now();
    float resultadoCPU = sumaSecuencial(A, N);
    auto fin = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> tiempoCPU = fin - inicio;

    float *d_A, *d_sumas;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    CUDA_CHECK(cudaMalloc(&d_sumas, blocks * sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float tiempo_ms = 0.0f;

    vector<float> sumasParciales(blocks);

    // Warm-up
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    reduccionPorBloque<<<blocks, threads>>>(d_A, d_sumas, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    reduccionPorBloque<<<blocks, threads>>>(d_A, d_sumas, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&tiempo_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(sumasParciales.data(), d_sumas, blocks * sizeof(float), cudaMemcpyDeviceToHost));

    float resultadoGPU = 0.0f;
    for (int i = 0; i < blocks; i++) {
        resultadoGPU += sumasParciales[i];
    }

    float diff = fabs(resultadoCPU - resultadoGPU);
    float maxVal = fmax(fabs(resultadoCPU), fabs(resultadoGPU));
    bool ok = diff <= fmax(1e-5f * maxVal, 1e-2f);

    cout << "[reduccion] blocks=" << blocks << " threads/block=" << threads
         << " tiempo=" << tiempo_ms << " ms"
         << " suma_cpu=" << resultadoCPU << " suma_gpu=" << resultadoGPU
         << " diff=" << diff
         << " -> " << (ok ? "CORRECTO" : "INCORRECTO") << "\n";

    cout << "[CPU] tiempo=" << tiempoCPU.count() << " ms resultado=" << resultadoCPU << "\n";
    cout << "[NOTA] Tiempo GPU INCLUYE copias de memoria CPU<->GPU\n";

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_sumas));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char** argv) {
    int N_default = 1 << 24;
    cout << "--- Problema 3: Reduccion paralela (suma de elementos) ---\n";
    probar(1024);
    probar(100000);
    probar(1 << 20);
    int N = (argc > 1) ? atoi(argv[1]) : N_default;
    probar(N);
    return 0;
}
