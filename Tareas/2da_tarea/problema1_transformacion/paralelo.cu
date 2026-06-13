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

void transformarSecuencial(const vector<float>& A, vector<float>& B, int N) {
    for (int i = 0; i < N; i++) {
        B[i] = 3.0f * A[i] + 7.0f;
    }
}

__global__ void transformarCUDA(float* A, float* B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        B[idx] = 3.0f * A[idx] + 7.0f;
    }
}

__global__ void transformarGridStride(float* A, float* B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    while (idx < N) {
        B[idx] = 3.0f * A[idx] + 7.0f;
        idx += stride;
    }
}

bool verificar(const vector<float>& B, const vector<float>& referencia, int N) {
    for (int i = 0; i < N; i++) {
        if (fabs(B[i] - referencia[i]) > 1e-5f) {
            cout << "Error en posicion " << i << endl;
            cout << "Resultado: " << B[i] << " Referencia: " << referencia[i] << endl;
            return false;
        }
    }
    return true;
}

double checksum(const vector<float>& V) {
    double s = 0.0;
    for (float x : V) {
        s += static_cast<double>(x);
    }
    return s;
}

void probar(int N, int threads = 256) {
    cout << "\n========== N = " << N << " | threads/block = " << threads << " ==========\n";

    vector<float> A(N);
    vector<float> B_cpu(N);
    vector<float> referencia(N);
    vector<float> B_cuda(N);
    vector<float> B_gs(N);

    for (int i = 0; i < N; i++) {
        A[i] = static_cast<float>(i % 100);
    }

    auto inicio = chrono::high_resolution_clock::now();
    transformarSecuencial(A, referencia, N);
    auto fin = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> tiempoCPU = fin - inicio;

    float *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));

    cudaEvent_t start, stop, k_start, k_stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&k_start));
    CUDA_CHECK(cudaEventCreate(&k_stop));
    float tiempo_kernel_ms = 0.0f;
    float tiempo_total_ms = 0.0f;

    int blocks = (N + threads - 1) / threads;

    // Warm-up
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    transformarCUDA<<<blocks, threads>>>(d_A, d_B, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Version 1 hilo por elemento
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(k_start));
    transformarCUDA<<<blocks, threads>>>(d_A, d_B, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(k_stop));
    CUDA_CHECK(cudaMemcpy(B_cuda.data(), d_B, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&tiempo_total_ms, start, stop));
    CUDA_CHECK(cudaEventSynchronize(k_stop));
    CUDA_CHECK(cudaEventElapsedTime(&tiempo_kernel_ms, k_start, k_stop));

    bool ok1 = verificar(B_cuda, referencia, N);
    cout << "[1hilo/elem] blocks=" << blocks << " threads/block=" << threads
         << " tiempo kernel=" << tiempo_kernel_ms << " ms"
         << " tiempo total (con copias)=" << tiempo_total_ms << " ms"
         << " checksum=" << checksum(B_cuda)
         << " -> " << (ok1 ? "CORRECTO" : "INCORRECTO") << "\n";

    // Version Grid-Stride Loop
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(k_start));
    transformarGridStride<<<blocks, threads>>>(d_A, d_B, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(k_stop));
    CUDA_CHECK(cudaMemcpy(B_gs.data(), d_B, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&tiempo_total_ms, start, stop));
    CUDA_CHECK(cudaEventSynchronize(k_stop));
    CUDA_CHECK(cudaEventElapsedTime(&tiempo_kernel_ms, k_start, k_stop));

    bool ok2 = verificar(B_gs, referencia, N);
    cout << "[grid-stride] blocks=" << blocks << " threads/block=" << threads
         << " tiempo kernel=" << tiempo_kernel_ms << " ms"
         << " tiempo total (con copias)=" << tiempo_total_ms << " ms"
         << " checksum=" << checksum(B_gs)
         << " -> " << (ok2 ? "CORRECTO" : "INCORRECTO") << "\n";

    cout << "[CPU] tiempo=" << tiempoCPU.count() << " ms checksum=" << checksum(referencia) << "\n";
    cout << "[NOTA] Se reportan por separado el tiempo de ejecucion puro del kernel y el tiempo total (incluye copias CPU<->GPU)\n";

    cout << "Primeros 10 valores: ";
    for (int i = 0; i < 10 && i < N; i++) {
        cout << referencia[i] << " ";
    }
    cout << endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(k_start));
    CUDA_CHECK(cudaEventDestroy(k_stop));
}

int main(int argc, char** argv) {
    int N_default = 1 << 24;
    int N = (argc > 1) ? atoi(argv[1]) : N_default;
    cout << "--- Problema 1: Transformacion lineal B[i] = 3*A[i] + 7 ---\n";
    
    if (argc > 2) {
        int threads = atoi(argv[2]);
        probar(N, threads);
    } else {
        // Mediciones del impacto del tamano de bloque en el rendimiento
        cout << "\n--- Comparacion de tamanos de bloque (Threads per Block) para N = " << N << " ---\n";
        for (int t : {32, 64, 128, 256, 512, 1024}) {
            probar(N, t);
        }
    }
    return 0;
}
