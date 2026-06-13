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

void stencilSecuencial(const vector<float>& A, vector<float>& B, int N) {
    if (N <= 0) return;
    if (N == 1) {
        B[0] = A[0];
        return;
    }
    B[0] = A[0] + A[1];
    for (int i = 1; i < N - 1; i++) {
        B[i] = A[i - 1] + A[i] + A[i + 1];
    }
    B[N - 1] = A[N - 2] + A[N - 1];
}

__global__ void stencilGlobal(float* A, float* B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        B[0] = A[0] + A[1];
    } else if (idx == N - 1) {
        B[N - 1] = A[N - 2] + A[N - 1];
    } else if (idx > 0 && idx < N - 1) {
        B[idx] = A[idx - 1] + A[idx] + A[idx + 1];
    }
}

__global__ void stencilShared(float* A, float* B, int N) {
    extern __shared__ float tile[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    tile[threadIdx.x + 1] = (idx < N) ? A[idx] : 0.0f;

    if (threadIdx.x == 0) {
        tile[0] = (idx > 0) ? A[idx - 1] : 0.0f;
    }
    if (threadIdx.x == blockDim.x - 1) {
        tile[blockDim.x + 1] = (idx + 1 < N) ? A[idx + 1] : 0.0f;
    }

    __syncthreads();

    if (idx < N) {
        B[idx] = tile[threadIdx.x] + tile[threadIdx.x + 1] + tile[threadIdx.x + 2];
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

void probar(int N) {
    if (N <= 0) return;
    cout << "\n========== N = " << N << " ==========\n";

    vector<float> A(N);
    vector<float> referencia(N);
    vector<float> B_global(N);
    vector<float> B_shared(N);

    for (int i = 0; i < N; i++) {
        A[i] = static_cast<float>((i % 50) * 0.5f);
    }

    auto inicio = chrono::high_resolution_clock::now();
    stencilSecuencial(A, referencia, N);
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

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Warm-up
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    stencilGlobal<<<blocks, threads>>>(d_A, d_B, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Version memoria global
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(k_start));
    stencilGlobal<<<blocks, threads>>>(d_A, d_B, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(k_stop));
    CUDA_CHECK(cudaMemcpy(B_global.data(), d_B, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&tiempo_total_ms, start, stop));
    CUDA_CHECK(cudaEventSynchronize(k_stop));
    CUDA_CHECK(cudaEventElapsedTime(&tiempo_kernel_ms, k_start, k_stop));

    bool ok1 = verificar(B_global, referencia, N);
    cout << "[global] blocks=" << blocks << " threads/block=" << threads
         << " tiempo kernel=" << tiempo_kernel_ms << " ms"
         << " tiempo total (con copias)=" << tiempo_total_ms << " ms"
         << " checksum=" << checksum(B_global)
         << " -> " << (ok1 ? "CORRECTO" : "INCORRECTO") << "\n";

    // Version shared memory
    size_t sharedBytes = (threads + 2) * sizeof(float);
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(k_start));
    stencilShared<<<blocks, threads, sharedBytes>>>(d_A, d_B, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(k_stop));
    CUDA_CHECK(cudaMemcpy(B_shared.data(), d_B, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&tiempo_total_ms, start, stop));
    CUDA_CHECK(cudaEventSynchronize(k_stop));
    CUDA_CHECK(cudaEventElapsedTime(&tiempo_kernel_ms, k_start, k_stop));

    bool ok2 = verificar(B_shared, referencia, N);
    cout << "[shared] blocks=" << blocks << " threads/block=" << threads
         << " tiempo kernel=" << tiempo_kernel_ms << " ms"
         << " tiempo total (con copias)=" << tiempo_total_ms << " ms"
         << " checksum=" << checksum(B_shared)
         << " -> " << (ok2 ? "CORRECTO" : "INCORRECTO") << "\n";

    cout << "[CPU] tiempo=" << tiempoCPU.count() << " ms checksum=" << checksum(referencia) << "\n";
    cout << "[NOTA] Se reportan por separado el tiempo de ejecucion puro del kernel y el tiempo total (incluye copias CPU<->GPU)\n";

    if (N == 1) {
        cout << "Unico valor: " << referencia[0] << endl;
    } else {
        cout << "Primeros 10 valores: ";
        for (int i = 0; i < 10 && i < N; i++) {
            cout << referencia[i] << " ";
        }
        cout << endl;
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(k_start));
    CUDA_CHECK(cudaEventDestroy(k_stop));
}

int main(int argc, char** argv) {
    int N_default = 1 << 24;
    cout << "--- Problema 2: Stencil 1D B[i] = A[i-1] + A[i] + A[i+1] ---\n";
    probar(1024);
    probar(100000);
    probar(1 << 20);
    int N = (argc > 1) ? atoi(argv[1]) : N_default;
    probar(N);
    return 0;
}
