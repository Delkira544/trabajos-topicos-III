#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>

using namespace std;

void stencilSecuencial(const vector<float>& A, vector<float>& B, int N) {
    if (N <= 0) {
        return;
    }

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

int main(int argc, char** argv) {
    int N = 1 << 24;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    vector<float> A(N);
    vector<float> B(N);
    vector<float> referencia(N);

    for (int i = 0; i < N; i++) {
        A[i] = static_cast<float>((i % 50) * 0.5f);
    }

    auto inicio = chrono::high_resolution_clock::now();
    stencilSecuencial(A, referencia, N);
    auto fin = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> tiempoCPU = fin - inicio;

    cout << "Problema 2: Stencil 1D" << endl;
    cout << "N: " << N << endl;
    cout << "Tiempo CPU: " << tiempoCPU.count() << " ms" << endl;
    cout << "Checksum CPU: " << checksum(referencia) << endl;

    cout << "Primeros 10 valores:" << endl;
    for (int i = 0; i < 10 && i < N; i++) {
        cout << referencia[i] << " ";
    }
    cout << endl;

    return 0;
}
