#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>

using namespace std;

float sumaSecuencial(const vector<float>& A, int N) {
    float suma = 0.0f;

    for (int i = 0; i < N; i++) {
        suma += A[i];
    }

    return suma;
}

int main(int argc, char** argv) {
    int N = 1 << 24;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    vector<float> A(N);

    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
    }

    auto inicio = chrono::high_resolution_clock::now();
    float resultado = sumaSecuencial(A, N);
    auto fin = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> tiempoCPU = fin - inicio;

    cout << "Problema 3: Reduccion de suma" << endl;
    cout << "N: " << N << endl;
    cout << "Resultado CPU: " << resultado << endl;
    cout << "Tiempo CPU: " << tiempoCPU.count() << " ms" << endl;

    return 0;
}
