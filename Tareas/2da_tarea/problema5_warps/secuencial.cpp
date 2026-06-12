#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstdlib>

using namespace std;

int contarMayoresSecuencial(const vector<int>& A, int N, int umbral) {
    int contador = 0;

    for (int i = 0; i < N; i++) {
        if (A[i] > umbral) {
            contador++;
        }
    }

    return contador;
}

int main(int argc, char** argv) {
    int N = 1 << 26;
    int umbral = 500;

    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (argc > 2) {
        umbral = atoi(argv[2]);
    }

    vector<int> A(N);

    mt19937 rng(12345);
    uniform_int_distribution<int> dist(0, 1000);

    for (int i = 0; i < N; i++) {
        A[i] = dist(rng);
    }

    auto inicio = chrono::high_resolution_clock::now();
    int resultado = contarMayoresSecuencial(A, N, umbral);
    auto fin = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> tiempoCPU = fin - inicio;

    cout << "Problema 5: Conteo con condicion" << endl;
    cout << "N: " << N << endl;
    cout << "Umbral: " << umbral << endl;
    cout << "Resultado CPU: " << resultado << endl;
    cout << "Tiempo CPU: " << tiempoCPU.count() << " ms" << endl;

    return 0;
}
