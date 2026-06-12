#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstdlib>

using namespace std;

const int NUM_BINS = 256;

void histogramaSecuencial(const vector<unsigned char>& A,
                          vector<unsigned int>& hist,
                          int N) {
    for (int i = 0; i < NUM_BINS; i++) {
        hist[i] = 0;
    }

    for (int i = 0; i < N; i++) {
        unsigned char valor = A[i];
        hist[valor]++;
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

int main(int argc, char** argv) {
    int N = 1 << 26;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    vector<unsigned char> A(N);
    vector<unsigned int> hist(NUM_BINS);
    vector<unsigned int> referencia(NUM_BINS);

    mt19937 rng(12345);
    uniform_int_distribution<int> dist(0, NUM_BINS - 1);

    for (int i = 0; i < N; i++) {
        A[i] = static_cast<unsigned char>(dist(rng));
    }

    auto inicio = chrono::high_resolution_clock::now();
    histogramaSecuencial(A, referencia, N);
    auto fin = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> tiempoCPU = fin - inicio;

    cout << "Problema 4: Histograma de 256 bins" << endl;
    cout << "N: " << N << endl;
    cout << "Tiempo CPU: " << tiempoCPU.count() << " ms" << endl;
    cout << "Suma de bins: " << sumaBins(referencia) << endl;

    cout << "Primeros 10 bins:" << endl;
    for (int i = 0; i < 10; i++) {
        cout << "hist[" << i << "] = " << referencia[i] << endl;
    }

    return 0;
}
