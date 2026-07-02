// Utilidades compartidas por las versiones CUDA C++ (clásico y Tile):
// parseo de args, I/O de imágenes (stb), timer con CUDA Events y estadísticas.
//
// IMPORTANTE: el .cu que incluya este header debe, ANTES, definir las macros de
// implementación e incluir los stb_*.h UNA sola vez (su bloque de implementación
// no tiene guard de inclusión única, por eso no se incluye aquí):
//   #define STB_IMAGE_IMPLEMENTATION
//   #define STB_IMAGE_WRITE_IMPLEMENTATION
//   #include "../third_party/stb_image.h"
//   #include "../third_party/stb_image_write.h"
//   #include "../common/common.cuh"
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1);} } while (0)

struct Args {
    std::string image;
    int   ksize  = 5;
    float sigma  = 1.0f;
    float scale  = 0.5f;
    int   reps   = 10;
    int   block  = 16;      // lado del bloque 2D (clásico) — efecto de configuración (§8)
    int   tn     = 256;     // tamaño de tile 1D (tile/cutile) — efecto de configuración (§8)
    std::string border = "clamp";
};

inline Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "--image" && i + 1 < argc) a.image = argv[++i];
        else if (k == "--ksize" && i + 1 < argc) a.ksize = atoi(argv[++i]);
        else if (k == "--sigma" && i + 1 < argc) a.sigma = (float)atof(argv[++i]);
        else if (k == "--scale" && i + 1 < argc) a.scale = (float)atof(argv[++i]);
        else if (k == "--reps"  && i + 1 < argc) a.reps  = atoi(argv[++i]);
        else if (k == "--block" && i + 1 < argc) a.block = atoi(argv[++i]);
        else if (k == "--tn"    && i + 1 < argc) a.tn    = atoi(argv[++i]);
        else if (k == "--border"&& i + 1 < argc) a.border = argv[++i];
    }
    return a;
}

inline std::string stem(const std::string& p) {
    size_t s = p.find_last_of("/\\");
    size_t b = (s == std::string::npos) ? 0 : s + 1;
    size_t d = p.find_last_of('.');
    size_t e = (d == std::string::npos || d < b) ? p.size() : d;
    return p.substr(b, e - b);
}

inline unsigned char* loadRGB(const std::string& path, int& w, int& h) {
    int comp;
    unsigned char* d = stbi_load(path.c_str(), &w, &h, &comp, 3);  // forzar 3 canales
    if (!d) { fprintf(stderr, "No se pudo cargar %s\n", path.c_str()); std::exit(1); }
    return d;
}

inline void savePNG(const std::string& path, int w, int h, int ch, const unsigned char* d) {
    stbi_write_png(path.c_str(), w, h, ch, d, w * ch);
}

inline void meanStd(const std::vector<float>& v, double& m, double& sd) {
    m = 0.0; for (float x : v) m += x; m /= v.size();
    double s = 0.0; for (float x : v) s += (x - m) * (x - m);
    sd = std::sqrt(s / v.size());
}

// Timer basado en CUDA Events (exigido por la pauta para medir kernels/transferencias).
struct GpuTimer {
    cudaEvent_t a, b;
    GpuTimer()  { cudaEventCreate(&a); cudaEventCreate(&b); }
    ~GpuTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void  start() { cudaEventRecord(a); }
    float stop()  { cudaEventRecord(b); cudaEventSynchronize(b);
                    float ms; cudaEventElapsedTime(&ms, a, b); return ms; }
};
