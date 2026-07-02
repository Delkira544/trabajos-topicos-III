// CUDA C++ CLÁSICO (sin Tile) — versión 2 de 4.
// Pipeline RGB: Gaussian (separable) -> Sobel -> resize bilineal.
// Kernels tradicionales con grilla/bloques/hilos e indexación explícita.
// Memoria: imagen interleaved RGB (uint8). Bordes: clamp (réplica).
// Mide H2D, cada kernel y D2H con CUDA Events; imprime 1 línea JSON (contrato bench/).
//
// Compilar:  cuda_classic\build.ps1   (carga vcvars + nvcc -arch=sm_89)
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image.h"
#include "../third_party/stb_image_write.h"
#include "../common/common.cuh"
#include <filesystem>

#define BS 16                         // bloque 16x16 hilos
__constant__ float c_gauss[33];       // kernel gaussiano 1D (ksize<=33)

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// --- Gaussian separable: pasada horizontal (uint8 -> float) ---
__global__ void gaussH(const unsigned char* in, float* tmp, int w, int h, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    for (int c = 0; c < 3; ++c) {
        float acc = 0.f;
        for (int t = -r; t <= r; ++t) {
            int xx = clampi(x + t, 0, w - 1);
            acc += c_gauss[t + r] * (float)in[(y * w + xx) * 3 + c];
        }
        tmp[(y * w + x) * 3 + c] = acc;
    }
}

// --- Gaussian separable: pasada vertical (float -> uint8) ---
__global__ void gaussV(const float* tmp, unsigned char* out, int w, int h, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    for (int c = 0; c < 3; ++c) {
        float acc = 0.f;
        for (int t = -r; t <= r; ++t) {
            int yy = clampi(y + t, 0, h - 1);
            acc += c_gauss[t + r] * tmp[(yy * w + x) * 3 + c];
        }
        float v = fminf(fmaxf(acc, 0.f), 255.f);
        out[(y * w + x) * 3 + c] = (unsigned char)v;   // truncado (igual que numpy astype)
    }
}

__device__ __forceinline__ float lumAt(const unsigned char* in, int w, int h, int x, int y) {
    x = clampi(x, 0, w - 1); y = clampi(y, 0, h - 1);
    int i = (y * w + x) * 3;
    return 0.299f * in[i] + 0.587f * in[i + 1] + 0.114f * in[i + 2];
}

// --- Sobel sobre luminancia: salida 1 canal (uint8) ---
__global__ void sobelK(const unsigned char* in, unsigned char* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float p00 = lumAt(in, w, h, x-1, y-1), p01 = lumAt(in, w, h, x, y-1), p02 = lumAt(in, w, h, x+1, y-1);
    float p10 = lumAt(in, w, h, x-1, y),                                  p12 = lumAt(in, w, h, x+1, y);
    float p20 = lumAt(in, w, h, x-1, y+1), p21 = lumAt(in, w, h, x, y+1), p22 = lumAt(in, w, h, x+1, y+1);
    float gx = -p00 + p02 - 2.f*p10 + 2.f*p12 - p20 + p22;
    float gy = -p00 - 2.f*p01 - p02 + p20 + 2.f*p21 + p22;
    float m = sqrtf(gx*gx + gy*gy);
    out[y * w + x] = (unsigned char)fminf(fmaxf(m, 0.f), 255.f);
}

// --- Resize bilineal: coordenada fraccional (out+0.5)/scale - 0.5 ---
__global__ void resizeK(const unsigned char* in, unsigned char* out,
                        int w, int h, int nw, int nh, float scale) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= nw || oy >= nh) return;
    float fx = (ox + 0.5f) / scale - 0.5f;
    float fy = (oy + 0.5f) / scale - 0.5f;
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    float dx = fx - x0, dy = fy - y0;
    int x0c = clampi(x0, 0, w-1), x1c = clampi(x0+1, 0, w-1);
    int y0c = clampi(y0, 0, h-1), y1c = clampi(y0+1, 0, h-1);
    for (int c = 0; c < 3; ++c) {
        float A = in[(y0c*w + x0c)*3 + c], B = in[(y0c*w + x1c)*3 + c];
        float C = in[(y1c*w + x0c)*3 + c], D = in[(y1c*w + x1c)*3 + c];
        float v = A*(1-dx)*(1-dy) + B*dx*(1-dy) + C*(1-dx)*dy + D*dx*dy;
        out[(oy*nw + ox)*3 + c] = (unsigned char)fminf(fmaxf(v, 0.f), 255.f);
    }
}

int main(int argc, char** argv) {
    Args a = parseArgs(argc, argv);
    if (a.image.empty()) { fprintf(stderr, "uso: --image <png> [--ksize --sigma --scale --reps]\n"); return 1; }

    int w, h;
    unsigned char* h_in = loadRGB(a.image, w, h);
    int nw = (int)floorf(w * a.scale + 0.5f);   // round half-up (== CPU)
    int nh = (int)floorf(h * a.scale + 0.5f);
    if (nw < 1) nw = 1; if (nh < 1) nh = 1;

    // kernel gaussiano 1D normalizado en host -> constant memory
    int r = a.ksize / 2;
    std::vector<float> kg(a.ksize);
    float sum = 0.f;
    for (int i = -r; i <= r; ++i) { kg[i+r] = expf(-(i*i) / (2.f*a.sigma*a.sigma)); sum += kg[i+r]; }
    for (float& v : kg) v /= sum;
    CUDA_CHECK(cudaMemcpyToSymbol(c_gauss, kg.data(), a.ksize * sizeof(float)));

    size_t nRGB = (size_t)w * h * 3, nGray = (size_t)w * h, nOut = (size_t)nw * nh * 3;
    unsigned char *d_in, *d_g, *d_s, *d_r; float* d_tmp;
    CUDA_CHECK(cudaMalloc(&d_in,  nRGB));
    CUDA_CHECK(cudaMalloc(&d_tmp, nRGB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g,   nRGB));
    CUDA_CHECK(cudaMalloc(&d_s,   nGray));
    CUDA_CHECK(cudaMalloc(&d_r,   nOut));
    std::vector<unsigned char> h_g(nRGB), h_s(nGray), h_r(nOut);

    int bs = a.block;                       // lado de bloque configurable (8/16/32)
    dim3 block(bs, bs);
    dim3 grid((w + bs - 1) / bs, (h + bs - 1) / bs);
    dim3 gridR((nw + bs - 1) / bs, (nh + bs - 1) / bs);

    std::vector<float> vt, vh, vg, vs, vr, vd, vgh, vgv;   // vgh/vgv: por-kernel gaussiano
    for (int it = 0; it < a.reps; ++it) {
        GpuTimer t;
        t.start(); CUDA_CHECK(cudaMemcpy(d_in, h_in, nRGB, cudaMemcpyHostToDevice)); float th = t.stop();

        t.start(); gaussH<<<grid, block>>>(d_in, d_tmp, w, h, r);  float tgh = t.stop();
        t.start(); gaussV<<<grid, block>>>(d_tmp, d_g, w, h, r);   float tgv = t.stop();
        float tg = tgh + tgv;

        t.start(); sobelK<<<grid, block>>>(d_g, d_s, w, h); float ts = t.stop();
        t.start(); resizeK<<<gridR, block>>>(d_g, d_r, w, h, nw, nh, a.scale); float tr = t.stop();
        CUDA_CHECK(cudaGetLastError());

        t.start();
        CUDA_CHECK(cudaMemcpy(h_g.data(), d_g, nRGB,  cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_s.data(), d_s, nGray, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_r.data(), d_r, nOut,  cudaMemcpyDeviceToHost));
        float td = t.stop();

        vt.push_back(th+tg+ts+tr+td); vh.push_back(th);
        vg.push_back(tg); vs.push_back(ts); vr.push_back(tr); vd.push_back(td);
        vgh.push_back(tgh); vgv.push_back(tgv);
    }

    // guardar salidas de la última corrida
    std::string name = stem(a.image);
    std::filesystem::create_directories("out");   // out/ relativo al cwd (raíz del proyecto)
    char tag[512];
    snprintf(tag, sizeof(tag), "out/%s__cuda_classic__k%d_s%.1f__x%g",
             name.c_str(), a.ksize, a.sigma, a.scale);   // tag sin block: config default sobreescribe
    savePNG(std::string(tag) + "__1gauss.png",  w,  h,  3, h_g.data());
    savePNG(std::string(tag) + "__2sobel.png",  w,  h,  1, h_s.data());
    savePNG(std::string(tag) + "__3resize.png", nw, nh, 3, h_r.data());

    auto mean = [](const std::vector<float>& v){ double m,s; meanStd(v,m,s); return m; };
    double tot_m, tot_s; meanStd(vt, tot_m, tot_s);

    printf("{\"version\":\"cuda_classic\",\"image\":\"%s\",\"width\":%d,\"height\":%d,"
           "\"gauss\":\"k%d_s%.1f\",\"scale\":%g,\"reps\":%d,"
           "\"total_ms_mean\":%.4f,\"total_ms_std\":%.4f,"
           "\"gaussian_ms\":%.4f,\"sobel_ms\":%.4f,\"resize_ms\":%.4f,"
           "\"kernel_ms\":%.4f,\"h2d_ms\":%.4f,\"d2h_ms\":%.4f,\"tile\":\"block%dx%d\","
           "\"notes\":\"gaussH=%.4f;gaussV=%.4f;sobel=%.4f;resize=%.4f\"}\n",
           name.c_str(), w, h, a.ksize, a.sigma, a.scale, a.reps,
           tot_m, tot_s, mean(vg), mean(vs), mean(vr),
           mean(vg)+mean(vs)+mean(vr), mean(vh), mean(vd), bs, bs,
           mean(vgh), mean(vgv), mean(vs), mean(vr));

    stbi_image_free(h_in);
    cudaFree(d_in); cudaFree(d_tmp); cudaFree(d_g); cudaFree(d_s); cudaFree(d_r);
    return 0;
}
