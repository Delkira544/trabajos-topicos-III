// CUDA TILE C++ — versión 3 de 4.
// Gaussian blur (separable) implementado con el MODELO DE TILES (cuda::tiles):
// se cargan tiles 1D y se acumulan versiones desplazadas del puntero base
// (vecino x = ±3 en RGB interleaved; vecino y = ±3W). Bordes: zero-pad (la API de
// tiles de alto nivel no ofrece padding clamp). Buffers con halo para evitar OOB.
//
// Sobel y resize se incluyen como kernels CUDA clásicos de apoyo (el resize bilineal
// es un gather con coordenadas fraccionales que NO mapea a la API de tiles de alto
// nivel; ver README). Mide con CUDA Events; imprime 1 línea JSON (contrato bench/).
//
// Compilar:  cuda_tile\build.ps1   (nvcc --enable-tile -std=c++20 -arch=sm_89)
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image.h"
#include "../third_party/stb_image_write.h"
#include "../common/common.cuh"
#include <cuda_tile.h>
#include <filesystem>

namespace ct = cuda::tiles;
using namespace ct::literals;

#define BS 16
// carga el tile bx de un span 1D que empieza en el puntero P (zero-pad fuera de rango)
#define LDTN(P, TNL) (ct::partition_view{ct::tensor_span{(P), ct::extents{L}}, ct::shape{TNL}}.load_masked(bx))

// ----------------- Gaussian separable EN MODELO TILE -----------------
// Una pasada 1D: out[i] = Σ_t w[t]·in[i + step·t], con step=3 (horizontal) o 3W (vertical).
// Pesos simétricos como escalares w0(centro)..w4 (radio ≤ 4 -> ksize ≤ 9).
// El tamaño de tile es constante de compilación: variantes 128/256/512 (efecto §8).
#define DEF_GAUSS(TNV)                                                                          \
__tile_global__ void gaussPass_##TNV(float* in, float* out, long long L, long long step, int r,\
                                     float w0, float w1, float w2, float w3, float w4) {       \
    auto bx = ct::bid().x;                                                                     \
    auto acc = w0 * LDTN(in, TNV##_ic);                                                        \
    if (r >= 1) acc = acc + w1 * (LDTN(in + step,     TNV##_ic) + LDTN(in - step,     TNV##_ic)); \
    if (r >= 2) acc = acc + w2 * (LDTN(in + 2 * step, TNV##_ic) + LDTN(in - 2 * step, TNV##_ic)); \
    if (r >= 3) acc = acc + w3 * (LDTN(in + 3 * step, TNV##_ic) + LDTN(in - 3 * step, TNV##_ic)); \
    if (r >= 4) acc = acc + w4 * (LDTN(in + 4 * step, TNV##_ic) + LDTN(in - 4 * step, TNV##_ic)); \
    ct::partition_view{ct::tensor_span{out, ct::extents{L}}, ct::shape{TNV##_ic}}.store_masked(acc, bx); \
}
DEF_GAUSS(128)
DEF_GAUSS(256)
DEF_GAUSS(512)

// ----------------- kernels clásicos de apoyo -----------------
__global__ void u8ToF(const unsigned char* s, float* d, long long L) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < L) d[i] = (float)s[i];
}
__global__ void fToU8(const float* s, unsigned char* d, long long L) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < L) d[i] = (unsigned char)fminf(fmaxf(s[i], 0.f), 255.f);
}
__device__ __forceinline__ int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
__device__ __forceinline__ float lumAt(const unsigned char* in, int w, int h, int x, int y) {
    x = clampi(x, 0, w - 1); y = clampi(y, 0, h - 1); int i = (y * w + x) * 3;
    return 0.299f * in[i] + 0.587f * in[i + 1] + 0.114f * in[i + 2];
}
// luminancia (glue clásico): RGB uint8 -> plano float
__global__ void lumKern(const unsigned char* in, float* lum, int w, int h) {
    long long M = (long long)w * h, i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < M) { long long p = i * 3; lum[i] = 0.299f*in[p] + 0.587f*in[p+1] + 0.114f*in[p+2]; }
}

// ----------------- Sobel EN MODELO TILE -----------------
// 3x3 sobre el plano de luminancia mediante tiles 1D desplazados (vecino x = ±1, y = ±W).
// lum apunta al inicio del plano dentro de un buffer con halo; out = magnitud (float).
#define LDMN(P, TNL) (ct::partition_view{ct::tensor_span{(P), ct::extents{M}}, ct::shape{TNL}}.load_masked(bx))
#define DEF_SOBEL(TNV)                                                                       \
__tile_global__ void sobelTile_##TNV(float* lum, float* mag, long long M, long long W) {    \
    auto bx = ct::bid().x;                                                                  \
    auto tl = LDMN(lum - W - 1, TNV##_ic), tc = LDMN(lum - W, TNV##_ic), tr = LDMN(lum - W + 1, TNV##_ic); \
    auto l  = LDMN(lum - 1, TNV##_ic),                                   rr = LDMN(lum + 1, TNV##_ic);     \
    auto bl = LDMN(lum + W - 1, TNV##_ic), bc = LDMN(lum + W, TNV##_ic), br = LDMN(lum + W + 1, TNV##_ic); \
    auto gx = (tr + 2.0f*rr + br) - (tl + 2.0f*l  + bl);                                    \
    auto gy = (bl + 2.0f*bc + br) - (tl + 2.0f*tc + tr);                                    \
    auto m  = ct::sqrt(gx*gx + gy*gy);                                                      \
    ct::partition_view{ct::tensor_span{mag, ct::extents{M}}, ct::shape{TNV##_ic}}.store_masked(m, bx); \
}
DEF_SOBEL(128)
DEF_SOBEL(256)
DEF_SOBEL(512)
__global__ void resizeK(const unsigned char* in, unsigned char* out,
                        int w, int h, int nw, int nh, float scale) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x, oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= nw || oy >= nh) return;
    float fx = (ox + 0.5f) / scale - 0.5f, fy = (oy + 0.5f) / scale - 0.5f;
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    float dx = fx - x0, dy = fy - y0;
    int x0c = clampi(x0,0,w-1), x1c = clampi(x0+1,0,w-1), y0c = clampi(y0,0,h-1), y1c = clampi(y0+1,0,h-1);
    for (int c = 0; c < 3; ++c) {
        float A = in[(y0c*w+x0c)*3+c], B = in[(y0c*w+x1c)*3+c], C = in[(y1c*w+x0c)*3+c], D = in[(y1c*w+x1c)*3+c];
        float v = A*(1-dx)*(1-dy) + B*dx*(1-dy) + C*(1-dx)*dy + D*dx*dy;
        out[(oy*nw+ox)*3+c] = (unsigned char)fminf(fmaxf(v,0.f),255.f);
    }
}

int main(int argc, char** argv) {
    Args a = parseArgs(argc, argv);
    if (a.image.empty()) { fprintf(stderr, "uso: --image <png> [--ksize --sigma --scale --reps]\n"); return 1; }
    int w, h;
    unsigned char* h_in = loadRGB(a.image, w, h);
    int nw = (int)floorf(w * a.scale + 0.5f), nh = (int)floorf(h * a.scale + 0.5f);
    if (nw < 1) nw = 1; if (nh < 1) nh = 1;
    int r = a.ksize / 2;

    if (r > 4) { fprintf(stderr, "ksize>9 no soportado por la version tile\n"); return 1; }
    std::vector<float> kg(a.ksize); float sum = 0.f;
    for (int i = -r; i <= r; ++i) { kg[i+r] = expf(-(i*i)/(2.f*a.sigma*a.sigma)); sum += kg[i+r]; }
    for (float& v : kg) v /= sum;
    float W[5] = {0,0,0,0,0};            // W[0]=centro, W[t]=peso simétrico del tap t
    W[0] = kg[r];
    for (int t = 1; t <= r; ++t) W[t] = kg[r + t];

    long long L = (long long)w * h * 3;
    long long HALO = (long long)3 * w * r + 1024;          // cubre shift vertical 3W·r y borde de tile
    long long Lpad = L + 2 * HALO;
    size_t nGray = (size_t)w * h, nOut = (size_t)nw * nh * 3;

    long long M = (long long)w * h;                  // píxeles (plano luminancia)
    long long HALO_S = (long long)w + 1024;          // cubre shift vertical ±W del Sobel
    unsigned char *d_in, *d_g, *d_s, *d_r;
    float *d_inf, *d_tmp, *d_outf, *d_lum, *d_mag;
    CUDA_CHECK(cudaMalloc(&d_in, L));
    CUDA_CHECK(cudaMalloc(&d_inf, Lpad * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp, Lpad * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outf, L * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lum, (M + 2 * HALO_S) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mag, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g, L));
    CUDA_CHECK(cudaMalloc(&d_s, nGray));
    CUDA_CHECK(cudaMalloc(&d_r, nOut));
    std::vector<unsigned char> h_g(L), h_s(nGray), h_r(nOut);

    dim3 block(BS, BS), grid((w+BS-1)/BS, (h+BS-1)/BS), gridR((nw+BS-1)/BS, (nh+BS-1)/BS);
    int t1 = 256; long long g1 = (L + t1 - 1) / t1;            // grid 1D conversiones
    int tn = (a.tn == 128 || a.tn == 512) ? a.tn : 256;        // tamaño de tile (variantes)
    long long gTile = (L + tn - 1) / tn;                       // grid 1D tiles gauss
    long long gTileS = (M + tn - 1) / tn;                      // grid 1D tiles sobel

    // despacho de la variante compilada según --tn
    auto launchGauss = [&](float* in, float* out, long long step) {
        if      (tn == 128) gaussPass_128<<<gTile, 1>>>(in, out, L, step, r, W[0],W[1],W[2],W[3],W[4]);
        else if (tn == 512) gaussPass_512<<<gTile, 1>>>(in, out, L, step, r, W[0],W[1],W[2],W[3],W[4]);
        else                gaussPass_256<<<gTile, 1>>>(in, out, L, step, r, W[0],W[1],W[2],W[3],W[4]);
    };
    auto launchSobel = [&](float* lum, float* mag) {
        if      (tn == 128) sobelTile_128<<<gTileS, 1>>>(lum, mag, M, (long long)w);
        else if (tn == 512) sobelTile_512<<<gTileS, 1>>>(lum, mag, M, (long long)w);
        else                sobelTile_256<<<gTileS, 1>>>(lum, mag, M, (long long)w);
    };

    std::vector<float> vt, vh, vg, vs, vr, vd, vgh, vgv, vst;  // vgh/vgv/vst: por-kernel
    for (int it = 0; it < a.reps; ++it) {
        GpuTimer t;
        t.start(); CUDA_CHECK(cudaMemcpy(d_in, h_in, L, cudaMemcpyHostToDevice)); float th = t.stop();

        t.start();
        CUDA_CHECK(cudaMemset(d_inf, 0, Lpad * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_tmp, 0, Lpad * sizeof(float)));
        u8ToF<<<g1, t1>>>(d_in, d_inf + HALO, L);
        float tprep = t.stop();
        t.start(); launchGauss(d_inf + HALO, d_tmp + HALO, 3);              float tgh = t.stop(); // horizontal
        t.start(); launchGauss(d_tmp + HALO, d_outf, (long long)3 * w);     float tgv = t.stop(); // vertical
        t.start(); fToU8<<<g1, t1>>>(d_outf, d_g, L);                       float tconv = t.stop();
        float tg = tprep + tgh + tgv + tconv;

        t.start();
        CUDA_CHECK(cudaMemset(d_lum, 0, (M + 2 * HALO_S) * sizeof(float)));
        lumKern<<<g1, t1>>>(d_g, d_lum + HALO_S, w, h);
        float tsprep = t.stop();
        t.start(); launchSobel(d_lum + HALO_S, d_mag);                      float tst = t.stop();
        t.start(); fToU8<<<(int)((M + t1 - 1) / t1), t1>>>(d_mag, d_s, M);  float tsconv = t.stop();
        float ts = tsprep + tst + tsconv;
        t.start(); resizeK<<<gridR, block>>>(d_g, d_r, w, h, nw, nh, a.scale); float tr = t.stop();
        CUDA_CHECK(cudaGetLastError());

        t.start();
        CUDA_CHECK(cudaMemcpy(h_g.data(), d_g, L, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_s.data(), d_s, nGray, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_r.data(), d_r, nOut, cudaMemcpyDeviceToHost));
        float td = t.stop();

        vt.push_back(th+tg+ts+tr+td); vh.push_back(th);
        vg.push_back(tg); vs.push_back(ts); vr.push_back(tr); vd.push_back(td);
        vgh.push_back(tgh); vgv.push_back(tgv); vst.push_back(tst);
    }

    std::string name = stem(a.image);
    std::filesystem::create_directories("out");
    char tag[512];
    snprintf(tag, sizeof(tag), "out/%s__cuda_tile__k%d_s%.1f__x%g", name.c_str(), a.ksize, a.sigma, a.scale);
    savePNG(std::string(tag) + "__1gauss.png",  w,  h,  3, h_g.data());
    savePNG(std::string(tag) + "__2sobel.png",  w,  h,  1, h_s.data());
    savePNG(std::string(tag) + "__3resize.png", nw, nh, 3, h_r.data());

    auto mean = [](const std::vector<float>& v){ double m,s; meanStd(v,m,s); return m; };
    double tot_m, tot_s; meanStd(vt, tot_m, tot_s);
    printf("{\"version\":\"cuda_tile\",\"image\":\"%s\",\"width\":%d,\"height\":%d,"
           "\"gauss\":\"k%d_s%.1f\",\"scale\":%g,\"reps\":%d,"
           "\"total_ms_mean\":%.4f,\"total_ms_std\":%.4f,"
           "\"gaussian_ms\":%.4f,\"sobel_ms\":%.4f,\"resize_ms\":%.4f,"
           "\"kernel_ms\":%.4f,\"h2d_ms\":%.4f,\"d2h_ms\":%.4f,\"tile\":\"tile1D_%d\","
           "\"notes\":\"gaussH=%.4f;gaussV=%.4f;sobelTile=%.4f\"}\n",
           name.c_str(), w, h, a.ksize, a.sigma, a.scale, a.reps,
           tot_m, tot_s, mean(vg), mean(vs), mean(vr),
           mean(vg)+mean(vs)+mean(vr), mean(vh), mean(vd), tn,
           mean(vgh), mean(vgv), mean(vst));

    stbi_image_free(h_in);
    cudaFree(d_in); cudaFree(d_inf); cudaFree(d_tmp); cudaFree(d_outf);
    cudaFree(d_lum); cudaFree(d_mag); cudaFree(d_g); cudaFree(d_s); cudaFree(d_r);
    return 0;
}
