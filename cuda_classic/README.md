# CUDA C++ clásico (sin Tile) — versión 2 de 4  ✅

Pipeline completa (Gaussian separable → Sobel → resize bilineal) con kernels
tradicionales: grilla/bloques/hilos e **indexación explícita**. Memoria: RGB
interleaved (uint8); kernel gaussiano 1D en `__constant__`. Bordes: clamp.

## Uso
```powershell
.\cuda_classic\build.ps1     # compila (nvcc -O3 -std=c++17 -arch=sm_89)
.\cuda_classic\pipeline.exe --image data\medium_2048.png --ksize 5 --sigma 1.0 `
    --scale 0.5 --reps 10 --block 16
```
- `--block N` : lado del bloque 2D (8/16/32) — para medir el efecto de configuración (§8).
- Mide con **CUDA Events**: H2D, cada kernel (gaussH, gaussV, sobel, resize) y D2H.
- El desglose por kernel va en el campo `notes` del JSON
  (`gaussH=..;gaussV=..;sobel=..;resize=..`).
- Salidas en `out/`, JSON por stdout (contrato de `bench/run_bench.py`).

## Validación
`python bench/validate.py --image data/medium_2048.png --version cuda_classic`
→ MAE ≈ 0.003, error máx 1–3 niveles (redondeo float): coincide con la referencia CPU.
