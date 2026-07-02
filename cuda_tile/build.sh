#!/usr/bin/env bash
# Compila pipeline.cu (CUDA Tile C++) en Linux: requiere -std=c++20 --enable-tile.
# GPU con compute capability 8.x+ (Ampere/Ada/Blackwell). GPU_ARCH=sm_89 para forzar.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

NVCC="$(find "$ROOT/.venv" -path '*/nvidia/cu13/bin/nvcc' -type f 2>/dev/null | head -1 || true)"
[ -z "$NVCC" ] && NVCC="$(command -v nvcc || true)"
[ -z "$NVCC" ] && { echo "ERROR: nvcc no encontrado. Ejecuta: pip install -r requirements.txt"; exit 1; }

ARCH="${GPU_ARCH:-}"
if [ -z "$ARCH" ]; then
    CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')"
    ARCH="sm_${CC:-89}"
fi

echo "nvcc: $NVCC  arch: $ARCH"
"$NVCC" -O3 -std=c++20 --enable-tile -arch="$ARCH" -cudart static \
    -o "$ROOT/cuda_tile/pipeline" "$ROOT/cuda_tile/pipeline.cu"
echo "OK -> cuda_tile/pipeline"
