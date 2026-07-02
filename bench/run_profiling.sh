#!/usr/bin/env bash
# Profiling con Nsight Systems (nsys) y/o Nsight Compute (ncu) en Linux — §6/§8.
# En Linux Nsight Systems suele instalarse con el CUDA Toolkit del sistema o via
# apt (nsight-systems) / descarga NVIDIA. Con UNA herramienta basta según la pauta.
# Genera perfiles y resúmenes en results/profiles/.
#
# Uso:  ./bench/run_profiling.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results/profiles

IMG="data/medium_2048.png"
ARGS=(--image "$IMG" --ksize 5 --sigma 1.0 --scale 0.5 --reps 3)

NSYS="$(command -v nsys || true)"
NCU="$(command -v ncu || true)"
if [ -z "$NSYS" ] && [ -z "$NCU" ]; then
    echo "Ni nsys ni ncu encontrados. Instala Nsight Systems (recomendado) y reintenta."
    exit 1
fi

for ver in cuda_classic cuda_tile; do
    exe="./$ver/pipeline"
    [ -x "$exe" ] || { echo "falta $exe (corre $ver/build.sh)"; continue; }
    if [ -n "$NSYS" ]; then
        echo "== nsys $ver =="
        "$NSYS" profile --stats=true -o "results/profiles/nsys_$ver" --force-overwrite=true \
            "$exe" "${ARGS[@]}" | tee "results/profiles/nsys_${ver}_summary.txt" | tail -3
        "$NSYS" stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --format table \
            --output "results/profiles/nsys_${ver}_stats" --force-export=true \
            "results/profiles/nsys_$ver.nsys-rep" >/dev/null || true
    fi
    if [ -n "$NCU" ]; then
        echo "== ncu $ver =="
        "$NCU" --set basic --launch-count 8 -o "results/profiles/ncu_$ver" --force-overwrite \
            "$exe" "${ARGS[@]}" | tee "results/profiles/ncu_${ver}_summary.txt" | tail -3
    fi
done
echo "OK -> results/profiles/"
