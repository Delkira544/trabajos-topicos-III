"""Sweep de configuración (§8): efecto del tamaño de bloque (CUDA clásico) y del
tamaño de tile (CUDA Tile C++ / cuTile Python) sobre una imagen fija.

No mide por sí mismo: invoca cada ejecutable (que mide con CUDA Events) y agrega el
JSON a results/config_sweep.csv. Comandos registrados en results/commands.log.

Uso:
  python bench/run_config_sweep.py                # medium_2048, k5 s1.0 x0.5, 10 reps
  python bench/run_config_sweep.py --image data/large_4096.png --reps 10
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
EXE = "pipeline.exe" if os.name == "nt" else "pipeline"   # binario según plataforma

BLOCKS = [8, 16, 32]          # cuda_classic: bloque NxN
TNS = [128, 256, 512]         # cuda_tile / cutile_py: tile 1D


def run_one(cmd: list[str], log) -> dict | None:
    log.write(" ".join(cmd) + "\n")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT)
    except subprocess.CalledProcessError as e:
        print(f"  [error] {' '.join(cmd)}\n{e.stderr.strip()[:300]}")
        return None
    line = [l for l in out.stdout.splitlines() if l.strip()][-1]
    return json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="data/medium_2048.png")
    ap.add_argument("--ksize", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    img = str(ROOT / args.image)
    base = ["--image", img, "--ksize", str(args.ksize), "--sigma", str(args.sigma),
            "--scale", str(args.scale), "--reps", str(args.reps)]
    rows: list[dict] = []
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "commands.log", "a", encoding="utf-8") as log:
        log.write(f"\n# config sweep {datetime.now().isoformat(timespec='seconds')}\n")
        for b in BLOCKS:
            print(f"  cuda_classic block{b}x{b}")
            rec = run_one([str(ROOT / "cuda_classic" / EXE), *base, "--block", str(b)], log)
            if rec: rows.append(rec)
        for tn in TNS:
            print(f"  cuda_tile tile1D_{tn}")
            rec = run_one([str(ROOT / "cuda_tile" / EXE), *base, "--tn", str(tn)], log)
            if rec: rows.append(rec)
        for tn in TNS:
            print(f"  cutile_py tile1D_{tn}")
            rec = run_one([sys.executable, str(ROOT / "cutile_py" / "stage.py"), *base, "--tn", str(tn)], log)
            if rec: rows.append(rec)

    if not rows:
        print("Sin resultados.")
        return
    df = pd.DataFrame(rows)
    df["megapixels"] = (df["width"] * df["height"]) / 1e6
    df["throughput_mpx_s"] = df["megapixels"] / (df["total_ms_mean"] / 1000.0)
    out = RESULTS / "config_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\nOK -> {out}  ({len(df)} filas)")
    print(df[["version", "tile", "gaussian_ms", "kernel_ms", "total_ms_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
