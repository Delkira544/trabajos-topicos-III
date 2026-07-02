"""Orquestador de benchmark: ejecuta cada versión, agrega tiempos y escribe el CSV.

NO mide nada por sí mismo: cada versión imprime su JSON de tiempos (medidos con
perf_counter en CPU y con CUDA Events en GPU) y aquí solo se parsea, se calcula
throughput + speed-up y se vuelca a results/results.csv. Cada comando ejecutado se
registra en results/commands.log (reproducibilidad, sección 6 de la pauta).

Uso:
  python bench/run_bench.py --versions cpu --reps 10
  python bench/run_bench.py --versions cpu cuda_classic cuda_tile cutile_py --reps 10
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
EXE = "pipeline.exe" if os.name == "nt" else "pipeline"   # binario según plataforma
sys.path.insert(0, str(ROOT / "bench"))
from schema import COLUMNS  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"

# Matriz experimental mínima (sección 7): 4 imágenes, 2 gaussian, 2 resize.
IMAGES = ["small_512", "medium_2048", "large_4096", "nondiv_1537x1021"]
GAUSS = [(5, 1.0), (9, 2.0)]
SCALES = [0.5, 1.75]

# Cómo invocar cada versión. CPU ya está implementada; las CUDA se enchufan aquí
# cuando existan (mismo contrato: imprimir 1 línea JSON de tiempos en stdout).
def cmd_for(version: str, image: Path, ksize: int, sigma: float, scale: float, reps: int):
    if version == "cpu":
        return [sys.executable, str(ROOT / "cpu" / "pipeline_cpu.py"),
                "--image", str(image), "--ksize", str(ksize), "--sigma", str(sigma),
                "--scale", str(scale), "--reps", str(reps), "--save"]
    if version == "cutile_py":
        return [sys.executable, str(ROOT / "cutile_py" / "stage.py"),
                "--image", str(image), "--ksize", str(ksize), "--sigma", str(sigma),
                "--scale", str(scale), "--reps", str(reps)]
    if version == "cuda_classic":
        return [str(ROOT / "cuda_classic" / EXE),
                "--image", str(image), "--ksize", str(ksize), "--sigma", str(sigma),
                "--scale", str(scale), "--reps", str(reps)]
    if version == "cuda_tile":
        return [str(ROOT / "cuda_tile" / EXE),
                "--image", str(image), "--ksize", str(ksize), "--sigma", str(sigma),
                "--scale", str(scale), "--reps", str(reps)]
    raise ValueError(version)


def run_one(cmd: list[str], log) -> dict | None:
    log.write(" ".join(cmd) + "\n")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        print(f"  [skip] ejecutable no encontrado: {cmd[0]}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"  [error] {cmd[0]}\n{e.stderr.strip()[:300]}")
        return None
    # toma la última línea no vacía de stdout como JSON
    line = [l for l in out.stdout.splitlines() if l.strip()][-1]
    return json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", default=["cpu"],
                    choices=["cpu", "cuda_classic", "cuda_tile", "cutile_py"])
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    rows: list[dict] = []
    with open(RESULTS / "commands.log", "a", encoding="utf-8") as log:
        log.write(f"\n# run {datetime.now().isoformat(timespec='seconds')}\n")
        for version in args.versions:
            for img_name in IMAGES:
                image = DATA / f"{img_name}.png"
                if not image.exists():
                    print(f"  [skip] falta {image.name} (corre data/generate_images.py)")
                    continue
                for ksize, sigma in GAUSS:
                    for scale in SCALES:
                        print(f"  {version:13s} {img_name:18s} k{ksize}_s{sigma} x{scale}")
                        rec = run_one(cmd_for(version, image, ksize, sigma, scale, args.reps), log)
                        if rec:
                            rows.append(rec)

    if not rows:
        print("Sin resultados.")
        return

    df = pd.DataFrame(rows)
    df["megapixels"] = (df["width"] * df["height"]) / 1e6
    # throughput: MPx procesados / tiempo total (s)
    df["throughput_mpx_s"] = df["megapixels"] / (df["total_ms_mean"] / 1000.0)

    # speed-up vs CPU para misma (image, gauss, scale)
    cpu = (df[df["version"] == "cpu"]
           .set_index(["image", "gauss", "scale"])["total_ms_mean"])
    def speedup(r):
        key = (r["image"], r["gauss"], r["scale"])
        base = cpu.get(key)
        return round(base / r["total_ms_mean"], 3) if base else None
    df["speedup_vs_cpu"] = df.apply(speedup, axis=1)

    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[COLUMNS]

    out_csv = RESULTS / "results.csv"
    if out_csv.exists():   # respaldo: nunca perder una corrida anterior
        backup = RESULTS / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        out_csv.rename(backup)
        print(f"(respaldo de la corrida anterior -> {backup.name})")
    df.to_csv(out_csv, index=False)
    print(f"\nOK -> {out_csv}  ({len(df)} filas)")


if __name__ == "__main__":
    main()
