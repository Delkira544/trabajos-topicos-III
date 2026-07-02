"""Captura el entorno reproducible exigido por la pauta (§7): hardware, driver,
CUDA Toolkit, nvcc, Python y paquetes -> results/environment.txt

Uso:  python bench/capture_env.py
"""
from __future__ import annotations
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# nvcc instalado por pip: layout distinto en Windows (.venv/Lib) y Linux (.venv/lib/pythonX.Y)
_hits = list(ROOT.glob(".venv/**/nvidia/cu13/bin/nvcc*"))
NVCC = _hits[0] if _hits else Path("nvcc")
OUT = ROOT / "results" / "environment.txt"
IS_WIN = platform.system() == "Windows"

PKGS = ["numpy", "pandas", "pillow", "streamlit", "altair",
        "cuda-tile", "cuda-toolkit", "cupy-cuda13x",
        "nvidia-cuda-nvcc", "nvidia-cuda-runtime", "nvidia-cuda-tileiras",
        "nvidia-cuda-nvrtc", "nvidia-nvvm", "nvidia-nvjitlink"]


def run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"(no disponible: {e})"


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    lines = [
        f"# Entorno experimental — generado {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Sistema operativo",
        f"{platform.system()} {platform.release()} ({platform.version()}), {platform.machine()}",
        "",
        "## GPU y driver (nvidia-smi)",
        run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
             "--format=csv,noheader"]),
        "",
        "## CUDA runtime soportado por el driver",
        next((l for l in run(["nvidia-smi"]).splitlines() if "CUDA Version" in l), "(n/d)").strip(),
        "",
        "## nvcc (compilador CUDA, instalado vía pip en el venv)",
        run([str(NVCC), "--version"]).splitlines()[-2] if NVCC.exists() else "(nvcc no encontrado)",
        "",
        "## Compilador host de nvcc",
        ((run(["cmd", "/c",
               r'call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul && cl']
              ).splitlines() or ["(cl no encontrado)"])[0]) if IS_WIN
        else (run(["gcc", "--version"]).splitlines() or ["(gcc no encontrado)"])[0],
        "",
        "## Python",
        f"{sys.version}  ({sys.executable})",
        "",
        "## Paquetes relevantes (pip)",
    ]
    freeze = run([sys.executable, "-m", "pip", "freeze"])
    wanted = {p.lower() for p in PKGS}
    for line in freeze.splitlines():
        name = line.split("==")[0].lower()
        if name in wanted:
            lines.append(f"  {line}")
    lines += ["", "## Flags de compilación",
              "  cuda_classic: nvcc -O3 -std=c++17 -arch=sm_89 -cudart static",
              "  cuda_tile:    nvcc -O3 -std=c++20 --enable-tile -arch=sm_89 -cudart static"]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"OK -> {OUT}")
    print("\n".join(lines[:14]))


if __name__ == "__main__":
    main()
