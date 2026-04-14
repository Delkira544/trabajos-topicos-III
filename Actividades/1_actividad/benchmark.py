"""
Benchmark comparativo: AES-CTR secuencial vs paralelo.

Mide el tiempo de cifrado para distintos números de workers y tamaños de archivo.
Cada combinación se repite N veces (default: 10) y se reporta el promedio,
lo que reduce el impacto del ruido del sistema operativo en los resultados.
Los archivos cifrados se guardan en encrypted/ y los descifrados en decrypted/.
Genera tabla CSV y gráficos de tiempo promedio y speedup.

Uso:
    python benchmark.py
    python benchmark.py --workers 1,2,4,8 --chunk-size 2097152
    python benchmark.py --files info_clav_10.csv,info_clav_100.csv --passphrase mi-clave
    python benchmark.py --repetitions 5

Dependencias:
    pip install pycryptodome psutil pandas matplotlib
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

import psutil
import pandas as pd
import matplotlib.pyplot as plt

# Importar funciones de cifrado desde los módulos de la actividad
sys.path.insert(0, str(Path(__file__).parent))
from cifrado_aes_ctr_secuencial import transform_file_ctr
from cifrado_aes_ctr_paralelo import transform_file_ctr_parallel

DEFAULT_WORKERS_LIST = [1, 2, 3, 4, 6, 8, 12, 16, 32]
DEFAULT_CHUNK_SIZE   = 1024 * 1024   # 1 MiB
DEFAULT_PASSPHRASE   = "clave-demo"
DEFAULT_FILES        = ["data/info_clav_10.csv", "data/info_clav_100.csv", "data/info_clav_1024.csv"]
DEFAULT_REPETITIONS  = 10

DIR_DATA      = "data"
DIR_ENCRYPTED = "encrypted"
DIR_DECRYPTED = "decrypted"


# ──────────────────────────────────────────────
# Información de hardware
# ──────────────────────────────────────────────

def print_hardware_info() -> None:
    ram_gib = psutil.virtual_memory().total / (1024 ** 3)
    info = [
        ("Sistema operativo",  platform.system() + " " + platform.release()),
        ("Procesador",         platform.processor() or platform.machine()),
        ("Núcleos físicos",    psutil.cpu_count(logical=False)),
        ("Núcleos lógicos",    os.cpu_count()),
        ("RAM total (GiB)",    f"{ram_gib:.2f}"),
        ("Python",             platform.python_version()),
    ]
    col_w = max(len(k) for k, _ in info) + 2
    sep = "-" * (col_w + 30)
    print("\n" + sep)
    print(" INFORMACION DE HARDWARE")
    print(sep)
    for key, val in info:
        print(f"  {key:<{col_w}}{val}")
    print(sep + "\n")


# ──────────────────────────────────────────────
# Bucle de benchmark
# ──────────────────────────────────────────────

def _media(valores: list[float]) -> float:
    """Retorna la media aritmética de una lista de floats."""
    return sum(valores) / len(valores)


def run_benchmark(
    files: list[str],
    workers_list: list[int],
    chunk_size: int,
    passphrase: str,
    repetitions: int,
) -> list[dict]:
    results: list[dict] = []

    # Crear carpetas de salida para cifrados y descifrados
    os.makedirs(DIR_ENCRYPTED, exist_ok=True)
    os.makedirs(DIR_DECRYPTED, exist_ok=True)

    for filepath in files:
        if not Path(filepath).exists():
            print(f"[ADVERTENCIA] Archivo no encontrado, se omite: {filepath}")
            continue

        stem = Path(filepath).stem
        size_mb = Path(filepath).stat().st_size / (1024 ** 2)
        print(f"\n-- Archivo: {filepath}  ({size_mb:.1f} MiB) | repeticiones={repetitions} --")

        # — Medicion secuencial: N repeticiones para promediar —
        seq_out = os.path.join(DIR_ENCRYPTED, f"{stem}_seq.enc")
        tiempos_seq: list[float] = []
        for i in range(repetitions):
            t = transform_file_ctr(filepath, seq_out, passphrase)
            tiempos_seq.append(t)
        t_seq = _media(tiempos_seq)
        print(f"  [Secuencial]  t_prom={t_seq:.6f}s  min={min(tiempos_seq):.6f}s  max={max(tiempos_seq):.6f}s")

        # — Medición paralela por número de workers: N repeticiones —
        for n_workers in workers_list:
            par_out = os.path.join(DIR_ENCRYPTED, f"{stem}_par_w{n_workers}.enc")
            tiempos_par: list[float] = []
            try:
                for i in range(repetitions):
                    t = transform_file_ctr_parallel(
                        filepath, par_out, passphrase,
                        n_workers=n_workers, chunk_size=chunk_size,
                    )
                    tiempos_par.append(t)
                t_par      = _media(tiempos_par)
                speedup    = t_seq / t_par if t_par > 0 else float("inf")
                eficiencia = speedup / n_workers
                print(
                    f"  [Paralelo]  workers={n_workers:>2d} | "
                    f"t_prom={t_par:.6f}s | speedup={speedup:.3f}x | efic={eficiencia:.3f}"
                )
            except Exception as exc:
                print(f"  [ERROR] workers={n_workers}: {exc}")
                t_par      = float("nan")
                speedup    = float("nan")
                eficiencia = float("nan")
                tiempos_par = [float("nan")] * repetitions

            results.append({
                "archivo":      Path(filepath).name,
                "tam_MB":       round(size_mb, 2),
                "workers":      n_workers,
                "repeticiones": repetitions,
                "t_seq_prom_s": round(t_seq, 6),
                "t_seq_min_s":  round(min(tiempos_seq), 6),
                "t_seq_max_s":  round(max(tiempos_seq), 6),
                "t_par_prom_s": round(t_par, 6),
                "t_par_min_s":  round(min(tiempos_par), 6),
                "t_par_max_s":  round(max(tiempos_par), 6),
                "speedup":      round(speedup, 4),
                "eficiencia":   round(eficiencia, 4),
            })

    return results


# ──────────────────────────────────────────────
# Gráficos
# ──────────────────────────────────────────────

def plot_results(df: pd.DataFrame, output_dir: str) -> None:
    archivos = df["archivo"].unique()

    # — Grafico 1: Tiempo promedio paralelo vs workers —
    fig, ax = plt.subplots(figsize=(9, 5))
    for archivo in archivos:
        sub = df[df["archivo"] == archivo].sort_values("workers")
        t_seq_prom = sub["t_seq_prom_s"].iloc[0]
        t_seq_min  = sub["t_seq_min_s"].iloc[0]
        t_seq_max  = sub["t_seq_max_s"].iloc[0]
        color = ax._get_lines.get_next_color()
        ax.plot(sub["workers"], sub["t_par_prom_s"], marker="o", color=color, label=archivo)
        ax.axhline(t_seq_prom, linestyle="--", color=color, alpha=0.7,
                   label=f"Sec. {archivo} (prom)")
        ax.axhspan(t_seq_min, t_seq_max, color=color, alpha=0.10)
    repeticiones = df["repeticiones"].iloc[0]
    ax.set_xlabel("Numero de workers")
    ax.set_ylabel("Tiempo promedio (s)")
    ax.set_title(
        f"Tiempo AES-CTR paralelo vs workers\n"
        f"(promedio de {repeticiones} repeticiones por configuracion)"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path_tiempo = os.path.join(output_dir, "benchmark_tiempo.png")
    fig.savefig(path_tiempo, dpi=150)
    plt.close(fig)
    print(f"  Grafico guardado: {path_tiempo}")

    # — Grafico 2: Speedup vs workers —
    fig, ax = plt.subplots(figsize=(9, 5))
    for archivo in archivos:
        sub = df[df["archivo"] == archivo].sort_values("workers")
        ax.plot(sub["workers"], sub["speedup"], marker="o", label=archivo)
    ax.set_xlabel("Numero de workers")
    ax.set_ylabel("Speedup (t_seq_prom / t_par_prom)")
    ax.set_title(
        f"Speedup AES-CTR paralelo vs workers\n"
        f"(promedio de {repeticiones} repeticiones por configuracion)"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path_speedup = os.path.join(output_dir, "benchmark_speedup.png")
    fig.savefig(path_speedup, dpi=150)
    plt.close(fig)
    print(f"  Grafico guardado: {path_speedup}")

    # — Grafico 3: Eficiencia vs workers —
    fig, ax = plt.subplots(figsize=(9, 5))
    for archivo in archivos:
        sub = df[df["archivo"] == archivo].sort_values("workers")
        ax.plot(sub["workers"], sub["eficiencia"], marker="s", label=archivo)
    ax.set_xlabel("Numero de workers")
    ax.set_ylabel("Eficiencia (speedup / workers)")
    ax.set_title(
        f"Eficiencia AES-CTR paralelo vs workers\n"
        f"(promedio de {repeticiones} repeticiones por configuracion)"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path_efic = os.path.join(output_dir, "benchmark_eficiencia.png")
    fig.savefig(path_efic, dpi=150)
    plt.close(fig)
    print(f"  Grafico guardado: {path_efic}")


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark AES-CTR secuencial vs paralelo"
    )
    parser.add_argument(
        "--workers",
        default=",".join(str(w) for w in DEFAULT_WORKERS_LIST),
        help="Lista de workers separados por coma (default: 1,2,3,4,6,8,12,16,32)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        dest="chunk_size",
        help=f"Tamaño de chunk en bytes (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--passphrase",
        default=DEFAULT_PASSPHRASE,
        help=f"Frase de contraseña (default: {DEFAULT_PASSPHRASE})",
    )
    parser.add_argument(
        "--files",
        default=",".join(DEFAULT_FILES),
        help="Rutas de archivos separadas por coma",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        dest="output_dir",
        help="Directorio de salida para CSV y gráficos (default: .)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help=f"Número de repeticiones por medición para promediar (default: {DEFAULT_REPETITIONS})",
    )
    args = parser.parse_args()

    workers_list = [int(w.strip()) for w in args.workers.split(",")]
    files        = [f.strip() for f in args.files.split(",")]

    print_hardware_info()

    results = run_benchmark(
        files=files,
        workers_list=workers_list,
        chunk_size=args.chunk_size,
        passphrase=args.passphrase,
        repetitions=args.repetitions,
    )

    if not results:
        print("\nNo se obtuvieron resultados. Verifica que los archivos existan.")
        sys.exit(1)

    df = pd.DataFrame(results)

    print("\n" + "-" * 70)
    print(" RESULTADOS")
    print("-" * 70)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", "{:.6f}".format)
    cols_mostrar = ["archivo", "tam_MB", "workers", "repeticiones",
                    "t_seq_prom_s", "t_par_prom_s", "speedup", "eficiencia"]
    print(df[cols_mostrar].to_string(index=False))

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV guardado: {csv_path}")

    print("\n Generando graficos...")
    plot_results(df, args.output_dir)


if __name__ == "__main__":
    main()
