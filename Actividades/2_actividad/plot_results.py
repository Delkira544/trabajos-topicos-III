"""
plot_results.py — Genera gráficos PNG desde los CSV de results/

Uso:
    python plot_results.py [--results-dir results] [--out-dir plots]

Por cada combinación (nombre, config) encontrada en results/ genera:
  1. Speedup vs Hilos (una línea por instancia, standard vs islands)
  2. Eficiencia vs Hilos
  3. Tiempo promedio vs Hilos (con barras de error = desv. est.)
  4. Mejor valor factible (barras, standard vs islands por instancia)
  5. Porcentaje de soluciones factibles (barras, standard vs islands)
"""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ─── Colores y estilos ────────────────────────────────────────────────────────
INSTANCE_COLORS = {
    "small":  "#2196F3",
    "medium": "#FF9800",
    "large":  "#E91E63",
}
INSTANCE_MARKERS = {"small": "o", "medium": "s", "large": "^"}
LINESTYLES = {"standard": "-", "islands": "--"}

INSTANCE_ORDER = ["small", "medium", "large"]
INSTANCE_LABELS = {"small": "small (100 ítems)",
                   "medium": "medium (1 000 ítems)",
                   "large": "large (10 000 ítems)"}


def instance_short(path: str) -> str:
    """'data/large' → 'large'"""
    return Path(path).name


def load_bench_csvs(results_dir: Path):
    """
    Carga todos los *_bench_*.csv de results/.
    Devuelve dict: {(nombre, config_id, variant) → DataFrame}
    """
    pattern = re.compile(r"^(.+?)_bench_(standard|islands)_(\d+)\.csv$")
    groups = {}
    for f in results_dir.glob("*.csv"):
        m = pattern.match(f.name)
        if not m:
            continue
        nombre, variant, config = m.group(1), m.group(2), int(m.group(3))
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        df["instance_short"] = df["Instance"].apply(instance_short)
        key = (nombre, config, variant)
        groups[key] = df
    return groups


def combine_standard_islands(groups, nombre, config):
    """Une standard e islands para un mismo (nombre, config)."""
    dfs = {}
    for variant in ("standard", "islands"):
        key = (nombre, config, variant)
        if key in groups:
            df = groups[key].copy()
            df["variant"] = variant
            dfs[variant] = df
    if not dfs:
        return None
    return pd.concat(dfs.values(), ignore_index=True)


# ─── Gráfico 1 — Speedup vs Hilos ────────────────────────────────────────────
def plot_speedup(df: pd.DataFrame, out_path: Path, title_suffix: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle(f"Speedup vs Hilos — {title_suffix}", fontsize=13, fontweight="bold")

    for ax, inst in zip(axes, INSTANCE_ORDER):
        sub = df[df["instance_short"] == inst]
        if sub.empty:
            ax.set_visible(False)
            continue

        # línea ideal
        threads = sorted(sub["Threads"].unique())
        ax.plot(threads, threads, "k:", linewidth=1, label="Ideal", zorder=1)

        for variant in ("standard", "islands"):
            vsub = sub[sub["variant"] == variant].sort_values("Threads")
            if vsub.empty:
                continue
            color = "#1976D2" if variant == "standard" else "#C62828"
            ax.plot(vsub["Threads"], vsub["Speedup"],
                    marker="o", color=color, linewidth=2,
                    linestyle=LINESTYLES[variant],
                    label=variant.capitalize(), zorder=2)

        ax.set_title(INSTANCE_LABELS[inst], fontsize=10)
        ax.set_xlabel("Hilos (T)")
        ax.set_ylabel("Speedup S(T)")
        ax.set_xticks(threads)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out_path}")


# ─── Gráfico 2 — Eficiencia vs Hilos ─────────────────────────────────────────
def plot_efficiency(df: pd.DataFrame, out_path: Path, title_suffix: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(f"Eficiencia vs Hilos — {title_suffix}", fontsize=13, fontweight="bold")

    for ax, inst in zip(axes, INSTANCE_ORDER):
        sub = df[df["instance_short"] == inst]
        if sub.empty:
            ax.set_visible(False)
            continue

        threads = sorted(sub["Threads"].unique())
        ax.axhline(1.0, color="k", linestyle=":", linewidth=1, label="Ideal")

        for variant in ("standard", "islands"):
            vsub = sub[sub["variant"] == variant].sort_values("Threads")
            if vsub.empty:
                continue
            color = "#1976D2" if variant == "standard" else "#C62828"
            eff = vsub["Efficiency"] * 100 if vsub["Efficiency"].max() <= 1.0 else vsub["Efficiency"]
            ax.plot(vsub["Threads"], eff,
                    marker="s", color=color, linewidth=2,
                    linestyle=LINESTYLES[variant],
                    label=variant.capitalize())

        ax.set_title(INSTANCE_LABELS[inst], fontsize=10)
        ax.set_xlabel("Hilos (T)")
        ax.set_ylabel("Eficiencia (%)")
        ax.set_xticks(threads)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
        ax.set_ylim(0, 115)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out_path}")


# ─── Gráfico 3 — Tiempo vs Hilos (con barras de error) ───────────────────────
def plot_time(df: pd.DataFrame, out_path: Path, title_suffix: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Tiempo promedio vs Hilos — {title_suffix}", fontsize=13, fontweight="bold")

    for ax, inst in zip(axes, INSTANCE_ORDER):
        sub = df[df["instance_short"] == inst]
        if sub.empty:
            ax.set_visible(False)
            continue

        threads = sorted(sub["Threads"].unique())
        x = np.arange(len(threads))
        width = 0.35

        for i, variant in enumerate(("standard", "islands")):
            vsub = sub[sub["variant"] == variant].sort_values("Threads")
            if vsub.empty:
                continue
            color = "#42A5F5" if variant == "standard" else "#EF5350"
            times = [vsub[vsub["Threads"] == t]["AvgTime(s)"].values[0]
                     if t in vsub["Threads"].values else 0 for t in threads]
            stds = [vsub[vsub["Threads"] == t]["StdTime(s)"].values[0]
                    if t in vsub["Threads"].values else 0 for t in threads]
            offset = (i - 0.5) * width
            ax.bar(x + offset, times, width, yerr=stds,
                   label=variant.capitalize(), color=color,
                   capsize=4, error_kw={"elinewidth": 1.5})

        ax.set_title(INSTANCE_LABELS[inst], fontsize=10)
        ax.set_xlabel("Hilos (T)")
        ax.set_ylabel("Tiempo (s)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in threads])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out_path}")


# ─── Gráfico 4 — Mejor valor factible (barras) ───────────────────────────────
def plot_best_value(df: pd.DataFrame, out_path: Path, title_suffix: str):
    # Solo con 1 hilo para comparar calidad de solución
    sub1 = df[df["Threads"] == 1].copy()

    instances = [i for i in INSTANCE_ORDER if i in sub1["instance_short"].values]
    x = np.arange(len(instances))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(f"Mejor valor factible (T=1) — {title_suffix}",
                 fontsize=13, fontweight="bold")

    for i, variant in enumerate(("standard", "islands")):
        vsub = sub1[sub1["variant"] == variant]
        values = []
        for inst in instances:
            row = vsub[vsub["instance_short"] == inst]
            values.append(row["BestFeasibleValue"].values[0] if not row.empty else 0)
        color = "#42A5F5" if variant == "standard" else "#EF5350"
        bars = ax.bar(x + (i - 0.5) * width, values, width,
                      label=variant.capitalize(), color=color)
        # etiqueta encima de cada barra
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01,
                        f"{val:,.0f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([INSTANCE_LABELS[i] for i in instances], fontsize=9)
    ax.set_ylabel("Valor total de ítems seleccionados")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out_path}")


# ─── Gráfico 5 — Porcentaje de soluciones factibles ──────────────────────────
def plot_feasible_pct(df: pd.DataFrame, out_path: Path, title_suffix: str):
    sub1 = df[df["Threads"] == 1].copy()

    instances = [i for i in INSTANCE_ORDER if i in sub1["instance_short"].values]
    x = np.arange(len(instances))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(f"% Soluciones factibles (T=1) — {title_suffix}",
                 fontsize=13, fontweight="bold")

    col = "Feasible%"

    for i, variant in enumerate(("standard", "islands")):
        vsub = sub1[sub1["variant"] == variant]
        values = []
        for inst in instances:
            row = vsub[vsub["instance_short"] == inst]
            values.append(row[col].values[0] if not row.empty else 0)
        color = "#42A5F5" if variant == "standard" else "#EF5350"
        bars = ax.bar(x + (i - 0.5) * width, values, width,
                      label=variant.capitalize(), color=color)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([INSTANCE_LABELS[i] for i in instances], fontsize=9)
    ax.set_ylabel("Individuos factibles en última generación (%)")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out_path}")


# ─── Gráfico 6 — Speedup solo islands (todas las instancias en uno) ──────────
def plot_speedup_islands_all(df: pd.DataFrame, out_path: Path, title_suffix: str):
    sub = df[df["variant"] == "islands"]
    if sub.empty:
        return

    threads_all = sorted(sub["Threads"].unique())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(f"Speedup Modelo de Islas — {title_suffix}", fontsize=12, fontweight="bold")

    ax.plot(threads_all, threads_all, "k:", linewidth=1, label="Ideal")

    for inst in INSTANCE_ORDER:
        isub = sub[sub["instance_short"] == inst].sort_values("Threads")
        if isub.empty:
            continue
        ax.plot(isub["Threads"], isub["Speedup"],
                marker=INSTANCE_MARKERS[inst],
                color=INSTANCE_COLORS[inst],
                linewidth=2, label=INSTANCE_LABELS[inst])

    ax.set_xlabel("Hilos (T)")
    ax.set_ylabel("Speedup S(T)")
    ax.set_xticks(threads_all)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Genera gráficos desde results/")
    parser.add_argument("--results-dir", default="results", help="Carpeta con los CSV")
    parser.add_argument("--out-dir", default="plots", help="Carpeta de salida PNG")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    groups = load_bench_csvs(results_dir)
    if not groups:
        print(f"No se encontraron archivos *_bench_*.csv en {results_dir}/")
        sys.exit(1)

    # Agrupar por (nombre, config)
    combos = defaultdict(list)
    for (nombre, config, variant) in groups:
        combos[(nombre, config)].append(variant)

    print(f"Encontradas {len(combos)} combinación(es) nombre×config:\n")
    for (nombre, config), variants in sorted(combos.items()):
        print(f"  {nombre!r}  config={config}  variantes={variants}")

    print()
    for (nombre, config), _ in sorted(combos.items()):
        df = combine_standard_islands(groups, nombre, config)
        if df is None:
            continue

        suffix = f"{nombre} — Config {config}"
        prefix = out_dir / f"{nombre}_config{config}"

        print(f"Generando gráficos para: {suffix}")
        plot_speedup(df, Path(f"{prefix}_speedup.png"), suffix)
        plot_efficiency(df, Path(f"{prefix}_eficiencia.png"), suffix)
        plot_time(df, Path(f"{prefix}_tiempo.png"), suffix)
        plot_best_value(df, Path(f"{prefix}_valor_factible.png"), suffix)
        plot_feasible_pct(df, Path(f"{prefix}_factibles_pct.png"), suffix)
        plot_speedup_islands_all(df, Path(f"{prefix}_speedup_islands.png"), suffix)
        print()

    print(f"Todos los gráficos guardados en: {out_dir}/")


if __name__ == "__main__":
    main()
