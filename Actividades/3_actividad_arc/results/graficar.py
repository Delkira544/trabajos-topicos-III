#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_results.py
================
Genera todas las figuras requeridas para el informe de la Actividad 3
(Algoritmo Genético con CUDA) a partir de los CSV producidos por
GUIAEJECUCION.md:

    exp1_main_design.csv
    exp2_block_size_effect.csv
    exp3_speedup.csv
    exp4_population_effect.csv
    table_feasibility.csv
    table_timing_summary.csv

Cubre las métricas obligatorias del enunciado:
    1.  Tiempo promedio de ejecución total
    2.  Desviación estándar del tiempo total
    3.  Tiempo de kernels CUDA
    4.  Tiempo de transferencia H->D y D->H
    5.  Mejor valor (fitness) factible encontrado
    6.  Mejor fitness obtenido (global, por variante)
    7.  Porcentaje de soluciones factibles
    8.  Speed-up respecto de la versión CPU secuencial
    9.  Comparación CUDA básica vs CUDA optimizada
    10. Efecto del tamaño de bloque
    11. Efecto del tamaño de población
    12. Calidad de solución versus tiempo de ejecución

Uso:
    python plot_results.py
    python plot_results.py --results-dir ../results --output-dir ../figures
    python plot_results.py --format png --no-show

Requisitos: pandas, numpy, matplotlib (sin dependencias adicionales).
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ════════════════════════════════════════════════════════════════════════════
# Configuración global
# ════════════════════════════════════════════════════════════════════════════

VARIANT_ORDER = ["sequential", "cuda_basic", "cuda_optimized"]
VARIANT_LABEL = {
    "sequential": "Secuencial (CPU)",
    "cuda_basic": "CUDA Básico",
    "cuda_optimized": "CUDA Optimizado",
}
VARIANT_COLOR = {
    "sequential": "#4C72B0",
    "cuda_basic": "#DD8452",
    "cuda_optimized": "#55A868",
}

INSTANCE_ORDER = ["small", "medium", "large"]
INSTANCE_LABEL = {
    "small": "Pequeña (100 ítems)",
    "medium": "Mediana (1000 ítems)",
    "large": "Grande (10000 ítems)",
}

POP_ORDER = [512, 1024, 4096]
POP_MARKERSIZE = {512: 35, 1024: 75, 4096: 140}

FILES = {
    "exp1": "exp1_main_design.csv",
    "exp2": "exp2_block_size_effect.csv",
    "exp3": "exp3_speedup.csv",
    "exp4": "exp4_population_effect.csv",
    "feasibility": "table_feasibility.csv",
    "timing": "table_timing_summary.csv",
}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.frameon": False,
})


# ════════════════════════════════════════════════════════════════════════════
# Utilidades
# ════════════════════════════════════════════════════════════════════════════

def load_csv(results_dir, key):
    """Carga uno de los CSV conocidos, o None si no existe."""
    path = os.path.join(results_dir, FILES[key])
    if not os.path.isfile(path):
        print(f"  [AVISO] No se encontró {path}, se omiten las figuras que lo requieren.")
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def present(values, order):
    """Devuelve el subconjunto de `order` presente en `values`, preservando el orden."""
    vals = set(values)
    return [v for v in order if v in vals]


def grouped_bars(ax, x_labels, series, colors=None, yerr=None, width_total=0.82):
    """
    Dibuja barras agrupadas en `ax`.
    x_labels : etiquetas del eje X (n grupos)
    series   : dict {nombre_serie: lista_de_valores}, len == n grupos
    yerr     : dict opcional {nombre_serie: lista_de_errores}
    """
    n_groups = len(x_labels)
    n_series = max(len(series), 1)
    width = width_total / n_series
    x = np.arange(n_groups)

    for i, (name, values) in enumerate(series.items()):
        offset = (i - (n_series - 1) / 2) * width
        err = yerr.get(name) if yerr else None
        color = colors.get(name) if colors else None
        ax.bar(x + offset, values, width=width * 0.92, label=name, color=color,
               yerr=err, capsize=3, edgecolor="white", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    return ax


def save_fig(fig, output_dir, name, fmt, show):
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.{fmt}")
    fig.savefig(path, bbox_inches="tight")
    print(f"  -> guardado: {path}")
    if show:
        plt.show()
    plt.close(fig)


def variant_legend(variants):
    return [Patch(facecolor=VARIANT_COLOR[v], label=VARIANT_LABEL[v]) for v in variants]


# ════════════════════════════════════════════════════════════════════════════
# 1 y 2. Tiempo promedio de ejecución total + desviación estándar
# ════════════════════════════════════════════════════════════════════════════

def plot_avg_time_and_std(df_timing, output_dir, fmt, show):
    """Usa table_timing_summary.csv (ya trae Promedio y Desv.Est)."""
    df = df_timing.rename(columns={
        "Instancia": "instance", "Población": "population", "Variante": "variant",
        "Promedio (ms)": "avg_ms", "Desv.Est (ms)": "std_ms",
    })
    instances = present(df["instance"], INSTANCE_ORDER)
    variants = present(df["variant"], VARIANT_ORDER)

    # --- Figura 1: barras de tiempo promedio ± desviación estándar ---
    fig, axes = plt.subplots(1, len(instances), figsize=(5.2 * len(instances), 4.4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, inst in zip(axes, instances):
        sub = df[df.instance == inst]
        pops = present(sub.population, POP_ORDER)
        series, errs = {}, {}
        for v in variants:
            vsub = sub[sub.variant == v].set_index("population").reindex(pops)
            series[v] = vsub["avg_ms"].values
            errs[v] = vsub["std_ms"].values
        grouped_bars(ax, [str(p) for p in pops], series,
                     colors={v: VARIANT_COLOR[v] for v in variants}, yerr=errs)
        ax.set_title(INSTANCE_LABEL.get(inst, inst))
        ax.set_xlabel("Tamaño de población")
        ax.set_ylabel("Tiempo de ejecución (ms)")
        ax.set_yscale("log")
    fig.legend(handles=variant_legend(variants), loc="upper center", ncol=len(variants),
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Tiempo promedio de ejecución total (± desviación estándar)", y=1.12, fontsize=13)
    save_fig(fig, output_dir, "01_tiempo_promedio_y_std", fmt, show)

    # --- Figura 2: boxplot de la dispersión (desviación estándar visualizada) ---
    # Aquí usamos directamente min/max junto con el promedio para mostrar el rango.
    fig, axes = plt.subplots(1, len(instances), figsize=(5.2 * len(instances), 4.4), sharey=False)
    axes = np.atleast_1d(axes)
    df_full = df  # incluye Min/Max
    for ax, inst in zip(axes, instances):
        sub = df_full[df_full.instance == inst]
        pops = present(sub.population, POP_ORDER)
        x = np.arange(len(pops))
        width = 0.82 / max(len(variants), 1)
        for i, v in enumerate(variants):
            vsub = sub[sub.variant == v].set_index("population").reindex(pops)
            offset = (i - (len(variants) - 1) / 2) * width
            avg = vsub["avg_ms"].values
            lo = avg - vsub["Min (ms)"].values
            hi = vsub["Max (ms)"].values - avg
            ax.errorbar(x + offset, avg, yerr=[lo, hi], fmt="o", color=VARIANT_COLOR[v],
                        capsize=4, label=VARIANT_LABEL[v], markersize=5)
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in pops])
        ax.set_title(INSTANCE_LABEL.get(inst, inst))
        ax.set_xlabel("Tamaño de población")
        ax.set_ylabel("Tiempo de ejecución (ms)")
        ax.set_yscale("log")
    fig.legend(handles=variant_legend(variants), loc="upper center", ncol=len(variants),
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Dispersión del tiempo de ejecución (promedio, mínimo y máximo)", y=1.12, fontsize=13)
    save_fig(fig, output_dir, "02_dispersion_tiempo_min_max", fmt, show)


# ════════════════════════════════════════════════════════════════════════════
# 3. Tiempo de kernels CUDA
# ════════════════════════════════════════════════════════════════════════════

def plot_kernel_times(df_exp1, output_dir, fmt, show):
    cuda = df_exp1[df_exp1.variant.isin(["cuda_basic", "cuda_optimized"])].copy()
    agg = cuda.groupby(["instance_size", "population", "variant"], as_index=False)[
        ["kernel_fitness_ms", "kernel_repro_ms"]].mean()

    instances = present(agg.instance_size, INSTANCE_ORDER)
    variants = present(agg.variant, ["cuda_basic", "cuda_optimized"])

    fig, axes = plt.subplots(1, len(instances), figsize=(5.2 * len(instances), 4.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, inst in zip(axes, instances):
        sub = agg[agg.instance_size == inst]
        pops = present(sub.population, POP_ORDER)
        x = np.arange(len(pops))
        width = 0.82 / max(len(variants), 1)
        for i, v in enumerate(variants):
            vsub = sub[sub.variant == v].set_index("population").reindex(pops)
            offset = (i - (len(variants) - 1) / 2) * width
            fit = vsub["kernel_fitness_ms"].values
            repro = vsub["kernel_repro_ms"].values
            ax.bar(x + offset, fit, width=width * 0.9, color=VARIANT_COLOR[v], edgecolor="white")
            ax.bar(x + offset, repro, width=width * 0.9, bottom=fit, color=VARIANT_COLOR[v],
                   edgecolor="white", hatch="//", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in pops])
        ax.set_title(INSTANCE_LABEL.get(inst, inst))
        ax.set_xlabel("Tamaño de población")
        ax.set_ylabel("Tiempo de kernel (ms)")
    handles = variant_legend(variants) + [
        Patch(facecolor="white", edgecolor="black", label="Kernel fitness"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Kernel reproducción"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Tiempo de kernels CUDA (fitness + reproducción)", y=1.16, fontsize=13)
    save_fig(fig, output_dir, "03_tiempo_kernels_cuda", fmt, show)


# ════════════════════════════════════════════════════════════════════════════
# 4. Tiempo de transferencia Host<->Device
# ════════════════════════════════════════════════════════════════════════════

def plot_transfer_times(df_exp1, output_dir, fmt, show):
    cuda = df_exp1[df_exp1.variant.isin(["cuda_basic", "cuda_optimized"])].copy()
    if "d2h_transfer_ms" not in cuda.columns:
        print("  [AVISO] No hay columna d2h_transfer_ms en exp1, se omite figura 04.")
        return
    agg = cuda.groupby(["instance_size", "population", "variant"], as_index=False)[
        ["h2d_transfer_ms", "d2h_transfer_ms"]].mean()

    instances = present(agg.instance_size, INSTANCE_ORDER)
    variants = present(agg.variant, ["cuda_basic", "cuda_optimized"])

    fig, axes = plt.subplots(1, len(instances), figsize=(5.2 * len(instances), 4.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, inst in zip(axes, instances):
        sub = agg[agg.instance_size == inst]
        pops = present(sub.population, POP_ORDER)
        x = np.arange(len(pops))
        width = 0.82 / max(len(variants), 1)
        for i, v in enumerate(variants):
            vsub = sub[sub.variant == v].set_index("population").reindex(pops)
            offset = (i - (len(variants) - 1) / 2) * width
            h2d = vsub["h2d_transfer_ms"].values
            d2h = vsub["d2h_transfer_ms"].values
            ax.bar(x + offset, h2d, width=width * 0.9, color=VARIANT_COLOR[v], edgecolor="white")
            ax.bar(x + offset, d2h, width=width * 0.9, bottom=h2d, color=VARIANT_COLOR[v],
                   edgecolor="white", hatch="//", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in pops])
        ax.set_title(INSTANCE_LABEL.get(inst, inst))
        ax.set_xlabel("Tamaño de población")
        ax.set_ylabel("Tiempo de transferencia (ms)")
    handles = variant_legend(variants) + [
        Patch(facecolor="white", edgecolor="black", label="Host → Device"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Device → Host"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Tiempo de transferencia Host↔Device", y=1.16, fontsize=13)
    save_fig(fig, output_dir, "04_tiempo_transferencias_h2d_d2h", fmt, show)


# ════════════════════════════════════════════════════════════════════════════
# 5 y 6. Mejor valor (fitness) factible / mejor fitness global
# ════════════════════════════════════════════════════════════════════════════

def plot_best_fitness(df_exp1, output_dir, fmt, show):
    variants = present(df_exp1.variant, VARIANT_ORDER)
    instances = present(df_exp1.instance_size, INSTANCE_ORDER)

    # --- Figura 5: mejor fitness factible promedio por configuración ---
    agg = df_exp1.groupby(["instance_size", "population", "variant"], as_index=False)[
        "best_fitness"].agg(["mean", "std"]).reset_index()

    fig, axes = plt.subplots(1, len(instances), figsize=(5.2 * len(instances), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, inst in zip(axes, instances):
        sub = agg[agg.instance_size == inst]
        pops = present(sub.population, POP_ORDER)
        series, errs = {}, {}
        for v in variants:
            vsub = sub[sub.variant == v].set_index("population").reindex(pops)
            series[v] = vsub["mean"].values
            errs[v] = vsub["std"].values
        grouped_bars(ax, [str(p) for p in pops], series,
                     colors={v: VARIANT_COLOR[v] for v in variants}, yerr=errs)
        ax.set_title(INSTANCE_LABEL.get(inst, inst))
        ax.set_xlabel("Tamaño de población")
        ax.set_ylabel("Mejor fitness factible")
    fig.legend(handles=variant_legend(variants), loc="upper center", ncol=len(variants),
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Mejor valor (fitness) factible encontrado por configuración", y=1.12, fontsize=13)
    save_fig(fig, output_dir, "05_mejor_fitness_factible_por_config", fmt, show)

    # --- Figura 6: mejor fitness global por variante ---
    fig, ax = plt.subplots(figsize=(6, 4.4))
    best_global = df_exp1.groupby("variant")["best_fitness"].max().reindex(variants)
    bars = ax.bar([VARIANT_LABEL[v] for v in variants], best_global.values,
                  color=[VARIANT_COLOR[v] for v in variants], edgecolor="white")
    for b, val in zip(bars, best_global.values):
        ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mejor fitness obtenido (global)")
    ax.set_title("Mejor fitness obtenido por variante (todas las instancias/poblaciones)")
    save_fig(fig, output_dir, "06_mejor_fitness_global_por_variante", fmt, show)


# ════════════════════════════════════════════════════════════════════════════
# 7. Porcentaje de soluciones factibles
# ════════════════════════════════════════════════════════════════════════════

def plot_feasibility(df_feas, output_dir, fmt, show):
    df = df_feas.rename(columns={
        "Instancia": "instance", "Población": "population", "Variante": "variant",
        "Factibles (%)": "feasible_pct",
    })
    instances = present(df.instance, INSTANCE_ORDER)
    variants = present(df.variant, VARIANT_ORDER)

    fig, axes = plt.subplots(1, len(instances), figsize=(5.2 * len(instances), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, inst in zip(axes, instances):
        sub = df[df.instance == inst]
        pops = present(sub.population, POP_ORDER)
        series = {}
        for v in variants:
            vsub = sub[sub.variant == v].set_index("population").reindex(pops)
            series[v] = vsub["feasible_pct"].values
        grouped_bars(ax, [str(p) for p in pops], series, colors={v: VARIANT_COLOR[v] for v in variants})
        ax.set_title(INSTANCE_LABEL.get(inst, inst))
        ax.set_xlabel("Tamaño de población")
        ax.set_ylabel("Soluciones factibles (%)")
        ax.set_ylim(0, 100)
    fig.legend(handles=variant_legend(variants), loc="upper center", ncol=len(variants),
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Porcentaje de soluciones factibles en la población final", y=1.12, fontsize=13)
    save_fig(fig, output_dir, "07_porcentaje_factibilidad", fmt, show)


# ════════════════════════════════════════════════════════════════════════════
# 8. Speed-up respecto de la versión secuencial
# ════════════════════════════════════════════════════════════════════════════

def plot_speedup(df_exp3, output_dir, fmt, show):
    instances = present(df_exp3.instance, INSTANCE_ORDER)

    fig, axes = plt.subplots(1, len(instances), figsize=(5.2 * len(instances), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    series_colors = {"speedup_basic": VARIANT_COLOR["cuda_basic"], "speedup_opt": VARIANT_COLOR["cuda_optimized"]}
    series_labels = {"speedup_basic": "CUDA Básico / Secuencial", "speedup_opt": "CUDA Optimizado / Secuencial"}
    for ax, inst in zip(axes, instances):
        sub = df_exp3[df_exp3.instance == inst]
        pops = present(sub.population, POP_ORDER)
        sub_idx = sub.set_index("population").reindex(pops)
        series = {"speedup_basic": sub_idx["speedup_basic"].values,
                  "speedup_opt": sub_idx["speedup_opt"].values}
        grouped_bars(ax, [str(p) for p in pops], series, colors=series_colors)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title(INSTANCE_LABEL.get(inst, inst))
        ax.set_xlabel("Tamaño de población")
        ax.set_ylabel("Speed-up (×)")
    handles = [Patch(facecolor=series_colors[k], label=series_labels[k]) for k in series_colors]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Speed-up respecto de la versión CPU secuencial", y=1.12, fontsize=13)
    save_fig(fig, output_dir, "08_speedup_vs_secuencial", fmt, show)


# ════════════════════════════════════════════════════════════════════════════
# 9. Comparación CUDA básica vs CUDA optimizada
# ════════════════════════════════════════════════════════════════════════════

def plot_cuda_basic_vs_optimized(df_exp1, output_dir, fmt, show):
    cuda = df_exp1[df_exp1.variant.isin(["cuda_basic", "cuda_optimized"])].copy()
    cuda["kernel_total_ms"] = cuda["kernel_fitness_ms"] + cuda["kernel_repro_ms"]
    instances = present(cuda.instance_size, INSTANCE_ORDER)

    fig, axes = plt.subplots(2, len(instances), figsize=(5.2 * len(instances), 8.4), sharex=False)
    for col, inst in enumerate(instances):
        sub = cuda[cuda.instance_size == inst]
        pops = present(sub.population, POP_ORDER)

        for row, (metric, ylabel) in enumerate([
            ("wall_time_ms", "Tiempo total de ejecución (ms)"),
            ("kernel_total_ms", "Tiempo total de kernels (ms)"),
        ]):
            ax = axes[row, col] if len(instances) > 1 else axes[row]
            positions = []
            data = []
            colors = []
            tick_pos = []
            x = 0
            for p in pops:
                tick_pos.append(x + 0.5)
                for v in ["cuda_basic", "cuda_optimized"]:
                    vals = sub[(sub.population == p) & (sub.variant == v)][metric].values
                    positions.append(x)
                    data.append(vals)
                    colors.append(VARIANT_COLOR[v])
                    x += 1
                x += 0.6
            bp = ax.boxplot(data, positions=positions, widths=0.8, patch_artist=True, showfliers=False)
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.85)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels([str(p) for p in pops])
            ax.set_xlabel("Tamaño de población")
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(INSTANCE_LABEL.get(inst, inst))
    fig.legend(handles=variant_legend(["cuda_basic", "cuda_optimized"]), loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("CUDA Básico vs. CUDA Optimizado: tiempo total y tiempo de kernels", y=1.06, fontsize=13)
    save_fig(fig, output_dir, "09_cuda_basico_vs_optimizado", fmt, show)


# ════════════════════════════════════════════════════════════════════════════
# 10. Efecto del tamaño de bloque
# ════════════════════════════════════════════════════════════════════════════

def plot_block_size_effect(df_exp2, output_dir, fmt, show):
    variants = present(df_exp2.variant, ["cuda_basic", "cuda_optimized"])
    blocks = sorted(df_exp2.block_size.unique())
    inst = df_exp2.instance.iloc[0] if "instance" in df_exp2.columns else ""
    pop = df_exp2.population.iloc[0] if "population" in df_exp2.columns else ""

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # Panel 1: tiempo total
    ax = axes[0]
    for v in variants:
        sub = df_exp2[df_exp2.variant == v].groupby("block_size")["wall_time_ms"].agg(["mean", "std"]).reindex(blocks)
        ax.errorbar(blocks, sub["mean"], yerr=sub["std"], marker="o", capsize=4,
                    color=VARIANT_COLOR[v], label=VARIANT_LABEL[v])
    ax.set_xlabel("Tamaño de bloque (hilos)")
    ax.set_ylabel("Tiempo total de ejecución (ms)")
    ax.set_xticks(blocks)
    ax.set_title("Tiempo total")

    # Panel 2: tiempo total de kernels
    ax = axes[1]
    df_exp2 = df_exp2.copy()
    df_exp2["kernel_total_ms"] = df_exp2["kernel_fitness_ms"] + df_exp2["kernel_repro_ms"]
    for v in variants:
        sub = df_exp2[df_exp2.variant == v].groupby("block_size")["kernel_total_ms"].agg(["mean", "std"]).reindex(blocks)
        ax.errorbar(blocks, sub["mean"], yerr=sub["std"], marker="o", capsize=4,
                    color=VARIANT_COLOR[v], label=VARIANT_LABEL[v])
    ax.set_xlabel("Tamaño de bloque (hilos)")
    ax.set_ylabel("Tiempo de kernels (ms)")
    ax.set_xticks(blocks)
    ax.set_title("Tiempo de kernels CUDA")

    # Panel 3: overhead de transferencia (%)
    ax = axes[2]
    if "transfer_overhead_pct" in df_exp2.columns:
        for v in variants:
            sub = df_exp2[df_exp2.variant == v].groupby("block_size")["transfer_overhead_pct"].agg(
                ["mean", "std"]).reindex(blocks)
            ax.errorbar(blocks, sub["mean"], yerr=sub["std"], marker="o", capsize=4,
                        color=VARIANT_COLOR[v], label=VARIANT_LABEL[v])
        ax.set_ylabel("Overhead de transferencia (%)")
    ax.set_xlabel("Tamaño de bloque (hilos)")
    ax.set_xticks(blocks)
    ax.set_title("Overhead de transferencia H↔D")

    fig.legend(handles=variant_legend(variants), loc="upper center", ncol=len(variants),
               bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(f"Efecto del tamaño de bloque (instancia={inst}, población={pop})", y=1.16, fontsize=13)
    save_fig(fig, output_dir, "10_efecto_tamano_bloque", fmt, show)


# ════════════════════════════════════════════════════════════════════════════
# 11. Efecto del tamaño de población
# ════════════════════════════════════════════════════════════════════════════

def plot_population_effect(df_exp1, df_exp4, output_dir, fmt, show):
    # --- 11a: a través de las 3 instancias, usando exp1 ---
    instances = present(df_exp1.instance_size, INSTANCE_ORDER)
    variants = present(df_exp1.variant, VARIANT_ORDER)

    fig, axes = plt.subplots(1, len(instances), figsize=(5.2 * len(instances), 4.4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, inst in zip(axes, instances):
        sub = df_exp1[df_exp1.instance_size == inst]
        for v in variants:
            agg = sub[sub.variant == v].groupby("population")["wall_time_ms"].agg(["mean", "std"])
            pops = present(agg.index, POP_ORDER)
            agg = agg.reindex(pops)
            ax.errorbar(pops, agg["mean"], yerr=agg["std"], marker="o", capsize=4,
                        color=VARIANT_COLOR[v], label=VARIANT_LABEL[v])
        ax.set_xscale("log", base=2)
        ax.set_xticks(POP_ORDER)
        ax.set_xticklabels([str(p) for p in POP_ORDER])
        ax.set_yscale("log")
        ax.set_xlabel("Tamaño de población")
        ax.set_ylabel("Tiempo total de ejecución (ms)")
        ax.set_title(INSTANCE_LABEL.get(inst, inst))
    fig.legend(handles=variant_legend(variants), loc="upper center", ncol=len(variants),
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Efecto del tamaño de población en el tiempo total", y=1.12, fontsize=13)
    save_fig(fig, output_dir, "11a_efecto_poblacion_tiempo_total", fmt, show)

    # --- 11b: detalle a nivel de kernel para la instancia grande (exp4) ---
    if df_exp4 is None:
        return
    variants4 = present(df_exp4.variant, VARIANT_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axes[0]
    for v in variants4:
        agg = df_exp4[df_exp4.variant == v].groupby("population")["wall_time_ms"].mean()
        pops = present(agg.index, POP_ORDER)
        ax.plot(pops, agg.reindex(pops), marker="o", color=VARIANT_COLOR[v], label=VARIANT_LABEL[v])
    ax.set_yscale("log")
    ax.set_xticks(POP_ORDER)
    ax.set_xlabel("Tamaño de población")
    ax.set_ylabel("Tiempo total de ejecución (ms)")
    ax.set_title("Tiempo total (instancia grande)")

    ax = axes[1]
    cuda_variants4 = [v for v in variants4 if v != "sequential"]
    for v in cuda_variants4:
        agg = df_exp4[df_exp4.variant == v].groupby("population")["kernel_time_ms"].mean()
        pops = present(agg.index, POP_ORDER)
        ax.plot(pops, agg.reindex(pops), marker="o", color=VARIANT_COLOR[v], label=VARIANT_LABEL[v])
    ax.set_xticks(POP_ORDER)
    ax.set_xlabel("Tamaño de población")
    ax.set_ylabel("Tiempo de kernels (ms)")
    ax.set_title("Tiempo de kernels (instancia grande)")

    fig.legend(handles=variant_legend(variants4), loc="upper center", ncol=len(variants4),
               bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Efecto del tamaño de población — detalle instancia grande", y=1.16, fontsize=13)
    save_fig(fig, output_dir, "11b_efecto_poblacion_detalle_large", fmt, show)


# ════════════════════════════════════════════════════════════════════════════
# 12. Calidad de solución versus tiempo de ejecución
# ════════════════════════════════════════════════════════════════════════════

def plot_quality_vs_time(df_exp1, output_dir, fmt, show):
    instances = present(df_exp1.instance_size, INSTANCE_ORDER)
    variants = present(df_exp1.variant, VARIANT_ORDER)

    fig, axes = plt.subplots(1, len(instances), figsize=(5.4 * len(instances), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, inst in zip(axes, instances):
        sub = df_exp1[df_exp1.instance_size == inst]
        for v in variants:
            vsub = sub[sub.variant == v]
            sizes = vsub["population"].map(POP_MARKERSIZE).fillna(60)
            ax.scatter(vsub["wall_time_ms"], vsub["best_fitness"], s=sizes,
                       color=VARIANT_COLOR[v], alpha=0.65, edgecolor="white", linewidth=0.5,
                       label=VARIANT_LABEL[v])
        ax.set_xscale("log")
        ax.set_xlabel("Tiempo total de ejecución (ms, escala log)")
        ax.set_ylabel("Mejor fitness factible")
        ax.set_title(INSTANCE_LABEL.get(inst, inst))

    color_handles = variant_legend(variants)
    size_handles = [Line2D([0], [0], marker="o", color="grey", linestyle="",
                           markersize=np.sqrt(POP_MARKERSIZE[p]), label=f"Población={p}")
                    for p in POP_ORDER]
    fig.legend(handles=color_handles + size_handles, loc="upper center",
               ncol=len(color_handles) + len(size_handles), bbox_to_anchor=(0.5, 1.1))
    fig.suptitle("Calidad de la solución versus tiempo de ejecución", y=1.18, fontsize=13)
    save_fig(fig, output_dir, "12_calidad_vs_tiempo", fmt, show)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Genera las figuras del informe a partir de los CSV de resultados.")
    parser.add_argument("--results-dir", default=".", help="Carpeta que contiene los CSV de resultados (default: '.')")
    parser.add_argument("--output-dir", default="figures", help="Carpeta de salida para las figuras (default: 'figures')")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"], help="Formato de las figuras (default: pdf)")
    parser.add_argument("--no-show", action="store_true", help="No mostrar las figuras en pantalla (solo guardarlas)")
    args = parser.parse_args()

    show = not args.no_show

    print(f"Leyendo CSV desde: {os.path.abspath(args.results_dir)}")
    print(f"Guardando figuras en: {os.path.abspath(args.output_dir)} (formato .{args.format})\n")

    df_exp1 = load_csv(args.results_dir, "exp1")
    df_exp2 = load_csv(args.results_dir, "exp2")
    df_exp3 = load_csv(args.results_dir, "exp3")
    df_exp4 = load_csv(args.results_dir, "exp4")
    df_feas = load_csv(args.results_dir, "feasibility")
    df_timing = load_csv(args.results_dir, "timing")

    if df_timing is not None:
        print("[1-2] Tiempo promedio de ejecución y desviación estándar...")
        plot_avg_time_and_std(df_timing, args.output_dir, args.format, show)

    if df_exp1 is not None:
        print("[3] Tiempo de kernels CUDA...")
        plot_kernel_times(df_exp1, args.output_dir, args.format, show)

        print("[4] Tiempo de transferencia Host<->Device...")
        plot_transfer_times(df_exp1, args.output_dir, args.format, show)

        print("[5-6] Mejor fitness factible / mejor fitness global...")
        plot_best_fitness(df_exp1, args.output_dir, args.format, show)

    if df_feas is not None:
        print("[7] Porcentaje de soluciones factibles...")
        plot_feasibility(df_feas, args.output_dir, args.format, show)

    if df_exp3 is not None:
        print("[8] Speed-up respecto de la versión secuencial...")
        plot_speedup(df_exp3, args.output_dir, args.format, show)

    if df_exp1 is not None:
        print("[9] Comparación CUDA básico vs CUDA optimizado...")
        plot_cuda_basic_vs_optimized(df_exp1, args.output_dir, args.format, show)

    if df_exp2 is not None:
        print("[10] Efecto del tamaño de bloque...")
        plot_block_size_effect(df_exp2, args.output_dir, args.format, show)

    if df_exp1 is not None:
        print("[11] Efecto del tamaño de población...")
        plot_population_effect(df_exp1, df_exp4, args.output_dir, args.format, show)

        print("[12] Calidad de solución versus tiempo de ejecución...")
        plot_quality_vs_time(df_exp1, args.output_dir, args.format, show)

    print("\nListo. Todas las figuras disponibles fueron generadas.")


if __name__ == "__main__":
    main()