#!/usr/bin/env python3
"""
Execute experiments for Actividad 3 — Genetic Algorithm with CUDA
Linux adaptation of GUIAEJECUCION.md (PowerShell → Python)

Usage:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --bin build/run --data data --results results
    python scripts/run_experiments.py --exp-only 1          # skip hardware, run only EXP 1
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import platform
from datetime import datetime

# ─────────────────────────────────────────────
# Constants (matches experimental design table)
# ─────────────────────────────────────────────
INSTANCES = ["small", "medium", "large"]
POPULATIONS = [1024, 4096, 16384]
VARIANTS = ["sequential", "cuda_basic", "cuda_optimized"]
SEEDS = list(range(42, 52))  # 42-51 (10 seeds)
GENERATIONS = 300
DEFAULT_BLOCK_SIZE = 128

# Block sizes for EXP 2
BLOCK_SIZES = [32, 64, 128, 256]
VARIANTS_CUDA = ["cuda_basic", "cuda_optimized"]

# Penalty weights (from penalty tuner calibration)
PEN_WEIGHT = 0.25
PEN_VOLUME = 0.0
PEN_CATEGORY = 0.25
PEN_INCOMP = 0.25
PEN_DEP = 0.25

# Regex patterns for parsing binary output
RE_PATTERNS = {
    "best_fitness": re.compile(r"Best fitness:\s+([\d.\-]+)"),
    "feasible": re.compile(r"Feasible:\s+(Yes|No)"),
    "wall_time_ms": re.compile(r"Wall-clock time \(ms\):\s*(\d+)"),
    "kernel_fitness_ms": re.compile(r"Kernel fitness\s+total \(ms\):\s+([\d.]+)"),
    "kernel_repro_ms": re.compile(r"Kernel repro\s+total \(ms\):\s+([\d.]+)"),
    "h2d_transfer_ms": re.compile(r"Transfer H->D\s+total \(ms\):\s+([\d.]+)"),
    "d2h_transfer_ms": re.compile(r"Transfer D->H\s+total \(ms\):\s+([\d.]+)"),
    "feasible_pct": re.compile(r"Feasible solutions \(%\):\s+([\d.]+)"),
    "init_fitness": re.compile(r"Initial best fitness:\s+([\d.\-]+)"),
    "final_fitness": re.compile(r"Final best fitness:\s+([\d.\-]+)"),
    "improvement": re.compile(r"Improvement:\s+([\d.\-]+)"),
    "transfer_overhead_pct": re.compile(r"Transfer overhead \(%\):\s+([\d.]+)"),
}


def run_binary(bin_path: str, args: list[str], timeout: int = 3600) -> str:
    """Run the binary and return stdout as string."""
    cmd = [bin_path] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            print(f"  [WARN] Binary returned code {result.returncode}: {result.stderr.strip()}")
        return result.stdout
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] Exceeded {timeout}s ({' '.join(cmd[-8:])})")
        return ""


def parse_output(stdout: str) -> dict[str, str]:
    """Extract metrics from binary stdout using regex patterns."""
    data: dict[str, str] = {}
    for key, pattern in RE_PATTERNS.items():
        m = pattern.search(stdout)
        data[key] = m.group(1) if m else ""
    return data


def build_cmd_args(
    data_dir: str,
    instance: str,
    variant: str,
    seed: int,
    population: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> list[str]:
    """Build the CLI arguments list for the run binary."""
    return [
        "-i", os.path.join(data_dir, instance),
        "-v", variant,
        "-t", "1",
        "-s", str(seed),
        "-p", str(population),
        "-g", str(GENERATIONS),
        "--pen-weight", str(PEN_WEIGHT),
        "--pen-volume", str(PEN_VOLUME),
        "--pen-category", str(PEN_CATEGORY),
        "--pen-incomp", str(PEN_INCOMP),
        "--pen-dep", str(PEN_DEP),
        "--block-size", str(block_size),
    ]


# ─────────────────────────────────────────────
# 5.0 Hardware Report
# ─────────────────────────────────────────────
def generate_hardware_report(results_dir: str) -> str:
    """Generate hardware report (CPU, GPU, CUDA, RAM)."""
    path = os.path.join(results_dir, "hardware.txt")
    lines: list[str] = []
    lines.append("=== HARDWARE REPORT ===")
    lines.append("")

    # Date
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Hostname: {platform.node()}")
    lines.append(f"OS: {platform.system()} {platform.release()}")
    lines.append(f"Platform: {platform.platform()}")
    lines.append("")

    # CPU
    lines.append("CPU:")
    try:
        with open("/proc/cpuinfo") as f:
            content = f.read()
        model_match = re.search(r"model name\s+:\s+(.+)", content)
        cores_match = re.search(r"cpu cores\s+:\s+(\d+)", content)
        siblings_match = re.search(r"siblings\s+:\s+(\d+)", content)
        if model_match:
            lines.append(f"  Model: {model_match.group(1)}")
        if cores_match:
            lines.append(f"  Cores: {cores_match.group(1)}")
        if siblings_match:
            lines.append(f"  Threads: {siblings_match.group(1)}")
        # Count physical CPUs
        phys_ids = set(re.findall(r"physical id\s+:\s+(\d+)", content))
        if phys_ids:
            lines.append(f"  Physical CPUs: {len(phys_ids)}")
    except Exception as e:
        lines.append(f"  Error: {e}")
    lines.append("")

    # GPU
    lines.append("GPU:")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                lines.append(f"  {line}")
        else:
            lines.append("  nvidia-smi not available or error")
    except FileNotFoundError:
        lines.append("  nvidia-smi not found (NVIDIA driver not installed?)")
    lines.append("")

    # CUDA version
    lines.append("CUDA Version:")
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                lines.append(f"  {line}")
        else:
            lines.append("  nvcc returned error")
    except FileNotFoundError:
        lines.append("  nvcc not found (CUDA Toolkit not in PATH?)")
    lines.append("")

    # RAM
    lines.append("RAM:")
    try:
        with open("/proc/meminfo") as f:
            for mem_line in f:
                if mem_line.startswith("MemTotal"):
                    lines.append(f"  {mem_line.strip()}")
                    break
    except Exception as e:
        lines.append(f"  Error: {e}")
    lines.append("")

    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  ✓ Hardware report: {path}")
    return path


# ─────────────────────────────────────────────
# EXP 1: Main Design
# ─────────────────────────────────────────────
def run_exp1(bin_path: str, data_dir: str, results_dir: str, timeout: int = 3600) -> str:
    """Run EXP 1 — Variants × Instances × Populations × 10 Seeds."""
    csv_path = os.path.join(results_dir, "exp1_main_design.csv")
    total = len(INSTANCES) * len(POPULATIONS) * len(VARIANTS) * len(SEEDS)
    current = 0

    fieldnames = [
        "instance_size", "population", "variant", "seed",
        "best_fitness", "feasible", "wall_time_ms",
        "kernel_fitness_ms", "kernel_repro_ms",
        "h2d_transfer_ms", "d2h_transfer_ms",
        "feasible_pct", "init_fitness", "final_fitness", "improvement",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for instance in INSTANCES:
            for pop in POPULATIONS:
                for variant in VARIANTS:
                    for seed in SEEDS:
                        current += 1
                        pct = round(current / total * 100, 1)
                        print(f"  [{pct}%] EXP1 -> {instance} | pop={pop} | {variant} | seed={seed}")

                        args = build_cmd_args(data_dir, instance, variant, seed, pop)
                        stdout = run_binary(bin_path, args, timeout=timeout)
                        parsed = parse_output(stdout)

                        row = {
                            "instance_size": instance,
                            "population": str(pop),
                            "variant": variant,
                            "seed": str(seed),
                            "best_fitness": parsed.get("best_fitness", ""),
                            "feasible": parsed.get("feasible", ""),
                            "wall_time_ms": parsed.get("wall_time_ms", ""),
                            "kernel_fitness_ms": parsed.get("kernel_fitness_ms", ""),
                            "kernel_repro_ms": parsed.get("kernel_repro_ms", ""),
                            "h2d_transfer_ms": parsed.get("h2d_transfer_ms", ""),
                            "d2h_transfer_ms": parsed.get("d2h_transfer_ms", ""),
                            "feasible_pct": parsed.get("feasible_pct", ""),
                            "init_fitness": parsed.get("init_fitness", ""),
                            "final_fitness": parsed.get("final_fitness", ""),
                            "improvement": parsed.get("improvement", ""),
                        }
                        writer.writerow(row)

    print(f"\n  ✓ EXP 1 completed: {csv_path} ({current} runs)")
    return csv_path


# ─────────────────────────────────────────────
# EXP 2: Block Size Effect
# ─────────────────────────────────────────────
def run_exp2(bin_path: str, data_dir: str, results_dir: str, timeout: int = 3600) -> str:
    """Run EXP 2 — Block Size effect (large instance, fixed population)."""
    csv_path = os.path.join(results_dir, "exp2_block_size_effect.csv")
    instance = "large"
    population = 4096
    total = len(BLOCK_SIZES) * len(VARIANTS_CUDA) * len(SEEDS)
    current = 0

    fieldnames = [
        "instance", "population", "block_size", "variant", "seed",
        "best_fitness", "feasible", "wall_time_ms",
        "kernel_fitness_ms", "kernel_repro_ms", "transfer_overhead_pct",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for block in BLOCK_SIZES:
            for variant in VARIANTS_CUDA:
                for seed in SEEDS:
                    current += 1
                    pct = round(current / total * 100, 1)
                    print(f"  [{pct}%] EXP2 -> block={block} | {variant} | seed={seed}")

                    args = build_cmd_args(data_dir, instance, variant, seed, population, block_size=block)
                    stdout = run_binary(bin_path, args, timeout=timeout)
                    parsed = parse_output(stdout)

                    row = {
                        "instance": instance,
                        "population": str(population),
                        "block_size": str(block),
                        "variant": variant,
                        "seed": str(seed),
                        "best_fitness": parsed.get("best_fitness", ""),
                        "feasible": parsed.get("feasible", ""),
                        "wall_time_ms": parsed.get("wall_time_ms", ""),
                        "kernel_fitness_ms": parsed.get("kernel_fitness_ms", ""),
                        "kernel_repro_ms": parsed.get("kernel_repro_ms", ""),
                        "transfer_overhead_pct": parsed.get("transfer_overhead_pct", ""),
                    }
                    writer.writerow(row)

    print(f"\n  ✓ EXP 2 completed: {csv_path} ({current} runs)")
    return csv_path


# ─────────────────────────────────────────────
# EXP 3: Speed-up Analysis (from EXP 1 data)
# ─────────────────────────────────────────────
def generate_exp3(exp1_csv: str, results_dir: str) -> str:
    """Derive EXP 3 from EXP 1 data: speedup_basic, speedup_opt."""
    csv_path = os.path.join(results_dir, "exp3_speedup.csv")

    # Read EXP 1 data
    rows: list[dict] = []
    with open(exp1_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    fieldnames = ["instance", "population", "speedup_basic", "speedup_opt"]
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for instance in INSTANCES:
            for pop in POPULATIONS:
                # Sequential times
                seq_times = [
                    float(r["wall_time_ms"])
                    for r in rows
                    if r["instance_size"] == instance
                    and r["population"] == str(pop)
                    and r["variant"] == "sequential"
                    and r["wall_time_ms"] != ""
                ]
                # CUDA basic times
                bas_times = [
                    float(r["wall_time_ms"])
                    for r in rows
                    if r["instance_size"] == instance
                    and r["population"] == str(pop)
                    and r["variant"] == "cuda_basic"
                    and r["wall_time_ms"] != ""
                ]
                # CUDA optimized times
                opt_times = [
                    float(r["wall_time_ms"])
                    for r in rows
                    if r["instance_size"] == instance
                    and r["population"] == str(pop)
                    and r["variant"] == "cuda_optimized"
                    and r["wall_time_ms"] != ""
                ]

                avg_seq = sum(seq_times) / len(seq_times) if seq_times else 0
                avg_bas = sum(bas_times) / len(bas_times) if bas_times else 0
                avg_opt = sum(opt_times) / len(opt_times) if opt_times else 0

                sp_bas = round(avg_seq / avg_bas, 2) if avg_bas > 0 else "N/A"
                sp_opt = round(avg_seq / avg_opt, 2) if avg_opt > 0 else "N/A"

                writer.writerow({
                    "instance": instance,
                    "population": str(pop),
                    "speedup_basic": str(sp_bas),
                    "speedup_opt": str(sp_opt),
                })

    print(f"  ✓ EXP 3 generated: {csv_path}")
    return csv_path


# ─────────────────────────────────────────────
# EXP 4: Population Size Effect (from EXP 1 data)
# ─────────────────────────────────────────────
def generate_exp4(exp1_csv: str, results_dir: str) -> str:
    """Derive EXP 4 from EXP 1 data — filter large, sum kernel times."""
    csv_path = os.path.join(results_dir, "exp4_population_effect.csv")

    rows: list[dict] = []
    with open(exp1_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["instance_size"] == "large":
                rows.append(row)

    fieldnames = [
        "instance", "population", "variant", "seed",
        "best_fitness", "feasible", "wall_time_ms", "kernel_time_ms",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            ktime = 0.0
            if row["kernel_fitness_ms"] != "" and row["kernel_repro_ms"] != "":
                ktime = round(float(row["kernel_fitness_ms"]) + float(row["kernel_repro_ms"]), 2)

            writer.writerow({
                "instance": row["instance_size"],
                "population": row["population"],
                "variant": row["variant"],
                "seed": row["seed"],
                "best_fitness": row["best_fitness"],
                "feasible": row["feasible"],
                "wall_time_ms": row["wall_time_ms"],
                "kernel_time_ms": str(ktime),
            })

    print(f"  ✓ EXP 4 generated: {csv_path}")
    return csv_path


# ─────────────────────────────────────────────
# Summary Tables for Report
# ─────────────────────────────────────────────
def generate_summary_tables(exp1_csv: str, exp3_csv: str, results_dir: str) -> None:
    """Generate aggregated summary tables for the report."""
    rows: list[dict] = []
    with open(exp1_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # ── Table 1: Timing summary ──
    t1_path = os.path.join(results_dir, "table_timing_summary.csv")
    t1_header = "Instancia,Población,Variante,Promedio (ms),Desv.Est (ms),Min (ms),Max (ms)"
    t1_lines: list[str] = [t1_header]

    for instance in INSTANCES:
        for pop in POPULATIONS:
            for variant in VARIANTS:
                times = [
                    float(r["wall_time_ms"])
                    for r in rows
                    if r["instance_size"] == instance
                    and r["population"] == str(pop)
                    and r["variant"] == variant
                    and r["wall_time_ms"] != ""
                ]
                if times:
                    n = len(times)
                    avg = sum(times) / n
                    variance = sum((t - avg) ** 2 for t in times) / n
                    stddev = variance ** 0.5
                    min_t = min(times)
                    max_t = max(times)
                    t1_lines.append(
                        f"{instance},{pop},{variant},"
                        f"{avg:.2f},{stddev:.2f},{min_t:.2f},{max_t:.2f}"
                    )

    with open(t1_path, "w", encoding="utf-8") as f:
        f.write("\n".join(t1_lines) + "\n")
    print(f"  ✓ Table 1 (timing): {t1_path}")

    # ── Table 2: Feasibility summary ──
    t2_path = os.path.join(results_dir, "table_feasibility.csv")
    t2_header = "Instancia,Población,Variante,Factibles (%)"
    t2_lines: list[str] = [t2_header]

    for instance in INSTANCES:
        for pop in POPULATIONS:
            for variant in VARIANTS:
                feasibles = [
                    float(r["feasible_pct"])
                    for r in rows
                    if r["instance_size"] == instance
                    and r["population"] == str(pop)
                    and r["variant"] == variant
                    and r["feasible_pct"] != ""
                ]
                if feasibles:
                    avg_feas = sum(feasibles) / len(feasibles)
                    t2_lines.append(f"{instance},{pop},{variant},{avg_feas:.2f}")

    with open(t2_path, "w", encoding="utf-8") as f:
        f.write("\n".join(t2_lines) + "\n")
    print(f"  ✓ Table 2 (feasibility): {t2_path}")

    # ── Table 3: Speed-up summary ──
    t3_path = os.path.join(results_dir, "table_speedup.csv")
    t3_header = "Instancia,Población,Speed-up CUDA Basic (avg),Speed-up CUDA Opt (avg)"
    t3_lines: list[str] = [t3_header]

    exp3_rows: list[dict] = []
    with open(exp3_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp3_rows.append(row)

    for instance in INSTANCES:
        for pop in POPULATIONS:
            basic_vals = [
                float(r["speedup_basic"])
                for r in exp3_rows
                if r["instance"] == instance
                and r["population"] == str(pop)
                and r["speedup_basic"] not in ("N/A", "")
            ]
            opt_vals = [
                float(r["speedup_opt"])
                for r in exp3_rows
                if r["instance"] == instance
                and r["population"] == str(pop)
                and r["speedup_opt"] not in ("N/A", "")
            ]
            avg_basic = f"{sum(basic_vals) / len(basic_vals):.2f}" if basic_vals else "N/A"
            avg_opt = f"{sum(opt_vals) / len(opt_vals):.2f}" if opt_vals else "N/A"
            t3_lines.append(f"{instance},{pop},{avg_basic},{avg_opt}")

    with open(t3_path, "w", encoding="utf-8") as f:
        f.write("\n".join(t3_lines) + "\n")
    print(f"  ✓ Table 3 (speedup): {t3_path}")

    print(f"\n  ✓ All summary tables in {results_dir}/")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run experiments for Actividad 3 — Genetic Algorithm with CUDA"
    )
    parser.add_argument(
        "--bin", default="build/run",
        help="Path to the run binary (default: build/run)",
    )
    parser.add_argument(
        "--data", default="data",
        help="Path to data directory (default: data)",
    )
    parser.add_argument(
        "--results", default="results",
        help="Path to results directory (default: results)",
    )
    parser.add_argument(
        "--skip-hardware", action="store_true",
        help="Skip hardware report generation",
    )
    parser.add_argument(
        "--exp-only", type=int, choices=[1, 2, 3, 4], default=None,
        help="Run only a single experiment (1-4). EXP 3 and 4 require EXP 1 data.",
    )
    parser.add_argument(
        "--timeout", type=int, default=3600,
        help="Per-run timeout in seconds (default: 3600)",
    )
    args = parser.parse_args()

    # Resolve relative to script location or cwd
    bin_path = os.path.abspath(args.bin)
    data_dir = os.path.abspath(args.data)
    results_dir = os.path.abspath(args.results)

    # Validate binary
    if not os.path.isfile(bin_path):
        print(f"[ERROR] Binary not found: {bin_path}")
        sys.exit(1)
    if not os.access(bin_path, os.X_OK):
        print(f"[ERROR] Binary not executable: {bin_path}")
        sys.exit(1)

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 58)
    print(" Actividad 3 — Genetic Algorithm with CUDA")
    print(" Experiment Runner (Linux / Python)")
    print("=" * 58)
    print(f"  Binary:      {bin_path}")
    print(f"  Data:        {data_dir}")
    print(f"  Results:     {results_dir}")
    print(f"  Generations: {GENERATIONS}")
    print(f"  Seeds:       {SEEDS[0]}-{SEEDS[-1]}")
    print(f"  Timeout:     {args.timeout}s per run")
    print()

    # ── Hardware report ──
    if not args.skip_hardware:
        print("[1/6] Generating hardware report...")
        generate_hardware_report(results_dir)
        print()

    # ── EXP 1 ──
    if args.exp_only is None or args.exp_only == 1:
        print(f"[2/6] Running EXP 1 — Main Design ({len(INSTANCES) * len(POPULATIONS) * len(VARIANTS) * len(SEEDS)} runs)...")
        exp1_csv = run_exp1(bin_path, data_dir, results_dir, timeout=args.timeout)
        print()
    else:
        exp1_csv = os.path.join(results_dir, "exp1_main_design.csv")

    # ── EXP 2 ──
    if args.exp_only is None or args.exp_only == 2:
        print(f"[3/6] Running EXP 2 — Block Size Effect ({len(BLOCK_SIZES) * len(VARIANTS_CUDA) * len(SEEDS)} runs)...")
        run_exp2(bin_path, data_dir, results_dir, timeout=args.timeout)
        print()

    # ── EXP 3 & 4 (derived from EXP 1) ──
    if args.exp_only is None or args.exp_only in (3, 4):
        if not os.path.isfile(exp1_csv):
            print(f"[ERROR] EXP 1 CSV not found: {exp1_csv}. Run EXP 1 first or use --exp-only 1.")
            sys.exit(1)

        if args.exp_only is None or args.exp_only == 3:
            print("[4/6] Generating EXP 3 — Speed-up Analysis...")
            exp3_csv = generate_exp3(exp1_csv, results_dir)
            print()

        if args.exp_only is None or args.exp_only == 4:
            print("[5/6] Generating EXP 4 — Population Size Effect...")
            generate_exp4(exp1_csv, results_dir)
            print()

    # ── Summary tables ──
    if args.exp_only is None:
        if not os.path.isfile(exp1_csv):
            print(f"[ERROR] EXP 1 CSV not found: {exp1_csv}")
            sys.exit(1)
        exp3_csv = os.path.join(results_dir, "exp3_speedup.csv")
        if not os.path.isfile(exp3_csv):
            print("[WARN] EXP 3 CSV not found, generating first...")
            exp3_csv = generate_exp3(exp1_csv, results_dir)

        print("[6/6] Generating summary tables for report...")
        generate_summary_tables(exp1_csv, exp3_csv, results_dir)
        print()

    print("=" * 58)
    print(" All tasks completed!")
    print(f" Results in: {results_dir}/")
    print("=" * 58)


if __name__ == "__main__":
    main()
