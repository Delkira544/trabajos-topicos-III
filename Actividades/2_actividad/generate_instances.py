"""
Generador de instancias para el problema de la mochila extendida
INFO1194 - Actividad 2

Genera los archivos CSV para instancias small, medium y large:
  - items.csv
  - category_rules.csv
  - incompatibilities.csv
  - dependencies.csv

Uso:
    python generate_instances.py [--seed SEED]

Opciones:
    --seed      Semilla aleatoria para reproducibilidad (default: 42)
"""

import argparse
import csv
import os
import random

# ─────────────────────────────────────────────
# Configuración de instancias
# ─────────────────────────────────────────────
INSTANCES = {
    "small": {"n_items": 100, "n_categories": 5},
    "medium": {"n_items": 1_000, "n_categories": 10},
    "large": {"n_items": 10_000, "n_categories": 20},
}

# Porcentaje de capacidad recomendado (instancia media)
CAPACITY_RATIO = 0.40

# Proporciones de incompatibilidades y dependencias sobre n_items
# Se ajustan dinámicamente según el tamaño de la instancia
BASE_INCOMPATIBILITY_RATIO = 0.03  # ~3% de pares incompatibles para small
BASE_DEPENDENCY_RATIO = 0.05  # ~5% de ítems con dependencia para small


def calculate_penalties(n_items: int) -> dict:
    """
    Calcula penalizaciones proporcionales al tamaño de la instancia.
    Valores base son para small (100 ítems), escalan linealmente.
    """
    # Valores base para instancia small (100 ítems)
    base_alpha = 10.0    # por kg exceso peso
    base_beta = 10.0     # por litro exceso volumen
    base_gamma = 50.0    # por violación de categoría
    base_delta = 200.0    # por par incompatible
    base_epsilon = 200.0  # por dependencia faltante

    # Factor de escala: small=1x, medium=10x, large=100x
    scale = n_items / 100

    return {
        "alpha": base_alpha * scale,
        "beta": base_beta * scale,
        "gamma": base_gamma * scale,
        "delta": base_delta * scale,
        "epsilon": base_epsilon * scale,
    }


def generate_items(n_items: int, n_categories: int, rng: random.Random) -> list[dict]:
    """Genera la lista de ítems con valor, peso, volumen y categoría proporcionales."""
    categories = [f"cat_{c}" for c in range(n_categories)]
    
    # Escalar rangos basado en el tamaño de la instancia: √(n/100)
    scale = max(1.0, (n_items / 100) ** 0.5)
    max_value = int(1000 * scale)
    max_weight = int(100 * scale)
    max_volume = int(100 * scale)
    
    items = []
    for i in range(n_items):
        items.append(
            {
                "id": i,
                "valor": rng.randint(1, max_value),
                "peso": rng.randint(1, max_weight),
                "volumen": rng.randint(1, max_volume),
                "categoria": rng.choice(categories),
            }
        )
    return items


def generate_category_rules(
    items: list[dict],
    n_categories: int,
    rng: random.Random,
) -> list[dict]:
    """Genera reglas de mínimo/máximo por categoría."""
    # Contar ítems por categoría
    counts: dict[str, int] = {}
    for it in items:
        counts[it["categoria"]] = counts.get(it["categoria"], 0) + 1

    rules = []
    for cat, total in counts.items():
        # Mínimo: entre 0 y 10% del total de esa categoría
        minimo = rng.randint(0, max(0, total // 10))
        # Máximo: entre 30% y 80% del total de esa categoría (siempre >= minimo)
        maximo = rng.randint(
            max(minimo, total // 4), max(minimo + 1, int(total * 0.80))
        )
        rules.append(
            {
                "categoria": cat,
                "minimo": minimo,
                "maximo": maximo,
            }
        )
    return rules


def generate_incompatibilities(
    n_items: int,
    ratio: float,
    rng: random.Random,
) -> list[dict]:
    """Genera pares de ítems incompatibles (sin repetir ni reflejar)."""
    n_pairs = int(n_items * ratio)
    pairs: set[tuple[int, int]] = set()

    attempts = 0
    max_attempts = n_pairs * 10
    while len(pairs) < n_pairs and attempts < max_attempts:
        a = rng.randint(0, n_items - 1)
        b = rng.randint(0, n_items - 1)
        if a != b:
            pair = (min(a, b), max(a, b))
            pairs.add(pair)
        attempts += 1

    return [{"id_item_a": a, "id_item_b": b} for a, b in sorted(pairs)]


def generate_dependencies(
    n_items: int,
    ratio: float,
    rng: random.Random,
) -> list[dict]:
    """Genera dependencias: si se selecciona id_item, se requiere id_requerido."""
    n_deps = int(n_items * ratio)
    deps: set[tuple[int, int]] = set()

    attempts = 0
    max_attempts = n_deps * 10
    while len(deps) < n_deps and attempts < max_attempts:
        item = rng.randint(0, n_items - 1)
        requerido = rng.randint(0, n_items - 1)
        if item != requerido:
            deps.add((item, requerido))
        attempts += 1

    return [{"id_item": item, "id_requerido": req} for item, req in sorted(deps)]


def write_csv(filepath: str, fieldnames: list[str], rows: list[dict]) -> None:
    """Escribe un archivo CSV con encabezado."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(name: str, items: list[dict], capacity_ratio: float) -> None:
    """Imprime un resumen de la instancia generada."""
    total_peso = sum(it["peso"] for it in items)
    total_volumen = sum(it["volumen"] for it in items)
    W = int(total_peso * capacity_ratio)
    V = int(total_volumen * capacity_ratio)
    print(f"\n  [{name}] {len(items)} ítems")
    print(
        f"    Suma total peso:    {total_peso:>10}  →  W = {W} ({capacity_ratio * 100:.0f}%)"
    )
    print(
        f"    Suma total volumen: {total_volumen:>10}  →  V = {V} ({capacity_ratio * 100:.0f}%)"
    )


def generate_instance(
    name: str, cfg: dict, rng: random.Random, base_dir: str = "data"
) -> None:
    """Genera todos los CSV para una instancia."""
    out_dir = os.path.join(base_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    n_items = cfg["n_items"]
    n_categories = cfg["n_categories"]

    # ── Calcular ratios proporcionales ────────────────────────
    # Ajustar ratios dinámicamente: cap máximo para evitar exceso
    incomp_ratio = min(BASE_INCOMPATIBILITY_RATIO, 50.0 / n_items)
    dep_ratio = min(BASE_DEPENDENCY_RATIO, 100.0 / n_items)

    # ── Generar datos ──────────────────────────────────────────
    items = generate_items(n_items, n_categories, rng)
    category_rules = generate_category_rules(items, n_categories, rng)
    incompatibilities = generate_incompatibilities(n_items, incomp_ratio, rng)
    dependencies = generate_dependencies(n_items, dep_ratio, rng)
    penalties = calculate_penalties(n_items)

    # ── Escribir CSV ───────────────────────────────────────────
    write_csv(
        os.path.join(out_dir, "items.csv"),
        ["id", "valor", "peso", "volumen", "categoria"],
        items,
    )
    write_csv(
        os.path.join(out_dir, "category_rules.csv"),
        ["categoria", "minimo", "maximo"],
        category_rules,
    )
    write_csv(
        os.path.join(out_dir, "incompatibilities.csv"),
        ["id_item_a", "id_item_b"],
        incompatibilities,
    )
    write_csv(
        os.path.join(out_dir, "dependencies.csv"),
        ["id_item", "id_requerido"],
        dependencies,
    )
    write_csv(
        os.path.join(out_dir, "penalty_config.csv"),
        ["penalty_type", "value"],
        [
            {"penalty_type": "peso_exceso", "value": penalties["alpha"]},
            {"penalty_type": "volumen_exceso", "value": penalties["beta"]},
            {"penalty_type": "categoria", "value": penalties["gamma"]},
            {"penalty_type": "incompatibilidad", "value": penalties["delta"]},
            {"penalty_type": "dependencia", "value": penalties["epsilon"]},
        ],
    )

    print_summary(name, items, CAPACITY_RATIO)
    print(f"    Reglas de categoría:   {len(category_rules)}")
    print(f"    Incompatibilidades:    {len(incompatibilities)}")
    print(f"    Dependencias:          {len(dependencies)}")
    print(f"    Penalización peso:     {penalties['alpha']:.1f}")
    print(f"    Penalización volumen:  {penalties['beta']:.1f}")
    print(f"    Penalización categ:    {penalties['gamma']:.1f}")
    print(f"    Penalización incomp:   {penalties['delta']:.1f}")
    print(f"    Penalización dep:      {penalties['epsilon']:.1f}")
    print(f"    Archivos en:           {out_dir}/")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generador de instancias CSV para el problema de la mochila extendida (INFO1194)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Directorio base de salida (default: data/)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("=" * 55)
    print(" Generador de instancias — Mochila Extendida (INFO1194)")
    print("=" * 55)
    print(f"  Semilla:          {args.seed}")
    print(f"  Directorio base:  {args.output}/")
    print(f"  Ratio capacidad:  {CAPACITY_RATIO * 100:.0f}% del total")

    for instance_name, config in INSTANCES.items():
        generate_instance(instance_name, config, rng, base_dir=args.output)

    print("\n✓ Instancias generadas correctamente.")
    print(
        f"  Estructura: {args.output}/{{small,medium,large}}/{{items,category_rules,incompatibilities,dependencies}}.csv"
    )


if __name__ == "__main__":
    main()
