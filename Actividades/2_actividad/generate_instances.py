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
    "small": {"n_items": 100},
    "medium": {"n_items": 1_000},
    "large": {"n_items": 10_000},
}

N_CATEGORIES = 5  # Fijo para todas las instancias

# Porcentaje de capacidad recomendado (instancia media)
CAPACITY_RATIO = 0.40

# Proporción fija de incompatibilidades y dependencias sobre n_items.
# Al ser constantes producen cantidades linealmente proporcionales al tamaño:
#   small (100):  5 incompatibilidades,  10 dependencias
#   medium (1000): 50 incompatibilidades, 100 dependencias
#   large (10000): 500 incompatibilidades, 1000 dependencias
INCOMPATIBILITY_RATIO = 0.03
DEPENDENCY_RATIO = 0.02


def calculate_penalties(n_items: int) -> dict:
    """
    Pesos de cada restricción dentro de violacion_norm (combinación convexa,
    Σ = 1.0). Mejora #2: las restricciones HARD (peso/volumen) llevan el
    mayor peso porque son obligatorias; las soft se reparten el resto.
    """
    return {
        "alpha": 0.30,  # exceso de peso (HARD)
        "beta": 0.30,  # exceso de volumen (HARD)
        "gamma": 0.05,  # categorías
        "delta": 0.20,  # incompatibilidades
        "epsilon": 0.15,  # dependencias
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


def generate_category_rules(items: list[dict]) -> list[dict]:
    """Genera reglas de mínimo/máximo proporcionales al conteo real de cada categoría.

    minimo = 5% del total de esa categoría  → el GA debe incluir al menos algunos ítems
    maximo = 40% del total de esa categoría → coherente con CAPACITY_RATIO
    Ambos valores escalan linealmente con la distribución real, independientemente del tamaño.
    """
    counts: dict[str, int] = {}
    for it in items:
        counts[it["categoria"]] = counts.get(it["categoria"], 0) + 1

    rules = []
    for cat, total in counts.items():
        minimo = max(0, int(total * 0.05))
        maximo = max(minimo + 1, int(total * 0.40))
        rules.append({"categoria": cat, "minimo": minimo, "maximo": maximo})
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
        a, b = rng.randint(0, n_items - 1), rng.randint(0, n_items - 1)
        if a != b:
            # id_requerido < id_item garantiza que no se forman ciclos directos
            deps.add((max(a, b), min(a, b)))
        attempts += 1

    return [{"id_item": item, "id_requerido": req} for item, req in sorted(deps)]


def write_csv(filepath: str, fieldnames: list[str], rows: list[dict]) -> None:
    """Escribe un archivo CSV con encabezado."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    name: str, items: list[dict], W: int, V: int, capacity_ratio: float
) -> None:
    """Imprime un resumen de la instancia generada."""
    total_peso = sum(it["peso"] for it in items)
    total_volumen = sum(it["volumen"] for it in items)
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

    incomp_ratio = INCOMPATIBILITY_RATIO
    dep_ratio = DEPENDENCY_RATIO

    # ── Generar datos ──────────────────────────────────────────
    items = generate_items(n_items, N_CATEGORIES, rng)
    category_rules = generate_category_rules(items)
    incompatibilities = generate_incompatibilities(n_items, incomp_ratio, rng)
    dependencies = generate_dependencies(n_items, dep_ratio, rng)
    penalties = calculate_penalties(n_items)

    # ── Capacidad de la mochila (proporcional al total de items) ─
    total_peso = sum(it["peso"] for it in items)
    total_volumen = sum(it["volumen"] for it in items)
    W = int(total_peso * CAPACITY_RATIO)
    V = int(total_volumen * CAPACITY_RATIO)

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
    write_csv(
        os.path.join(out_dir, "knapsack_config.csv"),
        ["max_weight", "max_volume"],
        [{"max_weight": W, "max_volume": V}],
    )

    print_summary(name, items, W, V, CAPACITY_RATIO)
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
