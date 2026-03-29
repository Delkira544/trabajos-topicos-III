import argparse

import numpy as np
from matplotlib import pyplot as plt

import mutual_link_simple as case_a
import mutual_link_simple_B as case_b


def Graficar_Comparativa(workers, resultados_a, resultados_b, dimensiones_interes):
    """
    dimensiones_interes: lista de los índices de las dimensiones que queremos graficar
    (ej: [0, 2, 3] para 5x5, 500x500 y 1000x1000)
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    titulos = ["Matriz Base (5,5)", "Matriz Base (500,500)", "Matriz Base (1000,1000)"]

    for idx, ax in enumerate(axs):
        dim_idx = dimensiones_interes[idx]

        ax.plot(
            workers, resultados_a[dim_idx], "o-", color="#1f77b4", label="A_paralelo"
        )
        ax.plot(
            workers, resultados_b[dim_idx], "s-", color="#ff7f0e", label="B_paralelo"
        )

        ax.set_title(titulos[idx], fontweight="bold")
        ax.set_ylabel("Tiempo promedio (s)")
        ax.set_xlabel("Número de Procesos (Workers)")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(
        "Comparativa de Rendimiento: Escalabilidad de Procesos",
        fontsize=16,
        fontweight="bold",
    )
    plt.show()


def Graficar(dimensions, mean_a, mean_b):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    color_a = "#1f77b4"
    color_b = "#ff7f0e"

    axs[0].plot(dimensions, mean_a, color=color_a, marker="o", label="Caso A")
    axs[0].set_ylabel("Tiempo (s)", fontsize=10)
    axs[0].set_title("Análisis Individual: Caso A", fontsize=12, fontweight="bold")
    axs[0].grid(True, linestyle=":", alpha=0.6)
    axs[0].legend(loc="upper left")

    axs[1].plot(dimensions, mean_b, color=color_b, marker="s", label="Caso B")
    axs[1].set_ylabel("Tiempo (s)", fontsize=10)
    axs[1].set_title("Análisis Individual: Caso B", fontsize=12, fontweight="bold")
    axs[1].grid(True, linestyle=":", alpha=0.6)
    axs[1].legend(loc="upper left")

    axs[2].plot(
        dimensions, mean_a, color=color_a, marker="o", linestyle="-", label="Caso A"
    )
    axs[2].plot(
        dimensions, mean_b, color=color_b, marker="s", linestyle="-", label="Caso B"
    )

    axs[2].set_xlabel("Dimensión de Entrada", fontsize=10)
    axs[2].set_ylabel("Tiempo promedio (s)", fontsize=10)
    axs[2].set_title("Comparación de Tiempos: A vs B", fontsize=12, fontweight="bold")
    axs[2].grid(True, linestyle="--", alpha=0.8)
    axs[2].legend(loc="upper left", fontsize=11)

    fig.suptitle(
        "Resultados Experimentales de Rendimiento",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    plt.show()


def main():
    dimensions = [5, 50, 500, 1000]
    times_a = np.zeros(shape=(len(dimensions), 10))
    times_b = np.zeros(shape=(len(dimensions), 10))
    print(times_a)

    for i, dimension in enumerate(dimensions):
        print(f"Probando Dimensión {dimension}...")
        for j in range(10):
            result_a, elapsed_time_a = case_a.simulate(dimension, dimension)
            result_b, elapsed_time_b = case_b.simulate(dimension, dimension)
            times_a[i, j] = elapsed_time_a
            times_b[i, j] = elapsed_time_b

    mean_a = np.mean(times_a, axis=1)
    mean_b = np.mean(times_b, axis=1)

    Graficar(dimensions, mean_a, mean_b)


def main_parallel():
    dimensions = [5, 50, 500, 1000]
    workers = [2, 4, 8, 16, 32]
    repeticiones = 10

    promedios_a = np.zeros((len(dimensions), len(workers)))
    promedios_b = np.zeros((len(dimensions), len(workers)))

    for i, dim in enumerate(dimensions):
        for j, n_w in enumerate(workers):
            tiempos_iter_a = []
            tiempos_iter_b = []

            print(f"Probando Dimensión {dim} con {n_w} workers...")

            for _ in range(repeticiones):
                _, t_a = case_a.simulate_parallel(dim, dim, n_w)
                tiempos_iter_a.append(t_a)

                _, t_b = case_b.simulate_parallel(dim, dim, n_w)
                tiempos_iter_b.append(t_b)

            promedios_a[i, j] = np.mean(tiempos_iter_a)
            promedios_b[i, j] = np.mean(tiempos_iter_b)

    Graficar_Comparativa(workers, promedios_a, promedios_b, [0, 2, 3])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulación de mutual outlinks")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Ejecutar la simulación en modo paralelo (solo caso A)",
    )
    args = parser.parse_args()
    if args.parallel:
        main_parallel()
    else:
        main()
