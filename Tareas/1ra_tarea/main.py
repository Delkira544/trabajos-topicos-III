import numpy as np
from matplotlib import pyplot as plt

import mutual_link_simple as case_a
import mutual_link_simple_B as case_b


def Graficar(dimensions, times_a, times_b):
    mean_a = np.mean(times_a, axis=1)
    mean_b = np.mean(times_b, axis=1)

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
    dimensions = [5, 50, 100, 1000]
    times_a = np.zeros(shape=(len(dimensions), 10))
    times_b = np.zeros(shape=(len(dimensions), 10))
    print(times_a)

    for i, dimension in enumerate(dimensions):
        for j in range(10):
            result_a, elapsed_time_a = case_a.simulate(dimension, dimension)
            result_b, elapsed_time_b = case_b.simulate(dimension, dimension)
            times_a[i, j] = elapsed_time_a
            times_b[i, j] = elapsed_time_b

    Graficar(dimensions, times_a, times_b)


if __name__ == "__main__":
    main()
