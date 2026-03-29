import numpy as np
from matplotlib import pyplot as plt

import mutual_link_simple as case_a
import mutual_link_simple_B as case_b


def graficar_comparativa(workers, promedios_a, promedios_b, dimensiones_interes, titulos):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    for idx, ax in enumerate(axs):
        dim_idx = dimensiones_interes[idx]

        ax.plot(workers, promedios_a[dim_idx], "o-", color="#1f77b4", label="A_paralelo")
        ax.plot(workers, promedios_b[dim_idx], "s-", color="#ff7f0e", label="B_paralelo")

        ax.set_title(titulos[idx], fontweight="bold")
        ax.set_ylabel("Tiempo promedio (s)")
        ax.set_xlabel("Número de Procesos (Workers)")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(
        "Comparativa de Rendimiento: A_paralelo vs B_paralelo",
        fontsize=16,
        fontweight="bold",
    )
    plt.show()


if __name__ == "__main__":
    dimensions = [5, 50, 500, 1000]
    workers = [2, 4, 8, 16, 32]
    repeticiones = 10

    promedios_a = np.zeros((len(dimensions), len(workers)))
    promedios_b = np.zeros((len(dimensions), len(workers)))

    for i, dim in enumerate(dimensions):
        for j, n_w in enumerate(workers):
            tiempos_a = []
            tiempos_b = []
            print(f"Probando Dimensión {dim}x{dim} con {n_w} workers...")
            for _ in range(repeticiones):
                _, t_a = case_a.simulate_parallel(dim, dim, n_w)
                tiempos_a.append(t_a)
                _, t_b = case_b.simulate_parallel(dim, dim, n_w)
                tiempos_b.append(t_b)
            promedios_a[i, j] = np.mean(tiempos_a)
            promedios_b[i, j] = np.mean(tiempos_b)

    titulos = ["Matriz Base (5,5)", "Matriz Base (500,500)", "Matriz Base (1000,1000)"]
    graficar_comparativa(workers, promedios_a, promedios_b, [0, 2, 3], titulos)
