import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def mutual_outlinks(links: np.ndarray):
    nr, nc = links.shape
    tot = 0
    for i in range(nr - 1):
        tmp = np.dot(links[i + 1 : nr, :], links[i, :].T)
        tot += np.sum(tmp)
    return tot / (nr * (nr - 1) / 2)


def simulate(nr, nc):
    links = np.random.choice([0, 1], size=(nr * nc), replace=True).reshape(nr, nc)
    start_time = time.time()
    result = mutual_outlinks(links)
    elapsed_time = time.time() - start_time
    print(f"Resultado B: {result}, Tiempo transcurrido: {elapsed_time:.12f} segundos")
    return result, elapsed_time

