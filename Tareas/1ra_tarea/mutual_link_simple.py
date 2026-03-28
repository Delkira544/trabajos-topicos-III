import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def sum_partial(i, links):
    nr, nc = links.shape
    parcial = 0
    for j in range(i + 1, nr):
        for k in range(nc):
            parcial += links[i, k] * links[j, k]
    return parcial


def mutual_outlinks_parallel(links, n_workers):
    nr, nc = links.shape
    tot = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        index_i = range(nr - 1)
        results = executor.map(sum_partial, index_i, [links] * (nr - 1))
    tot = sum(results)
    return tot / (nr * (nr - 1) / 2)


def mutual_outlinks(links):
    nr, nc = links.shape
    tot = 0
    for i in range(nr - 1):
        for j in range(i + 1, nr):
            for k in range(nc):
                tot += links[i, k] * links[j, k]
    return tot / (nr * (nr - 1) / 2)


def simulate(nr, nc):
    links = np.random.choice([0, 1], size=(nr * nc), replace=True).reshape(nr, nc)
    start_time = time.time()
    result = mutual_outlinks(links)
    elapsed_time = time.time() - start_time
    print(f"Resultado A: {result}, Tiempo transcurrido: {elapsed_time:.12f} segundos")
    return result, elapsed_time


def simulate_parallel(nr, nc, n_workers):
    links = np.random.choice([0, 1], size=(nr * nc), replace=True).reshape(nr, nc)
    start_time = time.time()
    result = mutual_outlinks_parallel(links, n_workers)
    elapsed_time = time.time() - start_time
    print(
        f"Resultado A_Paralelo: {result}, Tiempo transcurrido: {elapsed_time:.12f} segundos"
    )
    return result, elapsed_time
