import re
import time

import numpy as np


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
    print(f"Resultado: {result}, Tiempo transcurrido: {elapsed_time:.12f} segundos")
    return result, elapsed_time
