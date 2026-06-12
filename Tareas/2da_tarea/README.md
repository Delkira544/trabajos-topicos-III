# Tarea 2 — Paralelización de algoritmos secuenciales en CUDA C++

## GPU Utilizada / Versión CUDA

- **GPU:** NVIDIA GeForce RTX 4060 (Compute Capability 8.9)
- **Versión CUDA:** CUDA 13.2

---

## Compilación con CMake

```bash
cd Tareas/2da_tarea
cmake -B build && cmake --build build -j$(nproc)
```

## Ejecución

### Todos los programas en orden

```bash
cd build && ctest
```
Ejecuta: secuencial1 → secuencial2 → ... → secuencial5 → paralelo1 → ... → paralelo5

### Programas individuales

```bash
./build/secuencial1 [N]         # Transformación secuencial
./build/secuencial2 [N]         # Stencil secuencial
./build/secuencial3 [N]         # Reducción secuencial
./build/secuencial4 [N]         # Histograma secuencial
./build/secuencial5 [N]         # Conteo condicional secuencial

./build/paralelo1 [N]           # Transformación paralela
./build/paralelo2 [N]           # Stencil paralelo
./build/paralelo3 [N]           # Reducción paralela
./build/paralelo4 [N]           # Histograma paralelo
./build/paralelo5 [N] [umbral]  # Conteo condicional paralelo
```

N opcional — default: `1<<24` (P1-P3), `1<<26` (P4-P5). Umbral default: 500.

## Targets disponibles

| Target | Archivo fuente | Lenguaje |
|--------|---------------|----------|
| `secuencial1` | `problema1_transformacion/secuencial.cpp` | C++ (g++) |
| `secuencial2` | `problema2_stencil/secuencial.cpp` | C++ (g++) |
| `secuencial3` | `problema3_reduccion/secuencial.cpp` | C++ (g++) |
| `secuencial4` | `problema4_histograma/secuencial.cpp` | C++ (g++) |
| `secuencial5` | `problema5_warps/secuencial.cpp` | C++ (g++) |
| `paralelo1` | `problema1_transformacion/paralelo.cu` | CUDA (nvcc) |
| `paralelo2` | `problema2_stencil/paralelo.cu` | CUDA (nvcc) |
| `paralelo3` | `problema3_reduccion/paralelo.cu` | CUDA (nvcc) |
| `paralelo4` | `problema4_histograma/paralelo.cu` | CUDA (nvcc) |
| `paralelo5` | `problema5_warps/paralelo.cu` | CUDA (nvcc) |
