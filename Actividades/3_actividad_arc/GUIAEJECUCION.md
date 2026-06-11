# Guía de Compilación, Ejecución y Experimentos
## Actividad 3 — Algoritmo Genético con CUDA
## 2. Detectar arquitectura de tu GPU

Antes de compilar, identifica tu GPU:

```bash
nvidia-smi
```

Busca el modelo y usa el valor correcto en CMakeLists.txt:

| GPU                    | CMAKE_CUDA_ARCHITECTURES |
|------------------------|--------------------------|
| RTX 20xx (Turing)      | 75                        |
| RTX 30xx (Ampere)      | 86                        |
| RTX 40xx (Ada)         | 89                        |
| GTX 10xx (Pascal)      | 61                        |
| GTX 16xx (Turing)      | 75                        |
| Detección automática   | native                    |

Editar CMakeLists.txt:
```cmake
set(CMAKE_CUDA_ARCHITECTURES 86)   # ← cambia según tu GPU
```

O para detectar automáticamente (requiere CMake >= 3.24):
```cmake
set(CMAKE_CUDA_ARCHITECTURES native)
```

---

## 3. Compilación

```bash
# Desde la raíz del proyecto
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..
```

Si la compilación falla por arquitectura:
```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native
```

Si no encuentra CUDA:
```bash
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

---

## 4. Ejecución por variante

### Variante 1 — CPU Secuencial (línea base)
```bash
./build/run -i data/small -v sequential -t 1 -s 42
./build/run -i data/medium -v sequential -t 1 -s 42
./build/run -i data/large -v sequential -t 1 -s 42
```

### Variante 2 — CUDA Básico
```bash
# Instancia pequeña, bloque 128
./build/run -i data/small -v cuda_basic -t 1 -s 42 --block-size 128

# Instancia mediana
./build/run -i data/medium -v cuda_basic -t 1 -s 42 --block-size 128

# Instancia grande
./build/run -i data/large -v cuda_basic -t 1 -s 42 --block-size 128
```

### Variante 3 — CUDA Optimizado
```bash
./build/run -i data/small -v cuda_optimized -t 1 -s 42 --block-size 128
./build/run -i data/medium -v cuda_optimized -t 1 -s 42 --block-size 256
./build/run -i data/large -v cuda_optimized -t 1 -s 42 --block-size 256
```

### Con verbose para ver detalle por generación
```bash
./build/run -i data/small -v cuda_basic -t 1 -s 42 --verbose
```

---

## 5. Experimentos requeridos por la actividad

### 5.1 Comparación de variantes (instancia mediana, semilla fija)
```bash
./build/run -i data/medium -v sequential    -t 1 -s 42
./build/run -i data/medium -v cuda_basic    -t 1 -s 42 --block-size 128
./build/run -i data/medium -v cuda_optimized -t 1 -s 42 --block-size 128
```

### 5.2 Efecto del tamaño de bloque (instancia grande, cuda_basic)
```bash
./build/run -i data/large -v cuda_basic -t 1 -s 42 --block-size 32
./build/run -i data/large -v cuda_basic -t 1 -s 42 --block-size 64
./build/run -i data/large -v cuda_basic -t 1 -s 42 --block-size 128
./build/run -i data/large -v cuda_basic -t 1 -s 42 --block-size 256
./build/run -i data/large -v cuda_basic -t 1 -s 42 --block-size 512
```

### 5.3 Efecto del tamaño de instancia (bloque 128, 10 repeticiones)
```bash
for seed in 42 43 44 45 46 47 48 49 50 51; do
  ./build/run -i data/small  -v sequential    -t 1 -s $seed
  ./build/run -i data/small  -v cuda_basic    -t 1 -s $seed --block-size 128
  ./build/run -i data/small  -v cuda_optimized -t 1 -s $seed --block-size 128
  ./build/run -i data/medium -v sequential    -t 1 -s $seed
  ./build/run -i data/medium -v cuda_basic    -t 1 -s $seed --block-size 128
  ./build/run -i data/medium -v cuda_optimized -t 1 -s $seed --block-size 128
  ./build/run -i data/large  -v sequential    -t 1 -s $seed
  ./build/run -i data/large  -v cuda_basic    -t 1 -s $seed --block-size 128
  ./build/run -i data/large  -v cuda_optimized -t 1 -s $seed --block-size 128
done
```

### 5.4 Guardar resultados en CSV automáticamente
```bash
# Script bash para recolectar resultados
echo "instance,variant,block_size,seed,best_fitness,feasible,time_ms" > results/resultados.csv

for seed in 42 43 44 45 46 47 48 49 50 51; do
  for instance in small medium large; do
    for variant in sequential cuda_basic cuda_optimized; do
      for block in 128; do
        output=$(./build/run -i data/$instance -v $variant -t 1 -s $seed --block-size $block 2>&1)
        fitness=$(echo "$output" | grep "Best fitness:" | awk '{print $3}')
        feasible=$(echo "$output" | grep "Feasible:" | awk '{print $2}')
        time=$(echo "$output" | grep "Execution time:" | awk '{print $3}')
        echo "$instance,$variant,$block,$seed,$fitness,$feasible,$time" >> results/resultados.csv
      done
    done
  done
done
```

---

## 6. Medir tiempos con precisión (para el informe)

### Tiempo total del programa:
```bash
time ./build/run -i data/large -v cuda_basic -t 1 -s 42 --block-size 128
```

### Tiempo de kernels con nvprof (versiones CUDA antiguas):
```bash
nvprof ./build/run -i data/medium -v cuda_basic -t 1 -s 42
```

### Tiempo de kernels con Nsight Systems (CUDA moderno, recomendado):
```bash
# Instalar: sudo apt install nsight-systems
nsys profile --stats=true ./build/run -i data/medium -v cuda_basic -t 1 -s 42
```

### Tiempo de kernels con Nsight Compute (detalle por kernel):
```bash
ncu --target-processes all ./build/run -i data/small -v cuda_basic -t 1 -s 42
```

---

## 7. Verificar que la GPU está siendo usada

```bash
# En una terminal, mientras corre el programa:
watch -n 0.5 nvidia-smi

# O en la misma terminal antes de ejecutar:
nvidia-smi dmon -s u &
./build/run -i data/large -v cuda_basic -t 1 -s 42
```

---

## 8. Errores comunes y soluciones

| Error | Causa | Solución |
|-------|-------|----------|
| `nvcc not found` | CUDA no está en PATH | `export PATH=/usr/local/cuda/bin:$PATH` |
| `no kernel image for this device` | Arquitectura incorrecta | Cambiar `CMAKE_CUDA_ARCHITECTURES` |
| `out of memory` | Población muy grande en GPU | Reducir population_size en constants.hpp |
| `illegal memory access` | Bug de indexación | Reducir a instancia pequeña y revisar |
| `curand not found` | Falta librería | `sudo apt install libcurand-dev` |
| Resultado siempre inválido | Penalizaciones muy bajas | Revisar constants.hpp Penalty |

---

## 9. Hardware a reportar en el informe

```bash
# CPU
lscpu | grep "Model name"

# GPU
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Versión CUDA
nvcc --version

# RAM
free -h
```