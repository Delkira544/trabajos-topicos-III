# Definir la versión mínima de CMake y el nombre del proyecto
cmake_minimum_required(VERSION 3.18)
project(MiClienteCLI LANGUAGES CXX CUDA)

# ── Estándares de lenguaje ───────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── Compilación ──────────────────────────────────────────────────────────────
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Arquitectura CUDA ────────────────────────────────────────────────────────
# Ajusta según tu GPU:
#   Turing  (RTX 20xx):  75
#   Ampere  (RTX 30xx):  86
#   Ada     (RTX 40xx):  89
#   Pascal  (GTX 10xx):  61
# Para detectar automáticamente usar: set(CMAKE_CUDA_ARCHITECTURES native)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# ── Flags de optimización CUDA ───────────────────────────────────────────────
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# ── Includes ─────────────────────────────────────────────────────────────────
include_directories(include)
include_directories(external)
include_directories(src)

# ── Dependencias ─────────────────────────────────────────────────────────────
find_package(OpenMP COMPONENTS CXX REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ── Fuentes CUDA (kernels + solvers GPU) ─────────────────────────────────────
set(CUDA_SOURCES
  src/ga/cuda/fitness_kernel.cu
  src/ga/cuda/operators_kernel.cu
  src/ga/cuda/reduction_kernel.cu
  src/ga/solvers/CUDABasic.cu
  src/ga/solvers/CUDAOptimized.cu
)

set(COMMON_SOURCES
  src/cli/app_parser.cpp
  src/data/instance_loader.cpp
  src/config/constants.cpp
)

# --- Target: run ---
add_executable(run src/main.cpp ${COMMON_SOURCES} ${CUDA_SOURCES})
target_link_libraries(run PRIVATE OpenMP::OpenMP_CXX CUDA::cudart CUDA::curand)
set_target_properties(run PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# --- Target: penalty_tuner ---
# Added ${CUDA_SOURCES} to fix undefined reference errors
add_executable(penalty_tuner
  src/penalty_tuner_main.cpp
  src/penalty_tuner.cpp
  ${COMMON_SOURCES}
  ${CUDA_SOURCES}
)

target_link_libraries(penalty_tuner PRIVATE 
    OpenMP::OpenMP_CXX 
    CUDA::cudart 
    CUDA::curand
)
set_target_properties(penalty_tuner PROPERTIES CUDA_SEPARABLE_COMPILATION ON)