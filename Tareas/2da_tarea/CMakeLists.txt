cmake_minimum_required(VERSION 3.18)

project(Tarea2_CUDA LANGUAGES CUDA CXX)

# Manera correcta y nativa de pedir C++17 en CMake tanto para C++ como para CUDA
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Evita que nvcc herede banderas conflictivas en modo Debug
set(CMAKE_CUDA_ARCHITECTURES native)

# Agregamos la macro específica que necesitas usando funciones nativas por target
macro(configurar_target_cuda target_name)
    target_compile_definitions(${target_name} PRIVATE "__builtin_is_virtual_base_of(...)=0")
endmacro()

# Ejecutables Secuenciales (C++)
add_executable(secuencial1 problema1_transformacion/secuencial.cpp)
add_executable(secuencial2 problema2_stencil/secuencial.cpp)
add_executable(secuencial3 problema3_reduccion/secuencial.cpp)
add_executable(secuencial4 problema4_histograma/secuencial.cpp)
add_executable(secuencial5 problema5_warps/secuencial.cpp)

# Ejecutables Paralelos (CUDA)
add_executable(paralelo1 problema1_transformacion/paralelo.cu)
add_executable(paralelo2 problema2_stencil/paralelo.cu)
add_executable(paralelo3 problema3_reduccion/paralelo.cu)
add_executable(paralelo4 problema4_histograma/paralelo.cu)
add_executable(paralelo5 problema5_warps/paralelo.cu)

# Aplicar la macro solo a los ejecutables de CUDA para evitar conflictos
configurar_target_cuda(paralelo1)
configurar_target_cuda(paralelo2)
configurar_target_cuda(paralelo3)
configurar_target_cuda(paralelo4)
configurar_target_cuda(paralelo5)

# Pruebas automatizadas
enable_testing()
add_test(NAME test_secuencial1 COMMAND secuencial1)
add_test(NAME test_secuencial2 COMMAND secuencial2)
add_test(NAME test_secuencial3 COMMAND secuencial3)
add_test(NAME test_secuencial4 COMMAND secuencial4)
add_test(NAME test_secuencial5 COMMAND secuencial5)
add_test(NAME test_paralelo1 COMMAND paralelo1)
add_test(NAME test_paralelo2 COMMAND paralelo2)
add_test(NAME test_paralelo3 COMMAND paralelo3)
add_test(NAME test_paralelo4 COMMAND paralelo4)
add_test(NAME test_paralelo5 COMMAND paralelo5)
