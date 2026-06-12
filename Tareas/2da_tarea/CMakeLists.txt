cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_FLAGS "-O2 --std=c++17 -D__builtin_is_virtual_base_of\\(...\\)=0" CACHE STRING "CUDA flags" FORCE)

project(Tarea2_CUDA LANGUAGES CUDA CXX)

add_executable(secuencial1 problema1_transformacion/secuencial.cpp)
add_executable(secuencial2 problema2_stencil/secuencial.cpp)
add_executable(secuencial3 problema3_reduccion/secuencial.cpp)
add_executable(secuencial4 problema4_histograma/secuencial.cpp)
add_executable(secuencial5 problema5_warps/secuencial.cpp)

add_executable(paralelo1 problema1_transformacion/paralelo.cu)
add_executable(paralelo2 problema2_stencil/paralelo.cu)
add_executable(paralelo3 problema3_reduccion/paralelo.cu)
add_executable(paralelo4 problema4_histograma/paralelo.cu)
add_executable(paralelo5 problema5_warps/paralelo.cu)

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
