# Definir la versión mínima de CMake y el nombre del proyecto
cmake_minimum_required(VERSION 3.10)
project(MiClienteCLI)

# Asegura que el compilador use C++11 o superior para CLI11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compile
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Indicar las carpetas donde están los archivos .hpp (cabeceras)
# Esto reemplaza el flag -I de la terminal.
include_directories(include)
include_directories(external)

find_package(OpenMP REQUIRED)

# Crear el ejecutable final llamado "run"
# Aquí listamos todos los archivos .cpp que contienen la lógica.
add_executable(run
  src/main.cpp
  src/cli/app_parser.cpp
  src/data/instance_loader.cpp
  src/config/constants.cpp
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(run PRIVATE OpenMP::OpenMP_CXX)
endif()

# Crear ejecutable para tuning de penalidades
add_executable(penalty_tuner
  src/penalty_tuner_main.cpp
  src/penalty_tuner.cpp
  src/cli/app_parser.cpp
  src/data/instance_loader.cpp
  src/config/constants.cpp
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(penalty_tuner PRIVATE OpenMP::OpenMP_CXX)
endif()
