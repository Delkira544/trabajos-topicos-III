# 1. DETECCIÓN DEL SISTEMA OPERATIVO
ifeq ($(OS), Windows_NT)
    # Configuración para Windows
    RM = del /Q
    MKDIR = if not exist $(BUILD_DIR) mkdir
    EXT = .exe
    # Función para arreglar las barras en Windows ( / -> \ )
    FIX_PATH = $(subst /,\,$1)
else
    # Configuración para Linux / macOS
    RM = rm -rf
    MKDIR = mkdir -p
    EXT =
    FIX_PATH = $1
endif

# 2. Variables de configuración
CXX      = g++
CXXFLAGS = -Wall -Wextra -std=c++17 -fopenmp
TARGET   = mochila_ga$(EXT)

# Carpetas
SRC_DIR   = src
BUILD_DIR = build

# Buscamos los fuentes y definimos dónde irán los objetos
SRCS      = $(wildcard $(SRC_DIR)/*.cpp)
# Esto transforma 'src/main.cpp' en 'build/main.o'
OBJS      = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))

# 3. REGLAS PRINCIPALES
GENERATIONS = 700
POPULATION = 120
POPULATION_LARGE = 250   # mejora #9 — large requiere más exploración (10000-D)
THEADS = 8
SEED = 132512

# El objetivo final depende de los objetos en la carpeta build
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Regla para compilar los .cpp y guardarlos en build/
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@$(MKDIR) $(call FIX_PATH, $(BUILD_DIR))
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Regla para ejecutar
run: all
	./$(TARGET)

small: all
	./$(TARGET) --instance data/small \
		    	--population $(POPULATION) \
				--generations $(GENERATIONS) \
				--threads $(THEADS)  \
				--seed $(SEED) \
				--report-file result/report_small.csv

medium: all
	./$(TARGET) --instance data/medium \
       	        --population $(POPULATION) \
                --generations $(GENERATIONS) \
                --threads $(THEADS)  \
                --seed $(SEED) \
               	--report-file result/report_medium.csv


large: all
	./$(TARGET) --instance data/large \
	   	        --population $(POPULATION_LARGE) \
				--generations $(GENERATIONS) \
				--threads $(THEADS)  \
				--seed $(SEED) \
				--report-file result/report_large.csv

# benchmark: ejecuta la suite completa para una config dada y guarda
# los CSV con un prefijo personalizado.
#
# Uso:   make benchmark <config> <nombre>
# Ej:    make benchmark 1 experimento1
# Salida:
#   results/<nombre>_bench_standard_<config>.csv
#   results/<nombre>_bench_islands_<config>.csv
benchmark: all
	$(eval BENCH_ARGS := $(filter-out $@,$(MAKECMDGOALS)))
	$(eval BENCH_CONFIG := $(word 1,$(BENCH_ARGS)))
	$(eval BENCH_NAME := $(word 2,$(BENCH_ARGS)))
	$(if $(BENCH_CONFIG),,$(error Uso: make benchmark <config> <nombre>. ej: make benchmark 1 experimento1))
	$(if $(BENCH_NAME),,$(error Uso: make benchmark <config> <nombre>. ej: make benchmark 1 experimento1))
ifeq ($(OS), Windows_NT)
	@if not exist results mkdir results
else
	@mkdir -p results
endif
	@echo ""
	@echo "===================================================="
	@echo " Benchmark — config=$(BENCH_CONFIG)  nombre=$(BENCH_NAME)"
	@echo "===================================================="
	@echo ""
	@echo ">>> Variante: standard"
	./$(TARGET) --benchmark --variant standard --config $(BENCH_CONFIG) \
		--seed $(SEED) --threads $(THEADS) \
		--report-file results/$(BENCH_NAME)_bench_standard_$(BENCH_CONFIG).csv
	@echo ""
	@echo ">>> Variante: islands"
	./$(TARGET) --benchmark --variant islands --config $(BENCH_CONFIG) \
		--seed $(SEED) --threads $(THEADS) \
		--report-file results/$(BENCH_NAME)_bench_islands_$(BENCH_CONFIG).csv
	@echo ""
	@echo "✓ Benchmark terminado. Archivos:"
	@echo "  results/$(BENCH_NAME)_bench_standard_$(BENCH_CONFIG).csv"
	@echo "  results/$(BENCH_NAME)_bench_islands_$(BENCH_CONFIG).csv"

benchmark-island: all
	$(eval BENCH_ARGS := $(filter-out $@,$(MAKECMDGOALS)))
	$(eval BENCH_CONFIG := $(word 1,$(BENCH_ARGS)))
	$(eval BENCH_NAME := $(word 2,$(BENCH_ARGS)))
	$(if $(BENCH_CONFIG),,$(error Uso: make benchmark <config> <nombre>. ej: make benchmark 1 experimento1))
	$(if $(BENCH_NAME),,$(error Uso: make benchmark <config> <nombre>. ej: make benchmark 1 experimento1))
ifeq ($(OS), Windows_NT)
	@if not exist results mkdir results
else
	@mkdir -p results
endif
	@echo ">>> Variante: islands"
	./$(TARGET) --benchmark --variant islands --config $(BENCH_CONFIG) \
		--seed $(SEED) --threads $(THEADS) \
		--report-file results/$(BENCH_NAME)_bench_islands_$(BENCH_CONFIG).csv
	@echo "  results/$(BENCH_NAME)_bench_islands_$(BENCH_CONFIG).csv"


# Limpieza profunda
clean:
	$(RM) $(call FIX_PATH, $(BUILD_DIR)/*.o)
	$(RM) $(TARGET)
	@echo Limpieza completada.

.PHONY: all run clean benchmark

# Catch-all: silencia "No rule to make target ..." cuando se pasan args
# posicionales (e.g. el "1" y "experimento1" en `make benchmark 1 experimento1`).
# Sólo se aplica a targets no definidos explícitamente.
%:
	@:
