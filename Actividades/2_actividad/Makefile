# Sistema Operativo
ifeq ($(OS), Windows_NT)
    RM = del /Q
    MKDIR = if not exist $(BUILD_DIR) mkdir
    EXT = .exe
    FIX_PATH = $(subst /,\,$1)
else
    RM = rm -rf
    MKDIR = mkdir -p
    EXT =
    FIX_PATH = $1
endif

# Variables
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 -fopenmp
TARGET = mochila_ga$(EXT)
SRC_DIR = src
BUILD_DIR = build
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))

# Configuración por defecto
GEN = 700
POP = 120
POP_LARGE = 250
THREADS = 8
SEED = 132512

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@$(MKDIR) $(call FIX_PATH, $(BUILD_DIR))
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Ejecuciones estándar
small: all
	./$(TARGET) --instance data/small --population $(POP) --generations $(GEN) --threads $(THREADS) --seed $(SEED) --report-file results/report_small.csv

medium: all
	./$(TARGET) --instance data/medium --population $(POP) --generations $(GEN) --threads $(THREADS) --seed $(SEED) --report-file results/report_medium.csv

large: all
	./$(TARGET) --instance data/large --population $(POP_LARGE) --generations $(GEN) --threads $(THREADS) --seed $(SEED) --report-file results/report_large.csv

# Benchmark: ejecuta ambas variantes (standard e islands)
# Uso: make bench <config> <nombre>
# Ej: make bench 1 mi_experimento
bench: all
	$(eval CONFIG := $(filter-out $@,$(MAKECMDGOALS)))
	$(eval NAME := $(word 1,$(CONFIG)))
	$(eval NUM := $(word 2,$(CONFIG)))
	@mkdir -p results
	@echo "Benchmark: config=$(NUM) nombre=$(NAME)"
	./$(TARGET) --benchmark --variant standard --config $(NUM) --seed $(SEED) --threads $(THREADS) --report-file results/$(NAME)_bench_standard_$(NUM).csv
	./$(TARGET) --benchmark --variant islands --config $(NUM) --seed $(SEED) --threads $(THREADS) --report-file results/$(NAME)_bench_islands_$(NUM).csv

clean:
	$(RM) $(call FIX_PATH, $(BUILD_DIR)/*.o)
	$(RM) $(TARGET)
	@echo Limpieza completada.

.PHONY: all small medium large bench clean
%:
	@:
