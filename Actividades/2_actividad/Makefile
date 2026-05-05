# 1. DETECCIÓN DEL SISTEMA OPERATIVO
ifeq ($(OS), Windows_NT)
    # Configuración para Windows
    RM = del /Q
    MKDIR = mkdir
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

run small:
	./$(TARGET) --instance data/small \
				--variant islands \
				--threads 8 \
				--seed 123\

# Limpieza profunda
clean:
	$(RM) $(call FIX_PATH, $(BUILD_DIR)/*.o)
	$(RM) $(TARGET)
	@echo Limpieza completada.

.PHONY: all run clean
