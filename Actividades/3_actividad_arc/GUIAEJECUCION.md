# Guía de Compilación, Ejecución y Experimentos
## Actividad 3 — Algoritmo Genético con CUDA

---

##  VALIDACIÓN DEL CMakeLists.txt

El `CMakeLists.txt` actual está **correctamente configurado**:

- ✓ C++17 y CUDA C++17 standards
- ✓ Optimizaciones `-O3 --use_fast_math`
- ✓ Detección automática de arquitectura (fallback: 86)
- ✓ OpenMP y CUDA Toolkit correctamente enlazados
- ✓ Separable compilation activada
- ✓ Targets `run` y `penalty_tuner` bien definidos
- ✓ Todas las fuentes CUDA incluidas

**No requiere cambios.**

---

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
cmake --build . --config Release -j $env:NUMBER_OF_PROCESSORS
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

## 5. Plan Experimental Completo (SEGÚN RÚBRICA)

### 5.0 Reporte de Hardware (OBLIGATORIO antes de los experimentos)

Ejecuta esto y guarda en `results/hardware.txt`:

```bash
mkdir -p ../results
echo "=== HARDWARE REPORT ===" > ../results/hardware.txt
echo "" >> ../results/hardware.txt
echo "CPU:" >> ../results/hardware.txt
wmic os get caption, version, buildnumber >> ../results/hardware.txt
wmic cpu get name, cores, threads >> ../results/hardware.txt
echo "" >> ../results/hardware.txt
echo "GPU:" >> ../results/hardware.txt
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv >> ../results/hardware.txt
echo "" >> ../results/hardware.txt
echo "CUDA Version:" >> ../results/hardware.txt
nvcc --version >> ../results/hardware.txt
echo "" >> ../results/hardware.txt
echo "Total RAM:" >> ../results/hardware.txt
wmic os get totalvirtualmemory, totalvisiblememorylsize >> ../results/hardware.txt
```

**Contenido esperado en el informe:**
- CPU: Marca, cores, threads
- GPU: Modelo, memoria total, driver version
- CUDA: versión (ej: 12.0, 11.8)
- RAM disponible

---

### 5.1 Configuración Experimental

| Parámetro | Valores |
|-----------|---------|
| **Instancias** | Pequeña (100), Mediana (1000), Grande (10000) |
| **Población** | 1024, 4096, 16384 |
| **Generaciones** | 300 (fijo para toda la actividad) |
| **Variantes** | sequential, cuda_basic, cuda_optimized |
| **Block sizes** (GPU) | 32, 64, 128, 256, 512 |
| **Semillas** | 42, 43, 44, 45, 46, 47, 48, 49, 50, 51 (10 repeticiones) |
| **Tamaño de bloque por defecto** | 128 |

---

### 5.1a ⚠️ PASO CRÍTICO: Calibrar Pesos de Penalización

**ANTES de ejecutar los experimentos, debes correr el penalty_tuner una sola vez:**

```powershell
# En el directorio build, ejecuta:
.\Release\penalty_tuner.exe -i ..\data\large -p 4096 -g 100 --verbose
```

**Esto generará output similar a:**
```
Best penalties found:
  Weight: 15.5
  Volume: 12.0
  Category: 8.3
  Incomp: 18.5
  Dep: 20.0
```

**⚠️ IMPORTANTE**: Toma esos 5 valores y actualiza TODOS los comandos `$cmd` en los scripts EXP 1-4 más abajo. Reemplaza los placeholders:
```powershell
# De esto:
$cmd = "...\run.exe ... --block-size 128"

# A esto (con TUS valores encontrados por penalty_tuner):
$cmd = "...\run.exe ... --pen-weight 15.5 --pen-volume 12.0 --pen-category 8.3 --pen-incomp 18.5 --pen-dep 20.0 --block-size 128"
```

**Por qué es crítico**: Sin estos parámetros, tu algoritmo usa valores hardcodeados (antiguos) y los 455 experimentos estarían ejecutándose con configuración subóptima, invalidando TODO el análisis.

---

### 5.2 EXP 1: Variantes × Instancias × Poblaciones × 10 Semillas

**Objetivo**: Comparar rendimiento (CPU vs GPU básico vs GPU optimizado) en 3 tamaños de instancia con 3 configuraciones de población.

**Configuración**:
- Instancias: small (100), medium (1000), large (10000)
- Poblaciones: 1024, 4096, 16384
- Generaciones: 300
- Block size (GPU): 128 (fijo)
- Semillas: 42-51 (10 repeticiones)
- **Total de ejecuciones**: 3 × 3 × 3 × 10 = 270 ejecuciones

**Script PowerShell (Windows)**:

```powershell
# ─────────────────────────────────────────────────────────────────────────────
# EXP 1: Main experimental design
# ─────────────────────────────────────────────────────────────────────────────

cd .\build

New-Item -ItemType Directory -Force -Path "..\results" | Out-Null

$csv_exp1 = "..\results\exp1_main_design.csv"
$header = "instance_size,population,variant,seed,best_fitness,feasible,wall_time_ms,kernel_fitness_ms,kernel_repro_ms,h2d_transfer_ms,d2h_transfer_ms,feasible_pct,init_fitness,final_fitness,improvement"
$header | Out-File -FilePath $csv_exp1 -Encoding utf8

$instances = @("small", "medium", "large")
$populations = @(1024, 4096, 16384)
$variants = @("sequential", "cuda_basic", "cuda_optimized")
$seeds = 42..51
$total_runs = 270
$current_run = 0

foreach ($instance in $instances) {
    foreach ($pop in $populations) {
        foreach ($variant in $variants) {
            foreach ($seed in $seeds) {
                $current_run++
                $progress = [math]::Round(($current_run / $total_runs) * 100, 1)
                Write-Host "[$progress%] EXP1 -> $instance | pop=$pop | $variant | seed=$seed"
                
                # ⚠️ ACTUALIZA estos valores con los resultados de penalty_tuner (ver sección 5.1a)
                $pen_weight = 0.2      # ← CAMBIAR
                $pen_volume = 0.2      # ← CAMBIAR
                $pen_category = 0.2    # ← CAMBIAR
                $pen_incomp = 0.2      # ← CAMBIAR
                $pen_dep = 0.2         # ← CAMBIAR
                
                $cmd = ".\ Release\run.exe -i ..\data\$instance -v $variant -t 1 -s $seed -p $pop -g 300 --pen-weight $pen_weight --pen-volume $pen_volume --pen-category $pen_category --pen-incomp $pen_incomp --pen-dep $pen_dep --block-size 128"
                $output = & $cmd 2>&1 | Out-String
                
                # Extraer métricas con regex robusta
                $fitness = if ($output -match "Best fitness:\s+([\d.-]+)") { $matches[1] } else { "" }
                $feasible = if ($output -match "Feasible:\s+(Yes|No)") { $matches[1] } else { "" }
                $wall_ms = if ($output -match "Wall-clock time \(ms\):\s*(\d+)") { $matches[1] } else { "" }
                $kfit_ms = if ($output -match "Kernel fitness\s+total \(ms\):\s+([\d.]+)") { $matches[1] } else { "" }
                $krepro_ms = if ($output -match "Kernel repro\s+total \(ms\):\s+([\d.]+)") { $matches[1] } else { "" }
                $h2d_ms = if ($output -match "Transfer H->D\s+total \(ms\):\s+([\d.]+)") { $matches[1] } else { "" }
                $d2h_ms = if ($output -match "Transfer D->H\s+total \(ms\):\s+([\d.]+)") { $matches[1] } else { "" }
                $feasible_pct = if ($output -match "Feasible solutions \(%\):\s+([\d.]+)") { $matches[1] } else { "" }
                $init_fit = if ($output -match "Initial best fitness:\s+([\d.-]+)") { $matches[1] } else { "" }
                $final_fit = if ($output -match "Final best fitness:\s+([\d.-]+)") { $matches[1] } else { "" }
                $improvement = if (($init_fit -ne "") -and ($final_fit -ne "")) { [math]::Round([float]$final_fit - [float]$init_fit, 4) } else { "" }
                
                $line = "$instance,$pop,$variant,$seed,$fitness,$feasible,$wall_ms,$kfit_ms,$krepro_ms,$h2d_ms,$d2h_ms,$feasible_pct,$init_fit,$final_fit,$improvement"
                $line | Out-File -FilePath $csv_exp1 -Append -Encoding utf8
                
                Start-Sleep -Milliseconds 100  # Pequeña pausa entre ejecuciones
            }
        }
    }
}

Write-Host "`n✓ EXP 1 completado: $csv_exp1"
cd ..
```

---

### 5.3 EXP 2: Efecto del Tamaño de Bloque (GPU)

**Objetivo**: Medir impacto del block size en performance CUDA.

**Configuración**:
- Instancia: large (10000)
- Población: 4096
- Block sizes: 32, 64, 128, 256, 512
- Variantes: cuda_basic, cuda_optimized
- Semillas: 42-46 (5 repeticiones)
- **Total**: 5 × 2 × 5 = 50 ejecuciones

```powershell
# ─────────────────────────────────────────────────────────────────────────────
# EXP 2: Block size effect
# ─────────────────────────────────────────────────────────────────────────────

cd .\build

$csv_exp2 = "..\results\exp2_block_size_effect.csv"
"instance,population,block_size,variant,seed,best_fitness,feasible,wall_time_ms,kernel_fitness_ms,kernel_repro_ms,transfer_overhead_pct" | Out-File -FilePath $csv_exp2 -Encoding utf8

$block_sizes = @(32, 64, 128, 256, 512)
$variants = @("cuda_basic", "cuda_optimized")
$seeds = 42..46
$total_runs = 50
$current_run = 0

foreach ($block in $block_sizes) {
    foreach ($variant in $variants) {
        foreach ($seed in $seeds) {
            $current_run++
            $progress = [math]::Round(($current_run / $total_runs) * 100, 1)
            Write-Host "[$progress%] EXP2 -> block=$block | $variant | seed=$seed"
            
            # ⚠️ Usa los mismos valores de penalty_tuner (ver sección 5.1a)
            $pen_weight = 0.2      # ← ACTUALIZAR
            $pen_volume = 0.2      # ← ACTUALIZAR
            $pen_category = 0.2    # ← ACTUALIZAR
            $pen_incomp = 0.2      # ← ACTUALIZAR
            $pen_dep = 0.2         # ← ACTUALIZAR
            
            $cmd = ".\ Release\run.exe -i ..\data\large -v $variant -t 1 -s $seed -p 4096 -g 300 --pen-weight $pen_weight --pen-volume $pen_volume --pen-category $pen_category --pen-incomp $pen_incomp --pen-dep $pen_dep --block-size $block"
            $output = & $cmd 2>&1 | Out-String
            
            $fitness = if ($output -match "Best fitness:\s+([\d.-]+)") { $matches[1] } else { "" }
            $feasible = if ($output -match "Feasible:\s+(Yes|No)") { $matches[1] } else { "" }
            $wall_ms = if ($output -match "Wall-clock time \(ms\):\s*(\d+)") { $matches[1] } else { "" }
            $kfit_ms = if ($output -match "Kernel fitness\s+total \(ms\):\s+([\d.]+)") { $matches[1] } else { "" }
            $krepro_ms = if ($output -match "Kernel repro\s+total \(ms\):\s+([\d.]+)") { $matches[1] } else { "" }
            $overhead = if ($output -match "Transfer overhead \(%\):\s+([\d.]+)") { $matches[1] } else { "" }
            
            "large,4096,$block,$variant,$seed,$fitness,$feasible,$wall_ms,$kfit_ms,$krepro_ms,$overhead" | Out-File -FilePath $csv_exp2 -Append -Encoding utf8
        }
    }
}

Write-Host "`n✓ EXP 2 completado: $csv_exp2"
cd ..
```

---

### 5.4 EXP 3: Speed-up Analysis

**Objetivo**: Calcular speed-up de GPU vs CPU secuencial.

**Fórmula**: `Speed-up = Time_Sequential / Time_GPU`

```powershell
# ─────────────────────────────────────────────────────────────────────────────
# EXP 3: Speed-up analysis
# ─────────────────────────────────────────────────────────────────────────────

cd .\build

$csv_exp3 = "..\results\exp3_speedup.csv"
"instance,population,seed,time_seq_ms,time_cuda_basic_ms,time_cuda_opt_ms,speedup_basic,speedup_opt" | Out-File -FilePath $csv_exp3 -Encoding utf8

$instances = @("small", "medium", "large")
$populations = @(1024, 4096, 16384)
$seeds = 42..51

foreach ($instance in $instances) {
    foreach ($pop in $populations) {
        foreach ($seed in $seeds) {
            Write-Host "EXP3 -> $instance | pop=$pop | seed=$seed"
            
            # ⚠️ Valores de penalty_tuner (ver sección 5.1a)
            $pen_weight = 0.2      # ← ACTUALIZAR
            $pen_volume = 0.2      # ← ACTUALIZAR
            $pen_category = 0.2    # ← ACTUALIZAR
            $pen_incomp = 0.2      # ← ACTUALIZAR
            $pen_dep = 0.2         # ← ACTUALIZAR
            
            $pen_flags = "--pen-weight $pen_weight --pen-volume $pen_volume --pen-category $pen_category --pen-incomp $pen_incomp --pen-dep $pen_dep"
            
            # Sequential
            $cmd_seq = ".\ Release\run.exe -i ..\data\$instance -v sequential -t 1 -s $seed -p $pop -g 300 $pen_flags"
            $out_seq = & $cmd_seq 2>&1 | Out-String
            $time_seq = if ($out_seq -match "Wall-clock time \(ms\):\s*(\d+)") { [int]$matches[1] } else { 0 }
            
            # CUDA Basic
            $cmd_bas = ".\ Release\run.exe -i ..\data\$instance -v cuda_basic -t 1 -s $seed -p $pop -g 300 $pen_flags --block-size 128"
            $out_bas = & $cmd_bas 2>&1 | Out-String
            $time_bas = if ($out_bas -match "Wall-clock time \(ms\):\s*(\d+)") { [int]$matches[1] } else { 0 }
            
            # CUDA Optimized
            $cmd_opt = ".\ Release\run.exe -i ..\data\$instance -v cuda_optimized -t 1 -s $seed -p $pop -g 300 $pen_flags --block-size 128"
            $out_opt = & $cmd_opt 2>&1 | Out-String
            $time_opt = if ($out_opt -match "Wall-clock time \(ms\):\s*(\d+)") { [int]$matches[1] } else { 0 }
            
            # Calcular speed-up
            $sp_bas = if ($time_bas -gt 0) { [math]::Round([float]$time_seq / [float]$time_bas, 2) } else { "N/A" }
            $sp_opt = if ($time_opt -gt 0) { [math]::Round([float]$time_seq / [float]$time_opt, 2) } else { "N/A" }
            
            "$instance,$pop,$seed,$time_seq,$time_bas,$time_opt,$sp_bas,$sp_opt" | Out-File -FilePath $csv_exp3 -Append -Encoding utf8
        }
    }
}

Write-Host "`n✓ EXP 3 completado: $csv_exp3"
cd ..
```

---

### 5.5 EXP 4: Efecto del Tamaño de Población

**Objetivo**: Medir impacto de population_size en calidad y tiempo.

```powershell
# ─────────────────────────────────────────────────────────────────────────────
# EXP 4: Population size effect
# ─────────────────────────────────────────────────────────────────────────────

cd .\build

$csv_exp4 = "..\results\exp4_population_effect.csv"
"instance,population,variant,seed,best_fitness,feasible,wall_time_ms,kernel_time_ms" | Out-File -FilePath $csv_exp4 -Encoding utf8

$instance = "large"
$populations = @(1024, 4096, 16384)
$variants = @("sequential", "cuda_basic", "cuda_optimized")
$seeds = 42..46

foreach ($pop in $populations) {
    foreach ($variant in $variants) {
        foreach ($seed in $seeds) {
            Write-Host "EXP4 -> pop=$pop | $variant | seed=$seed"
            
            # ⚠️ Usa los valores de penalty_tuner (ver sección 5.1a)
            $pen_weight = 0.2      # ← ACTUALIZAR
            $pen_volume = 0.2      # ← ACTUALIZAR
            $pen_category = 0.2    # ← ACTUALIZAR
            $pen_incomp = 0.2      # ← ACTUALIZAR
            $pen_dep = 0.2         # ← ACTUALIZAR
            
            $cmd = ".\ Release\run.exe -i ..\data\$instance -v $variant -t 1 -s $seed -p $pop -g 300 --pen-weight $pen_weight --pen-volume $pen_volume --pen-category $pen_category --pen-incomp $pen_incomp --pen-dep $pen_dep --block-size 128"
            $output = & $cmd 2>&1 | Out-String
            
            $fitness = if ($output -match "Best fitness:\s+([\d.-]+)") { $matches[1] } else { "" }
            $feasible = if ($output -match "Feasible:\s+(Yes|No)") { $matches[1] } else { "" }
            $wall_ms = if ($output -match "Wall-clock time \(ms\):\s*(\d+)") { $matches[1] } else { "" }
            $ktime = if ($output -match "Kernel fitness\s+total \(ms\):\s+([\d.]+)") { $matches[1] } else { "" }
            
            "$instance,$pop,$variant,$seed,$fitness,$feasible,$wall_ms,$ktime" | Out-File -FilePath $csv_exp4 -Append -Encoding utf8
        }
    }
}

Write-Host "`n✓ EXP 4 completado: $csv_exp4"
cd ..
```

---

### 5.6 Generación de Tablas para el Informe

Después de los 4 experimentos, ejecuta esto para generar tablas resumidas:

```powershell
# ─────────────────────────────────────────────────────────────────────────────
# Summary: Generate tables for report
# ─────────────────────────────────────────────────────────────────────────────

$results_dir = "..\results"

# Tabla 1: Tiempo promedio y desviación estándar por variante
Write-Host "`n=== TABLA 1: Tiempo Promedio por Variante ===" 

$csv1 = "$results_dir\exp1_main_design.csv"
$data1 = Import-Csv $csv1

$summary1 = @"
Instancia,Población,Variante,Promedio (ms),Desv.Est (ms),Min (ms),Max (ms)
"@

foreach ($instance in @("small", "medium", "large")) {
    foreach ($pop in @(1024, 4096, 16384)) {
        foreach ($variant in @("sequential", "cuda_basic", "cuda_optimized")) {
            $subset = $data1 | Where-Object { 
                $_.instance_size -eq $instance -and $_.population -eq $pop -and $_.variant -eq $variant 
            }
            $times = @($subset.wall_time_ms | ForEach-Object { if ($_ -ne "") { [float]$_ } })
            
            if ($times.Count -gt 0) {
                $avg = [math]::Round(($times | Measure-Object -Average).Average, 2)
                $stddev = [math]::Round(($times | Measure-Object -Average | ForEach-Object { 
                    ($times | ForEach-Object { [math]::Pow($_ - $avg, 2) } | Measure-Object -Average).Average 
                } | ForEach-Object { [math]::Sqrt($_) }), 2)
                $min = [math]::Round(($times | Measure-Object -Minimum).Minimum, 2)
                $max = [math]::Round(($times | Measure-Object -Maximum).Maximum, 2)
                
                $summary1 += "`n$instance,$pop,$variant,$avg,$stddev,$min,$max"
            }
        }
    }
}

$summary1 | Out-File -FilePath "$results_dir\table_timing_summary.csv" -Encoding utf8
Write-Host $summary1

# Tabla 2: Porcentaje de soluciones factibles
Write-Host "`n=== TABLA 2: Porcentaje de Soluciones Factibles ===" 

$summary2 = @"
Instancia,Población,Variante,Factibles (%)
"@

foreach ($instance in @("small", "medium", "large")) {
    foreach ($pop in @(1024, 4096, 16384)) {
        foreach ($variant in @("sequential", "cuda_basic", "cuda_optimized")) {
            $subset = $data1 | Where-Object { 
                $_.instance_size -eq $instance -and $_.population -eq $pop -and $_.variant -eq $variant 
            }
            $feasibles = @($subset.feasible_pct | ForEach-Object { if ($_ -ne "") { [float]$_ } })
            
            if ($feasibles.Count -gt 0) {
                $avg_feasible = [math]::Round(($feasibles | Measure-Object -Average).Average, 2)
                $summary2 += "`n$instance,$pop,$variant,$avg_feasible"
            }
        }
    }
}

$summary2 | Out-File -FilePath "$results_dir\table_feasibility.csv" -Encoding utf8
Write-Host $summary2

# Tabla 3: Speed-up
Write-Host "`n=== TABLA 3: Speed-up (GPU vs CPU) ===" 

$csv3 = "$results_dir\exp3_speedup.csv"
$data3 = Import-Csv $csv3

$summary3 = @"
Instancia,Población,Speed-up CUDA Basic (avg),Speed-up CUDA Opt (avg)
"@

foreach ($instance in @("small", "medium", "large")) {
    foreach ($pop in @(1024, 4096, 16384)) {
        $subset = $data3 | Where-Object { $_.instance -eq $instance -and $_.population -eq $pop }
        $speedups_basic = @($subset.speedup_basic | ForEach-Object { if ($_ -ne "N/A") { [float]$_ } })
        $speedups_opt = @($subset.speedup_opt | ForEach-Object { if ($_ -ne "N/A") { [float]$_ } })
        
        $avg_basic = if ($speedups_basic.Count -gt 0) { [math]::Round(($speedups_basic | Measure-Object -Average).Average, 2) } else { "N/A" }
        $avg_opt = if ($speedups_opt.Count -gt 0) { [math]::Round(($speedups_opt | Measure-Object -Average).Average, 2) } else { "N/A" }
        
        $summary3 += "`n$instance,$pop,$avg_basic,$avg_opt"
    }
}

$summary3 | Out-File -FilePath "$results_dir\table_speedup.csv" -Encoding utf8
Write-Host $summary3

Write-Host "`n✓ Todas las tablas generadas en $results_dir\"
```

---

### 5.7 Uso del Script Completo

```bash
# 1. Compilar el proyecto
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j8
cd ..

# 2. Ejecutar TODOS los experimentos (IMPORTANTE: tardará horas)
# Opción 1: Windows PowerShell
cd .\build
# Pegar y ejecutar cada bloque de EXP 1, 2, 3, 4 arriba

# 3. Generar tablas resumen
# Ejecutar el bloque "Summary: Generate tables for report"

# 4. Los CSVs estarán en:
# ../results/exp1_main_design.csv
# ../results/exp2_block_size_effect.csv
# ../results/exp3_speedup.csv
# ../results/exp4_population_effect.csv
# ../results/table_timing_summary.csv
# ../results/table_feasibility.csv
# ../results/table_speedup.csv
```

---

## 6. Métricas Extraídas Automáticamente (para el Informe)
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