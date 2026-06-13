# Guía de Compilación, Ejecución y Experimentos
## Actividad 3 — Algoritmo Genético con CUDA

---
##  VALIDACIÓN DEL CMakeLists.txt

El `CMakeLists.txt` actual está:
- ✓ C++17 y CUDA C++17 standards
- ✓ Optimizaciones `-O3 --use_fast_math`
- ✓ Detección automática de arquitectura
- ✓ OpenMP y CUDA Toolkit correctamente enlazados
- ✓ Separable compilation activada

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

## 5. Plan Experimental 

### 5.0 Reporte de Hardware
Ejecuta esto y guarda en `results/hardware.txt`:

```powershell
mkdir ..\results -Force
"=== HARDWARE REPORT ===" > ..\results\hardware.txt
"" >> ..\results\hardware.txt
"CPU:" >> ..\results\hardware.txt
Get-CimInstance Win32_Processor |
    Select-Object Name, NumberOfCores, NumberOfLogicalProcessors |
    Out-File ..\results\hardware.txt -Append
"" >> ..\results\hardware.txt
"GPU:" >> ..\results\hardware.txt
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv |
    Out-File ..\results\hardware.txt -Append
"" >> ..\results\hardware.txt
"CUDA Version:" >> ..\results\hardware.txt
nvcc --version | Out-File ..\results\hardware.txt -Append
"" >> ..\results\hardware.txt
"RAM:" >> ..\results\hardware.txt
Get-CimInstance Win32_ComputerSystem |
    Select-Object TotalPhysicalMemory |
    Out-File ..\results\hardware.txt -Append
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

### 5.1a  Calibrar Pesos de Penalización
**Valores Óptimos Encontrados:**
```
Weight Excess:      0.25
Volume Excess:      0.0
Category Violation: 0.25
Incompatibility:    0.25
Dependency:         0.25
```
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
New-Item -ItemType Directory -Force -Path "..\results" | Out-Null

# ─────────────────────────────────────────────────────────────────────────────
# EXP 1: (Diseño principal)
# ─────────────────────────────────────────────────────────────────────────────
$csv_exp1 = "..\results\exp1_main_design.csv"
$header = "instance_size,population,variant,seed,best_fitness,feasible,wall_time_ms,kernel_fitness_ms,kernel_repro_ms,h2d_transfer_ms,d2h_transfer_ms,feasible_pct,init_fitness,final_fitness,improvement"
$header | Out-File -FilePath $csv_exp1 -Encoding utf8

$instances = @("small", "medium", "large")
$populations = @(512, 1024, 4096)
$variants = @("sequential", "cuda_basic", "cuda_optimized")
$seeds = 42..51
$total_runs_exp1 = 270
$current_run = 0

# Valores óptimos del penalty_tuner
$pen_weight = 0.25
$pen_volume = 0.00
$pen_category = 0.25
$pen_incomp = 0.25
$pen_dep = 0.25

foreach ($instance in $instances) {
    foreach ($pop in $populations) {
        foreach ($variant in $variants) {
            foreach ($seed in $seeds) {
                $current_run++
                $progress = [math]::Round(($current_run / $total_runs_exp1) * 100, 1)
                Write-Host "[$progress%] EXP1 -> $instance | pop=$pop | $variant | seed=$seed"
                
                $cmd = ".\Release\run.exe -i ..\data\$instance -v $variant -t 1 -s $seed -p $pop -g 300 --pen-weight $pen_weight --pen-volume $pen_volume --pen-category $pen_category --pen-incomp $pen_incomp --pen-dep $pen_dep --block-size 128"
                $output = & cmd.exe /c $cmd 2>&1 | Out-String
                
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
            }
        }
    }
}
Write-Host "`n✓ EXP 1 completado: $csv_exp1"

# ─────────────────────────────────────────────────────────────────────────────
# EXP 2:(Efecto del tamaño de bloque)
# ─────────────────────────────────────────────────────────────────────────────
$csv_exp2 = "..\results\exp2_block_size_effect.csv"
"instance,population,block_size,variant,seed,best_fitness,feasible,wall_time_ms,kernel_fitness_ms,kernel_repro_ms,transfer_overhead_pct" | Out-File -FilePath $csv_exp2 -Encoding utf8

$block_sizes = @(32, 64, 128, 256)
$variants_cuda = @("cuda_basic", "cuda_optimized")
# Se usan 10 semillas para cumplir con el requerimiento del PDF estrictamente
$total_runs_exp2 = $block_sizes.Count * $variants_cuda.Count * $seeds.Count
$current_run = 0

foreach ($block in $block_sizes) {
    foreach ($variant in $variants_cuda) {
        foreach ($seed in $seeds) {
            $current_run++
            $progress = [math]::Round(($current_run / $total_runs_exp2) * 100, 1)
            Write-Host "[$progress%] EXP2 -> block=$block | $variant | seed=$seed"
            
            $cmd = ".\Release\run.exe -i ..\data\large -v $variant -t 1 -s $seed -p 4096 -g 300 --pen-weight $pen_weight --pen-volume $pen_volume --pen-category $pen_category --pen-incomp $pen_incomp --pen-dep $pen_dep --block-size $block"
            $output = & cmd.exe /c $cmd 2>&1 | Out-String
            
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

# ─────────────────────────────────────────────────────────────────────────────
# EXP 3: Speed-up analysis 
# ─────────────────────────────────────────────────────────────────────────────
Write-Host "`nGenerando EXP 3 cruzando datos de EXP 1..."
$csv_exp3 = "..\results\exp3_speedup.csv"
"instance,population,speedup_basic,speedup_opt" | Out-File -FilePath $csv_exp3 -Encoding utf8

$data_exp1 = Import-Csv $csv_exp1

foreach ($instance in $instances) {
    foreach ($pop in $populations) {
        # Promedios de tiempo
        $seq_times = $data_exp1 | Where-Object { $_.instance_size -eq $instance -and $_.population -eq $pop -and $_.variant -eq "sequential" -and $_.wall_time_ms -ne "" }
        $bas_times = $data_exp1 | Where-Object { $_.instance_size -eq $instance -and $_.population -eq $pop -and $_.variant -eq "cuda_basic" -and $_.wall_time_ms -ne "" }
        $opt_times = $data_exp1 | Where-Object { $_.instance_size -eq $instance -and $_.population -eq $pop -and $_.variant -eq "cuda_optimized" -and $_.wall_time_ms -ne "" }
        
        $avg_seq = if ($seq_times) { ($seq_times | Measure-Object -Property wall_time_ms -Average).Average } else { 0 }
        $avg_bas = if ($bas_times) { ($bas_times | Measure-Object -Property wall_time_ms -Average).Average } else { 0 }
        $avg_opt = if ($opt_times) { ($opt_times | Measure-Object -Property wall_time_ms -Average).Average } else { 0 }
        
        $sp_bas = if ($avg_bas -gt 0) { [math]::Round([float]$avg_seq / [float]$avg_bas, 2) } else { "N/A" }
        $sp_opt = if ($avg_opt -gt 0) { [math]::Round([float]$avg_seq / [float]$avg_opt, 2) } else { "N/A" }
        
        "$instance,$pop,$sp_bas,$sp_opt" | Out-File -FilePath $csv_exp3 -Append -Encoding utf8
    }
}
Write-Host "✓ EXP 3 completado: $csv_exp3"

# ─────────────────────────────────────────────────────────────────────────────
# EXP 4: Population size effect 
# ─────────────────────────────────────────────────────────────────────────────
Write-Host "`nGenerando EXP 4 cruzando datos de EXP 1..."
$csv_exp4 = "..\results\exp4_population_effect.csv"
"instance,population,variant,seed,best_fitness,feasible,wall_time_ms,kernel_time_ms" | Out-File -FilePath $csv_exp4 -Encoding utf8

$data_exp4 = $data_exp1 | Where-Object { $_.instance_size -eq "large" }

foreach ($row in $data_exp4) {
    $ktime = 0
    # Sumar tiempos de kernel (fitness + repro) si existen en el registro
    if ($row.kernel_fitness_ms -ne "" -and $row.kernel_repro_ms -ne "") {
        $ktime = [math]::Round([float]$row.kernel_fitness_ms + [float]$row.kernel_repro_ms, 2)
    }
    
    "$($row.instance_size),$($row.population),$($row.variant),$($row.seed),$($row.best_fitness),$($row.feasible),$($row.wall_time_ms),$ktime" | Out-File -FilePath $csv_exp4 -Append -Encoding utf8
}
Write-Host "✓ EXP 4 completado: $csv_exp4"

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY: Tablas para el informe
# ─────────────────────────────────────────────────────────────────────────────
Write-Host "`n=== Generando Tablas de Resumen para tu Informe ==="
$summary1 = "Instancia,Población,Variante,Promedio (ms),Desv.Est (ms),Min (ms),Max (ms)"
foreach ($instance in $instances) {
    foreach ($pop in $populations) {
        foreach ($variant in $variants) {
            $subset = $data_exp1 | Where-Object { $_.instance_size -eq $instance -and $_.population -eq $pop -and $_.variant -eq $variant -and $_.wall_time_ms -ne "" }
            $times = @($subset.wall_time_ms | ForEach-Object { [float]$_ })
            if ($times.Count -gt 0) {
                $avg = [math]::Round(($times | Measure-Object -Average).Average, 2)
                $stddev = [math]::Round(($times | Measure-Object -Average | ForEach-Object { ($times | ForEach-Object { [math]::Pow($_ - $avg, 2) } | Measure-Object -Average).Average } | ForEach-Object { [math]::Sqrt($_) }), 2)
                $min = [math]::Round(($times | Measure-Object -Minimum).Minimum, 2)
                $max = [math]::Round(($times | Measure-Object -Maximum).Maximum, 2)
                $summary1 += "`n$instance,$pop,$variant,$avg,$stddev,$min,$max"
            }
        }
    }
}
$summary1 | Out-File -FilePath "..\results\table_timing_summary.csv" -Encoding utf8

$summary2 = "Instancia,Población,Variante,Factibles (%)"
foreach ($instance in $instances) {
    foreach ($pop in $populations) {
        foreach ($variant in $variants) {
            $subset = $data_exp1 | Where-Object { $_.instance_size -eq $instance -and $_.population -eq $pop -and $_.variant -eq $variant -and $_.feasible_pct -ne "" }
            $feasibles = @($subset.feasible_pct | ForEach-Object { [float]$_ })
            if ($feasibles.Count -gt 0) {
                $avg_feasible = [math]::Round(($feasibles | Measure-Object -Average).Average, 2)
                $summary2 += "`n$instance,$pop,$variant,$avg_feasible"
            }
        }
    }
}
$summary2 | Out-File -FilePath "..\results\table_feasibility.csv" -Encoding utf8

Write-Host "Tablas de resumen guardadas en ..\results\"
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