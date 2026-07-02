# Compila pipeline.cu (CUDA C++ clásico) usando el nvcc del venv + MSVC Build Tools.
# Uso:  .\cuda_classic\build.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$nvcc = Join-Path $root ".venv\Lib\site-packages\nvidia\cu13\bin\nvcc.exe"
$vc   = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$src  = Join-Path $PSScriptRoot "pipeline.cu"
$out  = Join-Path $PSScriptRoot "pipeline.exe"

if (-not (Test-Path $nvcc)) { throw "nvcc no encontrado: $nvcc" }
if (-not (Test-Path $vc))   { throw "vcvars64.bat no encontrado: $vc" }

# -arch=sm_89 = RTX 4060 (Ada); -cudart static => exe sin dependencia de cudart DLL
cmd /c "call `"$vc`" >nul 2>&1 && `"$nvcc`" -O3 -std=c++17 -arch=sm_89 -cudart static -o `"$out`" `"$src`""
if ($LASTEXITCODE -ne 0) { throw "Compilacion fallida (exit $LASTEXITCODE)" }
Write-Output "OK -> $out"
