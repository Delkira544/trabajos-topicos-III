# Compila pipeline.cu (CUDA Tile C++) con nvcc del venv + MSVC, modelo de tiles.
# Uso:  .\cuda_tile\build.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$nvcc = Join-Path $root ".venv\Lib\site-packages\nvidia\cu13\bin\nvcc.exe"
$vc   = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$src  = Join-Path $PSScriptRoot "pipeline.cu"
$out  = Join-Path $PSScriptRoot "pipeline.exe"
if (-not (Test-Path $nvcc)) { throw "nvcc no encontrado: $nvcc" }

# --enable-tile habilita el modelo de tiles; requiere -std=c++20
cmd /c "call `"$vc`" >nul 2>&1 && `"$nvcc`" -O3 -std=c++20 --enable-tile -arch=sm_89 -cudart static -o `"$out`" `"$src`""
if ($LASTEXITCODE -ne 0) { throw "Compilacion fallida (exit $LASTEXITCODE)" }
Write-Output "OK -> $out"
