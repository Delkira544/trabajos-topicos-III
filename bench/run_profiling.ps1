# Profiling con Nsight Systems (nsys) y Nsight Compute (ncu) — §6/§8 de la pauta.
# Requiere Nsight instalado (no viene por pip):
#   - Nsight Systems:  https://developer.nvidia.com/nsight-systems  (o winget/instalador CUDA Toolkit)
#   - Nsight Compute:  https://developer.nvidia.com/nsight-compute
# Genera perfiles en results\profiles\ y un resumen de texto por herramienta.
#
# Uso:  .\bench\run_profiling.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
New-Item -ItemType Directory -Force "results\profiles" | Out-Null

$img = "data\medium_2048.png"
$args_ = @("--image", $img, "--ksize", "5", "--sigma", "1.0", "--scale", "0.5", "--reps", "3")

# localizar herramientas (PATH o instalaciones típicas)
function Find-Tool($name, $patterns) {
    $c = Get-Command $name -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    foreach ($p in $patterns) {
        $hit = Get-ChildItem $p -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    return $null
}
$nsys = Find-Tool "nsys" @("C:\Program Files\NVIDIA Corporation\Nsight Systems*\target-windows-x64\nsys.exe",
                           "C:\Program Files\NVIDIA Corporation\Nsight Systems*\host-windows-x64\nsys.exe")
$ncu  = Find-Tool "ncu"  @("C:\Program Files\NVIDIA Corporation\Nsight Compute*\ncu.exe")

if (-not $nsys -and -not $ncu) {
    Write-Warning "Ni nsys ni ncu encontrados. Instala Nsight Systems y/o Nsight Compute y reintenta."
    exit 1
}

foreach ($ver in @("cuda_classic", "cuda_tile")) {
    $exe = ".\$ver\pipeline.exe"
    if ($nsys) {
        Write-Output "== nsys $ver =="
        & $nsys profile --stats=true -o "results\profiles\nsys_$ver" --force-overwrite=true $exe @args_ 2>&1 |
            Tee-Object "results\profiles\nsys_${ver}_summary.txt" | Select-Object -Last 5
    }
    if ($ncu) {
        Write-Output "== ncu $ver (kernels, 3 lanzamientos c/u) =="
        & $ncu --set basic --launch-count 8 -o "results\profiles\ncu_$ver" --force-overwrite $exe @args_ 2>&1 |
            Tee-Object "results\profiles\ncu_${ver}_summary.txt" | Select-Object -Last 5
    }
}
Write-Output "OK -> results\profiles\"
