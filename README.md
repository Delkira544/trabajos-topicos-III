# Actividad 4 

## Principio de diseño: medir ≠ visualizar

El rendimiento se mide **siempre** en los ejecutables/scripts de cada versión (CPU Events
para CPU, **CUDA Events / Nsight** para GPU) y se vuelca a `results/results.csv`.
**Streamlit NO mide nada**: solo lee ese CSV y las imágenes ya generadas. Así la UI no
contamina los números que se evalúan.

```
bench/  +  cpu|cuda_classic|cuda_tile|cutile_py   →  results/results.csv + out/*.png   (MEDICIÓN)
                                                            ↓
                                            app_streamlit.py  (SOLO LECTURA / VISUALIZACIÓN)
```

## Contrato común entre versiones

Cada versión es un ejecutable invocable por línea de comando que procesa una imagen,
guarda las salidas en `out/` e imprime en **stdout una sola línea JSON** con sus tiempos.
`bench/run_bench.py` invoca cada versión, parsea ese JSON, calcula throughput y speed-up,
y agrega todo al CSV. Ver el esquema en `bench/schema.py`.

Campos JSON esperados (los de GPU pueden ir `null` en CPU):
```json
{"version":"cpu","image":"medium_2048","width":2048,"height":2048,
 "gauss":"k5_s1.0","scale":0.5,"reps":10,
 "total_ms_mean":..,"total_ms_std":..,
 "gaussian_ms":..,"sobel_ms":..,"resize_ms":..,
 "kernel_ms":null,"h2d_ms":null,"d2h_ms":null}
```

## Entorno (venv, Python 3.13)

El proyecto usa un entorno virtual con **Python 3.13** y corre en **Windows y Linux**
(todas las dependencias GPU vienen por pip con wheels win_amd64 y manylinux).

**Windows:**
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1        # PowerShell  (o .\.venv\Scripts\activate.bat en cmd)
pip install -r requirements.txt
```

**Linux:**
```bash
python3.13 -m venv .venv            # también sirve 3.10–3.14
source .venv/bin/activate
pip install -r requirements.txt
chmod +x cuda_classic/build.sh cuda_tile/build.sh bench/run_profiling.sh
```

Requisitos comunes: GPU NVIDIA **Ampere/Ada/Blackwell** (CC 8.x+) y **driver R580+**
(CUDA 13). Compilador host para nvcc: MSVC Build Tools 2022 en Windows, **gcc** en Linux.

> Reportar en el informe: Python 3.13.12 + versiones de `pip list` (sección 7 de la pauta).

## Toolchain GPU (RTX 4060, Ada, compute capability 8.9)

Estado verificado en este equipo:

| Componente | Estado | Notas |
|---|---|---|
| GPU RTX 4060 (Ada, CC 8.9) | ✅ | Soporta CUDA Tile C++ y cuTile Python |
| **Driver NVIDIA 610.62** | ✅ | ≥ R580 → CUDA 13.x habilitado |
| `cuda-tile` (cuTile Python) 1.4 | ✅ venv | `import cuda.tile` — **verificado en GPU** |
| `tileiras` (compilador CUDA Tile) 13.3 | ✅ venv | bundled vía `[tileiras]` |
| `nvcc` 13.3.33 | ✅ venv | `.venv\Lib\site-packages\nvidia\cu13\bin\nvcc.exe` |
| runtime CUDA 13.3 + nvrtc | ✅ venv | — |
| `cupy-cuda13x` 14.1 | ✅ venv | arrays GPU + streams (manejo de arreglos, no filtrado) |
| **MSVC Build Tools 2022** (cl.exe 14.44) | ✅ | host compiler de nvcc — **compila+corre en GPU verificado** |

**No hace falta el CUDA Toolkit del sistema:** nvcc + tileiras + runtime + nvrtc vienen por
pip dentro del venv. Estado: **toolchain completo verificado end-to-end** (CUDA C++ y cuTile).

### Compilar CUDA C++ (.cu)

**Windows** (los `build.ps1` cargan MSVC automáticamente):
```powershell
.\cuda_classic\build.ps1
.\cuda_tile\build.ps1
```

**Linux** (autodetecta la arquitectura con nvidia-smi; forzar con `GPU_ARCH=sm_XX`):
```bash
./cuda_classic/build.sh
./cuda_tile/build.sh
```

Los binarios quedan como `pipeline.exe` (Windows) o `pipeline` (Linux); `bench/` elige
el nombre correcto según la plataforma automáticamente.

## Uso

Con el venv activado (o anteponiendo `.\.venv\Scripts\python.exe`):

```powershell
# 1) Generar imágenes de prueba (512², 2048², 4096², 1537x1021)
python data/generate_images.py

# 2) Ejecutar el benchmark completo (CPU ya funciona; CUDA se enchufa luego)
python bench/run_bench.py --versions cpu --reps 10

# 3) Visualizar resultados (no mide nada, solo lee results/ y out/)
streamlit run app_streamlit.py
```

