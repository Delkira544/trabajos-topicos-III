"""Esquema único del CSV de resultados, compartido por bench/ y app_streamlit.py.

Mantener aquí la fuente de verdad de las columnas evita que medición y
visualización se desincronicen.
"""

COLUMNS = [
    "version",        # cpu | cuda_classic | cuda_tile | cutile_py
    "image",          # nombre lógico (small_512, ...)
    "width", "height",
    "megapixels",
    "gauss",          # config gaussian (kN_sX)
    "scale",          # factor de resize
    "reps",
    "total_ms_mean", "total_ms_std",
    "gaussian_ms", "sobel_ms", "resize_ms",
    "kernel_ms",      # tiempo de kernels GPU (CUDA Events) — null en CPU
    "h2d_ms", "d2h_ms",  # transferencias host<->device — null en CPU
    "throughput_mpx_s",  # megapíxeles/seg (calculado por bench)
    "speedup_vs_cpu",    # vs CPU mismo (image,gauss,scale) — calculado por bench
    "tile",           # forma de tile (cuda_tile) o config bloque/grilla (cuda_classic)
    "notes",
]
