"""Referencia CPU secuencial de la pipeline (versión 1 de 4).

Sirve como verdad-terreno de correctitud y línea base de rendimiento. Implementa las
3 etapas con numpy (NO se usan filtros de OpenCV/SciPy: el cálculo es propio):

  - gaussian_blur   : convolución separable (1D horizontal + 1D vertical), kernel normalizado
  - sobel           : RGB->luminancia, gradientes Gx/Gy 3x3, magnitud sqrt(Gx^2+Gy^2)
  - bilinear_resize : cada píxel de salida desde coord. fraccional, 4 vecinos

Borde por defecto: 'clamp' (réplica del borde, equivale a np.pad mode='edge').

Contrato de salida: guarda PNGs en out/ e imprime UNA línea JSON con los tiempos.
Las versiones GPU deben respetar este MISMO contrato para que bench/ las agregue igual.

Uso:
  python cpu/pipeline_cpu.py --image data/medium_2048.png --ksize 5 --sigma 1.0 \
      --scale 0.5 --reps 10 --border clamp
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"


# --------------------------------------------------------------------------- #
# Etapas (implementación propia)
# --------------------------------------------------------------------------- #
def gaussian_kernel_1d(ksize: int, sigma: float) -> np.ndarray:
    """Kernel gaussiano 1D normalizado (suma = 1)."""
    r = ksize // 2
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


def _conv1d_axis(img: np.ndarray, k: np.ndarray, axis: int, border: str) -> np.ndarray:
    """Convolución 1D a lo largo de un eje, acumulando por tap (k iteraciones)."""
    r = len(k) // 2
    pad = [(0, 0)] * img.ndim
    pad[axis] = (r, r)
    mode = "edge" if border == "clamp" else "constant"
    p = np.pad(img, pad, mode=mode)
    out = np.zeros_like(img, dtype=np.float32)
    for i, w in enumerate(k):
        sl = [slice(None)] * img.ndim
        sl[axis] = slice(i, i + img.shape[axis])
        out += w * p[tuple(sl)]
    return out


def gaussian_blur(img: np.ndarray, ksize: int, sigma: float, border: str) -> np.ndarray:
    """Gaussian separable: pasada horizontal y luego vertical."""
    k = gaussian_kernel_1d(ksize, sigma)
    f = img.astype(np.float32)
    f = _conv1d_axis(f, k, axis=1, border=border)   # horizontal
    f = _conv1d_axis(f, k, axis=0, border=border)   # vertical
    return np.clip(f, 0, 255).astype(np.uint8)


def to_luminance(img: np.ndarray) -> np.ndarray:
    """RGB -> luminancia (Rec. 601)."""
    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return img.astype(np.float32) @ w


def sobel(img: np.ndarray, border: str) -> np.ndarray:
    """Detección de bordes Sobel sobre luminancia; magnitud saturada a [0,255]."""
    lum = to_luminance(img)
    mode = "edge" if border == "clamp" else "constant"
    p = np.pad(lum, ((1, 1), (1, 1)), mode=mode)
    gx = (-p[:-2, :-2] + p[:-2, 2:]
          - 2 * p[1:-1, :-2] + 2 * p[1:-1, 2:]
          - p[2:, :-2] + p[2:, 2:])
    gy = (-p[:-2, :-2] - 2 * p[:-2, 1:-1] - p[:-2, 2:]
          + p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:])
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return np.clip(mag, 0, 255).astype(np.uint8)


def bilinear_resize(img: np.ndarray, scale: float) -> np.ndarray:
    """Resize bilineal. Mapea cada píxel de salida a coordenada fraccional de entrada."""
    h, w = img.shape[:2]
    ch = img.shape[2] if img.ndim == 3 else 1
    # round half-up (igual que lroundf en C++) para que CPU y CUDA generen el mismo tamaño
    nh, nw = max(1, int(np.floor(h * scale + 0.5))), max(1, int(np.floor(w * scale + 0.5)))
    f = img.astype(np.float32).reshape(h, w, ch)

    # centro de píxel: (out + 0.5)/scale - 0.5  -> coords fraccionales en la entrada
    ys = (np.arange(nh, dtype=np.float32) + 0.5) / scale - 0.5
    xs = (np.arange(nw, dtype=np.float32) + 0.5) / scale - 0.5
    y0 = np.floor(ys).astype(np.int32); x0 = np.floor(xs).astype(np.int32)
    dy = (ys - y0)[:, None, None]; dx = (xs - x0)[None, :, None]

    y0c = np.clip(y0, 0, h - 1); y1c = np.clip(y0 + 1, 0, h - 1)
    x0c = np.clip(x0, 0, w - 1); x1c = np.clip(x0 + 1, 0, w - 1)

    A = f[y0c][:, x0c]; B = f[y0c][:, x1c]   # arriba-izq, arriba-der
    C = f[y1c][:, x0c]; D = f[y1c][:, x1c]   # abajo-izq, abajo-der
    out = (A * (1 - dx) * (1 - dy) + B * dx * (1 - dy)
           + C * (1 - dx) * dy + D * dx * dy)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out.reshape(nh, nw, ch) if img.ndim == 3 else out.reshape(nh, nw)


# --------------------------------------------------------------------------- #
# Arnés de medición (CPU): mide cada etapa con perf_counter, reps repeticiones
# --------------------------------------------------------------------------- #
def time_ms(fn, reps: int):
    """Devuelve (resultado_última_corrida, mean_ms, std_ms) sobre 'reps' repeticiones."""
    ts = []
    res = None
    for _ in range(reps):
        t0 = time.perf_counter()
        res = fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(ts)
    return res, float(a.mean()), float(a.std())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ksize", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--border", default="clamp", choices=["clamp", "zero"])
    ap.add_argument("--save", action="store_true", help="guardar imágenes de salida en out/")
    args = ap.parse_args()

    img = np.asarray(Image.open(args.image).convert("RGB"))
    h, w = img.shape[:2]
    name = Path(args.image).stem

    g, g_ms, _ = time_ms(lambda: gaussian_blur(img, args.ksize, args.sigma, args.border), args.reps)
    s, s_ms, _ = time_ms(lambda: sobel(g, args.border), args.reps)
    r, r_ms, _ = time_ms(lambda: bilinear_resize(g, args.scale), args.reps)

    # tiempo total = suma de etapas, repetido para media/desv estándar globales
    def full():
        gg = gaussian_blur(img, args.ksize, args.sigma, args.border)
        sobel(gg, args.border)
        bilinear_resize(gg, args.scale)
    _, tot_ms, tot_std = time_ms(full, args.reps)

    if args.save:
        OUT.mkdir(exist_ok=True)
        tag = f"{name}__cpu__k{args.ksize}_s{args.sigma}__x{args.scale}"
        Image.fromarray(g).save(OUT / f"{tag}__1gauss.png")
        Image.fromarray(s).save(OUT / f"{tag}__2sobel.png")
        Image.fromarray(r.squeeze()).save(OUT / f"{tag}__3resize.png")

    print(json.dumps({
        "version": "cpu", "image": name, "width": w, "height": h,
        "gauss": f"k{args.ksize}_s{args.sigma}", "scale": args.scale, "reps": args.reps,
        "total_ms_mean": round(tot_ms, 4), "total_ms_std": round(tot_std, 4),
        "gaussian_ms": round(g_ms, 4), "sobel_ms": round(s_ms, 4), "resize_ms": round(r_ms, 4),
        "kernel_ms": None, "h2d_ms": None, "d2h_ms": None,
    }))


if __name__ == "__main__":
    main()
