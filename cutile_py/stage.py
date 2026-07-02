"""Etapa en cuTile Python (versión 4 de 4): Gaussian blur separable.

Implementa el desenfoque gaussiano con el MISMO modelo de tiles que la versión C++,
pero desde Python con `cuda.tile`: tiles 1D y acumulación de versiones desplazadas del
puntero base (vecino x = ±3 en RGB interleaved, y = ±3W). CuPy se usa SOLO para alocar
arreglos en GPU y copiar host<->device (permitido); el filtro lo hace el kernel cuTile.

Bordes: zero-pad (igual que la versión Tile C++). Mide con CUDA Events (cupy) y guarda
out/<img>__cutile_py__...__1gauss.png. Imprime 1 línea JSON (contrato bench/).

Uso:
  python cutile_py/stage.py --image data/medium_2048.png --ksize 5 --sigma 1.0 \
      --scale 0.5 --reps 10
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import cupy as cp
from PIL import Image
import cuda.tile as ct

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
TN = 256  # tamaño de tile 1D


@ct.kernel
def gauss_pass(c, r1, l1, r2, l2, r3, l3, r4, l4, out, w0, w1, w2, w3, w4):
    """Una pasada 1D: out[i] = Σ_t w[t]·(vecino +t + vecino -t). Pesos simétricos.
    Taps no usados llegan con peso 0 y vistas = c (suman 0)."""
    i = ct.bid(0)
    acc = w0 * ct.load(c, (i,), (TN,))
    acc = acc + w1 * (ct.load(r1, (i,), (TN,)) + ct.load(l1, (i,), (TN,)))
    acc = acc + w2 * (ct.load(r2, (i,), (TN,)) + ct.load(l2, (i,), (TN,)))
    acc = acc + w3 * (ct.load(r3, (i,), (TN,)) + ct.load(l3, (i,), (TN,)))
    acc = acc + w4 * (ct.load(r4, (i,), (TN,)) + ct.load(l4, (i,), (TN,)))
    ct.store(out, (i,), acc)


def run_pass(imgf_pad, halo, L, step, weights, strm):
    """Lanza gauss_pass sobre un buffer con halo; devuelve resultado float (len L padded)."""
    ntiles = (L + TN - 1) // TN
    out = cp.zeros(ntiles * TN, dtype=cp.float32)
    base = halo
    # vistas desplazadas (cupy slice = vista con offset de puntero)
    c = imgf_pad[base:]
    def sh(t):  # par (derecha, izquierda) desplazado t pasos
        return imgf_pad[base + step * t:], imgf_pad[base - step * t:]
    r1, l1 = sh(1); r2, l2 = sh(2); r3, l3 = sh(3); r4, l4 = sh(4)
    ct.launch(strm, (ntiles,), gauss_pass,
              (c, r1, l1, r2, l2, r3, l3, r4, l4, out, *weights))
    return out[:L]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ksize", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=0.5)  # ignorado (etapa = gaussian)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--tn", type=int, default=256, choices=[128, 256, 512],
                    help="tamaño de tile 1D (el kernel se JIT-compila con este valor)")
    args = ap.parse_args()
    global TN
    TN = args.tn   # el kernel se compila en el primer launch leyendo esta constante
    r = args.ksize // 2
    if r > 4:
        raise SystemExit("ksize>9 no soportado")

    img = np.asarray(Image.open(args.image).convert("RGB"))
    h, w = img.shape[:2]
    L = h * w * 3
    name = Path(args.image).stem

    # pesos gaussianos 1D normalizados, simétricos -> W[0]=centro, W[t]=tap t
    k = np.exp(-(np.arange(-r, r + 1) ** 2) / (2.0 * args.sigma ** 2))
    k /= k.sum()
    W = [0.0] * 5
    W[0] = float(k[r])
    for t in range(1, r + 1):
        W[t] = float(k[r + t])

    HALO = 3 * w * 4 + TN + 16          # cubre shift vertical 3W·4 + tamaño de tile
    strm = cp.cuda.Stream()

    host_f = img.reshape(-1).astype(np.float32)   # interleaved RGB -> float (L,)

    def timed(fn, reps):
        ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
        ts, res = [], None
        for _ in range(reps):
            with strm:
                ev0.record(strm); res = fn(); ev1.record(strm)
            ev1.synchronize()
            ts.append(cp.cuda.get_elapsed_time(ev0, ev1))
        return res, float(np.mean(ts)), float(np.std(ts))

    # H2D
    d_in, h2d, _ = timed(lambda: cp.asarray(host_f), args.reps)

    # Gaussian (2 pasadas tile) ---------------------------------------------------
    def gaussian():
        inpad = cp.zeros(L + 2 * HALO, dtype=cp.float32)
        inpad[HALO:HALO + L] = d_in
        tmp = run_pass(inpad, HALO, L, 3, W, strm)        # horizontal (step=3)
        tmppad = cp.zeros(L + 2 * HALO, dtype=cp.float32)
        tmppad[HALO:HALO + L] = tmp
        out = run_pass(tmppad, HALO, L, 3 * w, W, strm)   # vertical (step=3W)
        return cp.clip(out, 0, 255)
    gaussian(); strm.synchronize()   # warm-up: compila el kernel (JIT) fuera de la medición
    d_out, g_ms, g_sd = timed(gaussian, args.reps)

    # total y D2H
    res, d2h, _ = timed(lambda: cp.asnumpy(d_out), args.reps)
    tot_mean = h2d + g_ms + d2h

    OUT.mkdir(exist_ok=True)
    gauss_u8 = res.astype(np.uint8).reshape(h, w, 3)
    tag = f"{name}__cutile_py__k{args.ksize}_s{args.sigma:.1f}__x{args.scale:g}"
    Image.fromarray(gauss_u8).save(OUT / f"{tag}__1gauss.png")

    print(json.dumps({
        "version": "cutile_py", "image": name, "width": w, "height": h,
        "gauss": f"k{args.ksize}_s{args.sigma:.1f}", "scale": args.scale, "reps": args.reps,
        "total_ms_mean": round(tot_mean, 4), "total_ms_std": round(g_sd, 4),
        "gaussian_ms": round(g_ms, 4), "sobel_ms": None, "resize_ms": None,
        "kernel_ms": round(g_ms, 4), "h2d_ms": round(h2d, 4), "d2h_ms": round(d2h, 4),
        "tile": f"tile1D_{TN}",
    }))


if __name__ == "__main__":
    main()
