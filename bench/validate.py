"""Validación numérica: compara las salidas de una versión GPU contra la referencia CPU.

Lee los PNG ya generados en out/ para la misma (imagen, ksize, sigma, scale) y reporta
MAE y error máximo por etapa. Útil para la "validación visual y numérica" de la pauta.

Uso:
  python bench/validate.py --image data/medium_2048.png --ksize 5 --sigma 1.0 --scale 0.5 \
      --version cuda_classic
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
STAGES = ["1gauss", "2sobel", "3resize"]


def tag(version, name, k, s, scale):
    return f"{name}__{version}__k{k}_s{s:.1f}__x{scale:g}"


def load(p):
    return np.asarray(Image.open(p)).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ksize", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--version", default="cuda_classic")
    ap.add_argument("--ref", default="cpu")
    ap.add_argument("--margin", type=int, default=None,
                    help="px de borde a excluir para el MAE de interior (default=ksize). "
                         "Útil para la versión tile, cuyos bordes usan zero-pad en vez de clamp.")
    args = ap.parse_args()
    name = Path(args.image).stem
    margin = args.margin if args.margin is not None else args.ksize

    print(f"Validación {args.version} vs {args.ref}  ({name} k{args.ksize} s{args.sigma} x{args.scale})")
    print(f"{'etapa':10s} {'MAE':>9s} {'max':>6s} | {'MAE_int':>9s} {'max_int':>7s}  estado")
    ok_all = True
    for st in STAGES:
        pa = OUT / f"{tag(args.ref, name, args.ksize, args.sigma, args.scale)}__{st}.png"
        pb = OUT / f"{tag(args.version, name, args.ksize, args.sigma, args.scale)}__{st}.png"
        if not pa.exists() or not pb.exists():
            print(f"{st:10s}  (falta PNG: corre ambas versiones con --save)")
            ok_all = False
            continue
        a, b = load(pa), load(pb)
        if a.shape != b.shape:
            print(f"{st:10s}  shapes distintos {a.shape} vs {b.shape}")
            ok_all = False
            continue
        d = np.abs(a - b)
        mae, mx = float(d.mean()), float(d.max())
        # interior: excluye 'margin' px de cada borde (donde tile usa zero-pad/cruce de fila)
        m = margin if st != "3resize" else max(1, int(margin * args.scale))
        di = d[m:-m, m:-m] if d.shape[0] > 2*m and d.shape[1] > 2*m else d
        mae_i, mx_i = float(di.mean()), float(di.max())
        ok = mae_i < 1.0 and mx_i <= 4      # se valida el INTERIOR
        ok_all &= ok
        print(f"{st:10s} {mae:9.4f} {mx:6.0f} | {mae_i:9.4f} {mx_i:7.0f}  {'OK' if ok else 'REVISAR'}")
    print("RESULTADO (interior):", "coincide con CPU" if ok_all else "hay diferencias a revisar")


if __name__ == "__main__":
    main()
