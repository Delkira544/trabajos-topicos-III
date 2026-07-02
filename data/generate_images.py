"""Genera las imágenes de prueba exigidas por el diseño experimental (sección 7).

Tamaños: pequeña 512x512, mediana 2048x2048, grande 4096x4096 y una NO divisible
por tiles típicos (1537x1021). Se sintetizan patrones con bordes, gradientes y
detalle fino para que Gaussian/Sobel/resize muestren efecto visible.

Uso:  python data/generate_images.py
"""
from pathlib import Path
import numpy as np
from PIL import Image

OUT = Path(__file__.replace("\\", "/")).resolve().parent
SIZES = {
    "small_512":      (512, 512),
    "medium_2048":    (2048, 2048),
    "large_4096":     (4096, 4096),
    "nondiv_1537x1021": (1537, 1021),  # no divisible por tiles 16/32
}


def synth(w: int, h: int) -> np.ndarray:
    """Patrón RGB determinista: gradientes + tablero + círculos = bordes y textura."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = xx / w, yy / h
    r = (255 * u)                                   # gradiente horizontal
    g = (255 * v)                                   # gradiente vertical
    board = (((xx // 32) + (yy // 32)) % 2) * 90    # tablero -> bordes nítidos
    cx, cy = w / 2, h / 2
    rings = (np.sin(np.hypot(xx - cx, yy - cy) / 12.0) * 0.5 + 0.5) * 255  # detalle fino
    b = 0.5 * board + 0.5 * rings
    img = np.stack([r, g, b], axis=-1)
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> None:
    for name, (w, h) in SIZES.items():
        arr = synth(w, h)
        path = OUT / f"{name}.png"
        Image.fromarray(arr, "RGB").save(path)
        print(f"  generada {path.name}  ({w}x{h})")
    print(f"OK -> imágenes en {OUT}")


if __name__ == "__main__":
    main()
