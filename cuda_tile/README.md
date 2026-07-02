# CUDA Tile C++ — versión 3 de 4  ✅ (Gaussian + Sobel en tile)

Implementación con el modelo de programación **por tiles** (`cuda::tiles`, header
`cuda_tile.h`, compilado con `nvcc --enable-tile -std=c++20`).

## Qué está en modelo tile y qué no (y por qué)

| Etapa | Implementación | Motivo |
|---|---|---|
| **Gaussian (separable)** | **Tile** | Convolución regular: encaja en el modelo. Tiles 1D + acumulación de versiones desplazadas del puntero base (vecino x = ±3 en RGB interleaved, y = ±3W). |
| **Sobel 3×3** | **Tile** | Igual patrón sobre el plano de luminancia (vecino ±1, ±W). |
| **Resize bilineal** | Kernel clásico (apoyo) | Es un *gather* con coordenadas fraccionales dependientes de datos; la API de tiles de alto nivel **no ofrece gather/scatter**, así que no mapea de forma natural. |

## Decisiones técnicas y límites de la API (para defender)

- **Índice de bloque:** en tile code no se usa `blockIdx`; se obtiene con `ct::bid()`.
- **Lanzamiento:** `kernel<<<grid, 1>>>` — el segundo argumento debe ser 1 (el compilador
  de tiles gestiona los hilos).
- **Bordes:** `partition_view::load_masked` solo soporta padding **zero/constante**
  (no clamp). Por eso la versión tile difiere de la CPU **solo en los píxeles del borde**
  (validado: interior MAE ≈ 0.003). Ver `bench/validate.py --version cuda_tile`.
- **Halo:** los buffers se alocan con relleno (`HALO`) para que los tiles desplazados no
  lean fuera de rango.
- **`extract` es por bloques** (múltiplos del shape), no por offset de elementos → no sirve
  para desplazamientos de ±1 píxel; por eso se desplaza el puntero base en su lugar.

## Rendimiento observado

El modelo tile aquí resulta **más lento** que el CUDA clásico (separable 1D con varias
pasadas y `memset` vs. kernel 2D en bloques). Es un resultado válido y esperable: la pauta
pide explícitamente discutir que Tile no garantiza mejora.

## Compilar
```powershell
.\cuda_tile\build.ps1
```
