"""
Versión paralela de AES-CTR para la actividad de paralelización.

Qué hace este archivo:
- Cifra y descifra archivos binarios usando AES-CTR de forma paralela.
- Utiliza ProcessPoolExecutor con asignación dinámica de tareas (dynamic scheduling).
- Cada chunk se cifra de forma independiente calculando el offset de contador correcto,
  garantizando un resultado byte a byte idéntico a la versión secuencial.
- Permite configurar el número de workers y el tamaño de chunk por CLI.

Estrategia de paralelismo:
- AES-CTR permite procesar bloques de forma independiente porque el keystream
  del bloque i solo depende del contador i, no del bloque i-1.
- Se divide el archivo en chunks de tamaño fijo (múltiplo de 16 bytes).
- Cada chunk se despacha a un worker con su block_offset precalculado.
- Los futures se mantienen en orden de submission para reensamblar el archivo
  en el orden correcto.

Asignación dinámica:
- Se usa executor.submit() uno a uno, no map(chunksize=...) con partición estática.
- Los workers toman tareas del pool interno de ProcessPoolExecutor conforme quedan
  libres, sin conocer de antemano cuántas tareas hay ni cuáles les tocarán.

Dependencia:
    pip install pycryptodome

También pueden usar el requirements.txt.

Ejemplos:
    python cifrado_aes_ctr_paralelo.py enc info_clav_10.csv info_clav_10_ctr.enc clave-demo
    python cifrado_aes_ctr_paralelo.py dec info_clav_10_ctr.enc info_clav_10_ctr_dec.csv clave-demo
    python cifrado_aes_ctr_paralelo.py enc info_clav_10.csv info_clav_10_ctr.enc clave-demo --workers 8
    python cifrado_aes_ctr_paralelo.py enc info_clav_10.csv info_clav_10_ctr.enc clave-demo --workers 4 --chunk-size 2097152
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, Future

from Crypto.Cipher import AES

DEFAULT_NONCE = b"CTRnonce"          # 8 bytes, igual que la versión secuencial
DEFAULT_CHUNK_SIZE = 1024 * 1024     # 1 MiB = 1 048 576 bytes (múltiplo de 16)
DEFAULT_WORKERS = os.cpu_count() or 4


def derive_key_from_passphrase(passphrase: str, key_size: int = 32) -> bytes:
    if key_size not in (16, 24, 32):
        raise ValueError("key_size debe ser 16, 24 o 32")
    digest = hashlib.sha256(passphrase.encode("utf-8")).digest()
    return digest[:key_size]


def encrypt_chunk_worker(
    chunk_data: bytes,
    key: bytes,
    nonce: bytes,
    block_offset: int,
) -> bytes:
    """
    Cifra (o descifra) chunk_data con AES-CTR iniciando el contador en block_offset.

    Función definida a nivel de módulo (no closure, no método) para ser
    serializable por pickle, requisito de ProcessPoolExecutor en Windows.

    Args:
        chunk_data:   Bytes a cifrar/descifrar.
        key:          Clave AES de 16, 24 o 32 bytes.
        nonce:        Nonce de 8 bytes para AES-CTR.
        block_offset: Valor inicial del contador (= byte_offset // 16).

    Returns:
        Bytes cifrados/descifrados del mismo tamaño que chunk_data.
    """
    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce, initial_value=block_offset)
    return cipher.encrypt(chunk_data)


def transform_file_ctr_parallel(
    input_path: str,
    output_path: str,
    passphrase: str,
    n_workers: int = DEFAULT_WORKERS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    nonce: bytes = DEFAULT_NONCE,
) -> float:
    """
    Cifra o descifra input_path en output_path usando AES-CTR de forma paralela.

    Asignación dinámica: se hace submit() de cada chunk conforme se lee el archivo.
    Los workers del pool toman tareas a medida que quedan disponibles.
    Los resultados se reensamblan en orden de submission para preservar la
    integridad del archivo.

    Args:
        input_path:  Ruta al archivo de entrada.
        output_path: Ruta al archivo de salida.
        passphrase:  Frase de contraseña para derivar la clave AES-256.
        n_workers:   Número máximo de procesos worker.
        chunk_size:  Tamaño en bytes de cada chunk (debe ser múltiplo de 16).
        nonce:       Nonce de 8 bytes para AES-CTR.

    Returns:
        Tiempo de ejecución en segundos (float).
    """
    if chunk_size % AES.block_size != 0:
        raise ValueError(
            f"chunk_size ({chunk_size}) debe ser múltiplo de AES.block_size ({AES.block_size})"
        )

    key = derive_key_from_passphrase(passphrase, key_size=32)

    start_time = time.perf_counter()

    with open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures: list[Future[bytes]] = []
            byte_offset: int = 0

            # — Fase de submission: despachar chunks dinámicamente —
            while True:
                chunk = f_in.read(chunk_size)
                if not chunk:
                    break
                block_offset = byte_offset // AES.block_size
                future = executor.submit(
                    encrypt_chunk_worker, chunk, key, nonce, block_offset
                )
                futures.append(future)
                byte_offset += len(chunk)

        # — Fase de escritura: reensamblar en orden de submission —
        for future in futures:
            f_out.write(future.result())

    end_time = time.perf_counter()
    return end_time - start_time


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cifrado/descifrado AES-CTR paralelo con ProcessPoolExecutor"
    )
    parser.add_argument(
        "operation",
        choices=["enc", "dec"],
        help="enc para cifrar, dec para descifrar",
    )
    parser.add_argument("input",      help="Archivo de entrada")
    parser.add_argument("output",     help="Archivo de salida")
    parser.add_argument("passphrase", help="Frase de contraseña")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Número de workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        dest="chunk_size",
        help=(
            f"Tamaño de chunk en bytes (default: {DEFAULT_CHUNK_SIZE}). "
            "Debe ser múltiplo de 16."
        ),
    )
    args = parser.parse_args()

    elapsed = transform_file_ctr_parallel(
        args.input,
        args.output,
        args.passphrase,
        n_workers=args.workers,
        chunk_size=args.chunk_size,
    )
    print(
        f"[Paralelo] AES-CTR {args.operation} | "
        f"workers={args.workers} | chunk_size={args.chunk_size} B | "
        f"tiempo={elapsed:.6f} s"
    )


if __name__ == "__main__":
    main()
