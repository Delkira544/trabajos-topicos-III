"""
!!!! Importante Favor de leer !!!!
Base secuencial de AES-CTR para la actividad de paralelización.

Qué hace este archivo:
- cifra y descifra archivos binarios usando AES-CTR
- sirve como línea base secuencial para comparar rendimiento
- será el punto de partida para que el estudiante paralelice el trabajo por chunks

Dependencia:
    pip install pycryptodome

Tambien pueden usar el requirements.txt

Ejemplos:
    python cifrado_aes_ctr_secuencial.py enc info_clav_10.csv info_clav_10_ctr.enc clave-demo
    python cifrado_aes_ctr_secuencial.py dec info_clav_10_ctr.enc info_clav_10_ctr_dec.csv clave-demo
"""

from __future__ import annotations

import hashlib
import sys
import time

from Crypto.Cipher import AES

DEFAULT_NONCE = b"CTRnonce"   # 8 bytes
DEFAULT_IO_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def derive_key_from_passphrase(passphrase: str, key_size: int = 32) -> bytes:
    if key_size not in (16, 24, 32):
        raise ValueError("key_size debe ser 16, 24 o 32")
    digest = hashlib.sha256(passphrase.encode("utf-8")).digest()
    return digest[:key_size]


def transform_file_ctr(input_path: str, output_path: str, passphrase: str, nonce: bytes = DEFAULT_NONCE) -> float:
    key = derive_key_from_passphrase(passphrase, key_size=32)
    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)

    start_time = time.perf_counter()
    with open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
        while True:
            chunk = f_in.read(DEFAULT_IO_CHUNK_SIZE)
            if not chunk:
                break
            f_out.write(cipher.encrypt(chunk))
    end_time = time.perf_counter()
    return end_time - start_time


def main() -> None:
    if len(sys.argv) != 5:
        print(
            "Uso: python cifrado_aes_ctr_secuencial.py <enc|dec> "
            "<archivo_entrada> <archivo_salida> <passphrase>"
        )
        sys.exit(1)

    operation = sys.argv[1].lower()
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    passphrase = sys.argv[4]

    if operation not in {"enc", "dec"}:
        raise ValueError("La operación debe ser enc o dec")

    # En AES-CTR, cifrar y descifrar usan la misma transformación.
    elapsed = transform_file_ctr(input_path, output_path, passphrase)
    print(f"Tiempo de ejecución secuencial AES-CTR ({operation}): {elapsed:.6f} s")


if __name__ == "__main__":
    main()
