"""
validar_resultados.py
Itera sobre todos los .enc generados por benchmark.py y verifica:
  Fase 1 - Ciphertext identico: cada _par_w*.enc debe ser byte-identico
           al _seq.enc de referencia para el mismo archivo base.
  Fase 2 - Descifrado correcto: descifrar _seq.enc debe reproducir el
           CSV original en data/.

Uso:
    python validar_resultados.py
    python validar_resultados.py --passphrase mi-clave
    python validar_resultados.py --enc-dir encrypted --data-dir data --dec-dir decrypted
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from validador import files_are_equal
from cifrado_aes_ctr_paralelo import transform_file_ctr_parallel

ENCRYPTED_DIR = "encrypted"
DATA_DIR = "data"
DECRYPTED_DIR = "decrypted"
DEFAULT_PASSPHRASE = "clave-demo"

SEP = "-" * 64


def discover_bases(enc_dir: str) -> list[str]:
    """Detecta los prefijos base que tienen archivo _seq.enc."""
    bases = []
    for name in os.listdir(enc_dir):
        m = re.match(r"^(.+)_seq\.enc$", name)
        if m:
            bases.append(m.group(1))
    return sorted(bases)


def validate_ciphertext(enc_dir: str, bases: list[str]) -> tuple[int, int]:
    """Compara cada _par_w*.enc contra su _seq.enc de referencia."""
    ok = fail = 0
    print()
    print(SEP)
    print("Fase 1: Ciphertext identico (paralelo == secuencial)")
    print(SEP)
    for base in bases:
        seq_path = os.path.join(enc_dir, f"{base}_seq.enc")
        par_files = sorted(
            f for f in os.listdir(enc_dir)
            if re.match(rf"^{re.escape(base)}_par_w\d+\.enc$", f)
        )
        if not par_files:
            print(f"  [{base}] Sin archivos paralelos, omitido.")
            continue
        for par_name in par_files:
            m = re.search(r"_w(\d+)\.enc$", par_name)
            workers = m.group(1) if m else "?"
            par_path = os.path.join(enc_dir, par_name)
            equal = files_are_equal(seq_path, par_path)
            estado = "OK       " if equal else "DIFERENTE"
            print(f"  {base}  workers={workers:>2}  {estado}")
            if equal:
                ok += 1
            else:
                fail += 1
    return ok, fail


def validate_decrypt(
    enc_dir: str,
    data_dir: str,
    dec_dir: str,
    bases: list[str],
    passphrase: str,
) -> tuple[int, int]:
    """Descifra cada .enc y compara el resultado con el CSV original."""
    os.makedirs(dec_dir, exist_ok=True)
    ok = fail = 0
    print()
    print(SEP)
    print("Fase 2: Descifrado correcto (dec(*.enc) == CSV original)")
    print(SEP)
    for base in bases:
        original_csv = os.path.join(data_dir, f"{base}.csv")
        if not os.path.exists(original_csv):
            print(f"  [{base}] CSV original no encontrado en {original_csv}, omitido.")
            continue

        # recolectar todos los .enc de este base: seq + par_w*
        enc_files = sorted(
            f for f in os.listdir(enc_dir)
            if re.match(rf"^{re.escape(base)}(_seq|_par_w\d+)\.enc$", f)
        )
        for enc_name in enc_files:
            enc_path = os.path.join(enc_dir, enc_name)
            dec_name = enc_name.replace(".enc", "_dec.csv")
            dec_csv  = os.path.join(dec_dir, dec_name)
            label    = enc_name.replace(base + "_", "").replace(".enc", "")
            print(f"  {base}  [{label}]  Descifrando... ", end="", flush=True)
            transform_file_ctr_parallel(enc_path, dec_csv, passphrase, n_workers=os.cpu_count() or 4)
            equal = files_are_equal(original_csv, dec_csv)
            estado = "OK" if equal else "DIFERENTE"
            print(estado)
            if equal:
                ok += 1
            else:
                fail += 1
    return ok, fail


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valida ciphertext y descifrado de todos los .enc generados."
    )
    parser.add_argument("--passphrase", default=DEFAULT_PASSPHRASE,
                        help=f"Contrasena usada al cifrar (default: {DEFAULT_PASSPHRASE})")
    parser.add_argument("--enc-dir", default=ENCRYPTED_DIR,
                        help=f"Carpeta con archivos .enc (default: {ENCRYPTED_DIR})")
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help=f"Carpeta con CSVs originales (default: {DATA_DIR})")
    parser.add_argument("--dec-dir", default=DECRYPTED_DIR,
                        help=f"Carpeta destino para descifrados (default: {DECRYPTED_DIR})")
    args = parser.parse_args()

    if not os.path.isdir(args.enc_dir):
        print(f"Error: carpeta '{args.enc_dir}' no encontrada.")
        print("Ejecuta primero 'python benchmark.py' para generar los archivos .enc.")
        sys.exit(1)

    bases = discover_bases(args.enc_dir)
    if not bases:
        print(f"No se encontraron archivos _seq.enc en '{args.enc_dir}'.")
        sys.exit(1)

    print()
    print("================================================================")
    print("         VALIDACION DE RESULTADOS — ACTIVIDAD 1                 ")
    print("================================================================")
    print(f"Archivos base detectados: {', '.join(bases)}")

    ok1, fail1 = validate_ciphertext(args.enc_dir, bases)
    ok2, fail2 = validate_decrypt(
        args.enc_dir, args.data_dir, args.dec_dir, bases, args.passphrase
    )

    total_ok = ok1 + ok2
    total_fail = fail1 + fail2

    print()
    print("================================================================")
    print("  RESUMEN")
    print("================================================================")
    print(f"  Fase 1 (ciphertext identico): {ok1:>3} OK  {fail1:>3} DIFERENTE")
    print(f"  Fase 2 (descifrado correcto): {ok2:>3} OK  {fail2:>3} DIFERENTE")
    print(f"  TOTAL:                        {total_ok:>3} OK  {total_fail:>3} DIFERENTE")
    print("================================================================")
    print()

    if total_fail == 0:
        print("Todos los archivos son correctos.")
    else:
        print(f"ADVERTENCIA: {total_fail} verificacion(es) fallaron.")

    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
