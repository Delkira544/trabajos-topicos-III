import csv
import io
import os
import random
import sys

DIR_DATA = "data"

frutas_verduras = [
    "Manzana",
    "Pera",
    "Banana",
    "Naranja",
    "Fresa",
    "Mango",
    "Uva",
    "Melón",
    "Sandía",
    "Kiwi",
    "Piña",
    "Cereza",
    "Durazno",
    "Lima",
    "Limón",
    "Papaya",
]

animales = [
    "Perro",
    "Gato",
    "Elefante",
    "León",
    "Tigre",
    "Oso",
    "Lobo",
    "Jirafa",
    "Cebra",
    "Mono",
    "Rinoceronte",
    "Hipopótamo",
    "Vaca",
    "Cerdo",
    "Caballo",
    "Gallina",
    "Pato",
    "Conejo",
    "Oveja",
    "Burro",
    "Zorro",
    "Camello",
    "Iguana",
]


def generar_fila():
    fruta = random.choice(frutas_verduras)
    animal = random.choice(animales)
    latitud = round(random.uniform(-90, 90), 6)
    longitud = round(random.uniform(-180, 180), 6)
    numero = random.randint(1, 100)
    return [fruta, animal, latitud, longitud, numero]


def main():
    if len(sys.argv) not in (2, 3):
        print("Uso: python generate_files.py <tamano_en_MB> [nombre_archivo.csv]")
        sys.exit(1)

    try:
        size_mb = float(sys.argv[1])
    except ValueError:
        print("El tamaño debe ser un número en MB.")
        sys.exit(1)

    # Si se pasa nombre de archivo, se usa tal cual;
    # si no, se genera el nombre por convención dentro de data/
    if len(sys.argv) == 3:
        filename = sys.argv[2]
    else:
        size_int = int(size_mb)
        filename = os.path.join(DIR_DATA, f"info_clav_{size_int}.csv")

    # Crear la carpeta data/ si el archivo va a guardarse ahí
    parent = os.path.dirname(filename)
    if parent:
        os.makedirs(parent, exist_ok=True)

    target_size = int(size_mb * 1024 * 1024)

    with open(filename, mode="wb") as bin_file:
        text_file = io.TextIOWrapper(bin_file, encoding="utf-8", newline="")
        writer = csv.writer(text_file)
        writer.writerow(["Fruta_Verdura", "Animal", "Latitud", "Longitud", "Numero"])

        while True:
            writer.writerow(generar_fila())
            text_file.flush()
            if bin_file.tell() >= target_size:
                break

    print(f"Archivo '{filename}' generado con un tamaño aproximado de {size_mb} MB.")


if __name__ == "__main__":
    main()
