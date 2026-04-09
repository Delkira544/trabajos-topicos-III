import sys
import os

def files_are_equal(file1, file2):
    if not os.path.exists(file1) or not os.path.exists(file2):
        print("Error: uno o ambos archivos no existen.")
        return False

    if os.path.getsize(file1) != os.path.getsize(file2):
        return False

    chunk_size = 4096
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        while True:
            b1 = f1.read(chunk_size)
            b2 = f2.read(chunk_size)
            if b1 != b2:
                return False
            if not b1:
                return True

def main():
    if len(sys.argv) != 3:
        print("Uso: python validador.py <archivo1> <archivo2>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    if files_are_equal(file1, file2):
        print("Los archivos son iguales.")
    else:
        print("Los archivos son diferentes.")

if __name__ == "__main__":
    main()
