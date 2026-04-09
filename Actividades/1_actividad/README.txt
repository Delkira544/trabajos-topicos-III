=====================================

Archivos incluidos
------------------
1. generate_files.py
   Genera archivos CSV grandes para pruebas.

2. validador.py
   Compara dos archivos en binario.

3. cifrado_aes_ctr_secuencial.py
   Línea base secuencial de AES-CTR.

4. requirements.txt
   Dependencias sugeridas.

Flujo sugerido (IMPORTANTE REVISAR RUBRICA POR SI FALTA ALGUN PUNTO)
--------------
1. Generar o recibir un archivo de entrada.
2. Ejecutar la versión secuencial para verificar cifrado y descifrado.
3. Crear una versión paralela propia basada en chunks.
4. Medir tiempos con distintos workers.
5. Comparar con la versión secuencial.
6. Validar que el archivo descifrado sea igual al original.

Ejemplos (Cambiar tu-clave)
--------
python generate_files.py 10 info_clav_10.csv
python cifrado_aes_ctr_secuencial.py enc info_clav_10.csv info_clav_10_ctr.enc tu-clave
python cifrado_aes_ctr_secuencial.py dec info_clav_10_ctr.enc info_clav_10_ctr_dec.csv tu-clave
python validador.py info_clav_10.csv info_clav_10_ctr_dec.csv


python generate_files.py 1024 info_clav_1024.csv
python cifrado_aes_ctr_secuencial.py enc info_clav_1024.csv info_clav_1024_ctr.enc tu-clave
python cifrado_aes_ctr_secuencial.py dec info_clav_1024_ctr.enc info_clav_1024_ctr_dec.csv tu-clave
python validador.py info_clav_1024.csv info_clav_1024_ctr_dec.csv

Nota
----
En AES-CTR, el cifrado y el descifrado usan la misma transformación.

Para crear el entorno

conda create -n INFO1195 python=3.11 -y
conda activate INFO1195
pip install -r requirements.txt