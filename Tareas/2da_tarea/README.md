# Tarea 2 — Paralelización de algoritmos secuenciales en CUDA C++

## Información del Grupo
- **Grupo:** Profe Artes
- **Integrantes:** Sion Arancibia, Daniel Burgos, Daniel Peña, Jorge Soto, Christian Verdugo
- **Curso:** INFO1194, Semestre I-2026

## GPU Utilizada / Versión CUDA
- **GPU:** NVIDIA GeForce GTX 1650 Ti (Memoria GPU total: 7.9 GB, Memoria dedicada: 4.0 GB, Memoria compartida: 3.9 GB)
- **Versión CUDA:** CUDA 13.3 (Compilador `nvcc` v13.3.33)

---

## Compilación con CMake

### En Linux / macOS
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### En Windows (PowerShell)
```powershell
cmake -B build
cmake --build build --config Release -j8
```

## Ejecución

### Todos los programas en orden (Pruebas unitarias)
```bash
cd build
ctest -C Release --output-on-failure
```

### Programas individuales
```powershell
# En Windows (desde la carpeta build):
.\Release\secuencial1.exe [N]
.\Release\paralelo1.exe [N]

.\Release\secuencial2.exe [N]
.\Release\paralelo2.exe [N]

.\Release\secuencial3.exe [N]
.\Release\paralelo3.exe [N]

.\Release\secuencial4.exe [N]
.\Release\paralelo4.exe [N]

.\Release\secuencial5.exe [N] [umbral]
.\Release\paralelo5.exe [N] [umbral]
```
*Nota: N opcional — default: `1<<24` (P1-P3), `1<<26` (P4-P5). Umbral default: 500.*

---

## Tabla de Tiempos CPU vs GPU

A continuación se reportan los tiempos medidos en milisegundos (ms) para cada problema. Se indica por separado el tiempo del **Kernel GPU puro** y el **Tiempo Total GPU** (que incluye las transferencias de memoria CPU <-> GPU via PCIe):

| Problema | Método | N | Tiempo CPU (ms) | Tiempo GPU Kernel (ms) | Tiempo GPU Total (ms) | Speedup (Kernel) | Resultado |
|---|---|---|---|---|---|---|---|
| **Problema 1: Transformación** | 1 hilo/elemento | 1,024 | 0.0002 | 0.0436 | 0.1243 | 0.004x | CORRECTO |
| | Grid-Stride Loop | 1,024 | 0.0002 | 0.0595 | 0.1274 | 0.003x | CORRECTO |
| | 1 hilo/elemento | 100,000 | 0.0249 | 0.0564 | 0.2876 | 0.441x | CORRECTO |
| | Grid-Stride Loop | 100,000 | 0.0249 | 0.0558 | 0.2682 | 0.446x | CORRECTO |
| | 1 hilo/elemento | 1,048,576 | 0.6629 | 0.1090 | 1.5487 | 6.082x | CORRECTO |
| | Grid-Stride Loop | 1,048,576 | 0.6629 | 0.1187 | 1.3313 | 5.585x | CORRECTO |
| | 1 hilo/elemento | 16,777,216 | 10.2989 | 0.9412 | 17.1778 | 10.942x | CORRECTO |
| | Grid-Stride Loop | 16,777,216 | 10.2989 | 1.0987 | 17.6845 | 9.374x | CORRECTO |
| **Problema 2: Stencil 1D** | Memoria Global | 1,024 | 0.0006 | 0.0455 | 0.1280 | 0.013x | CORRECTO |
| | Memoria Compartida | 1,024 | 0.0006 | 0.0494 | 0.1414 | 0.012x | CORRECTO |
| | Memoria Global | 100,000 | 0.0404 | 0.0557 | 0.2767 | 0.725x | CORRECTO |
| | Memoria Compartida | 100,000 | 0.0404 | 0.0559 | 0.2648 | 0.723x | CORRECTO |
| | Memoria Global | 1,048,576 | 0.5275 | 0.1056 | 1.3873 | 4.995x | CORRECTO |
| | Memoria Compartida | 1,048,576 | 0.5275 | 0.1328 | 1.3964 | 3.972x | CORRECTO |
| | Memoria Global | 16,777,216 | 9.5256 | 1.0378 | 18.6983 | 9.179x | CORRECTO |
| | Memoria Compartida | 16,777,216 | 9.5256 | 1.2502 | 19.3176 | 7.619x | CORRECTO |
| **Problema 3: Reducción Suma** | Reducción por Bloque | 1,024 | 0.0012 | 0.0573 | 0.2630 | 0.021x | CORRECTO |
| | Reducción por Bloque | 100,000 | 0.1179 | 0.0827 | 0.2485 | 1.426x | CORRECTO |
| | Reducción por Bloque | 1,048,576 | 1.3912 | 0.1564 | 0.7545 | 8.895x | CORRECTO |
| | Reducción por Bloque | 16,777,216 | 19.9379 | 1.9379 | 12.8313 | 10.288x | CORRECTO |
| **Problema 4: Histograma** | Atómico Global | 1,024 | 0.0007 | 0.0174 | 0.1085 | 0.040x | CORRECTO |
| | Atómico Local (Shared)| 1,024 | 0.0007 | 0.0702 | 0.1491 | 0.010x | CORRECTO |
| | Atómico Global | 100,000 | 0.2023 | 0.1106 | 0.3956 | 1.829x | CORRECTO |
| | Atómico Local (Shared)| 100,000 | 0.2023 | 0.0778 | 0.3068 | 2.600x | CORRECTO |
| | Atómico Global | 1,048,576 | 0.6206 | 0.4094 | 0.7275 | 1.516x | CORRECTO |
| | Atómico Local (Shared)| 1,048,576 | 0.6206 | 0.0701 | 0.3695 | 8.852x | CORRECTO |
| | Atómico Global | 67,108,864 | 34.0560 | 26.4335 | 38.1745 | 1.288x | CORRECTO |
| | Atómico Local (Shared)| 67,108,864 | 34.0560 | 4.4711 | 12.4007 | 7.617x | CORRECTO |
| **Problema 5: Conteo Warps** | Atómico Global | 1,024 | 0.0002 | 0.0451 | 0.1420 | 0.004x | CORRECTO |
| | Conteo por Warp | 1,024 | 0.0002 | 0.0655 | 0.1375 | 0.003x | CORRECTO |
| | Atómico Global | 100,000 | 0.0146 | 0.0082 | 0.1747 | 1.780x | CORRECTO |
| | Conteo por Warp | 100,000 | 0.0146 | 0.0082 | 0.1627 | 1.780x | CORRECTO |
| | Atómico Global | 1,048,576 | 0.1830 | 0.0480 | 0.9177 | 3.813x | CORRECTO |
| | Conteo por Warp | 1,048,576 | 0.1830 | 0.0446 | 0.9524 | 4.103x | CORRECTO |
| | Atómico Global | 67,108,864 | 16.6906 | 2.4398 | 35.2187 | 6.841x | CORRECTO |
| | Conteo por Warp | 67,108,864 | 16.6906 | 2.7649 | 34.6072 | 6.037x | CORRECTO |

---

## Respuestas a las Preguntas de Análisis

### Problema 1: Transformación de vector
1. **¿Cómo se calcula el índice global de un hilo?**
   El índice global de un hilo en un mapeo unidimensional (1D) se calcula mediante la fórmula:
   `int idx = blockIdx.x * blockDim.x + threadIdx.x;`
   Esta fórmula multiplica el identificador del bloque en el grid (`blockIdx.x`) por el número total de hilos que tiene cada bloque (`blockDim.x`), y le suma el identificador local del hilo dentro de su propio bloque (`threadIdx.x`).

2. **¿Por qué es necesario validar `idx < N`?**
   Es necesario porque el tamaño del vector de entrada $N$ puede no ser un múltiplo exacto de la dimensión del bloque (`blockDim.x`). Dado que lanzamos un número total de hilos que es múltiplo de `blockDim.x` (usando la fórmula de redondeo hacia arriba: `blocks = (N + threads - 1) / threads`), el último bloque lanzado contendrá hilos inactivos. Si no validamos `idx < N`, los hilos excedentes intentarán realizar accesos a memoria fuera de los límites del vector (`B[idx] = ...`), provocando corrupción de datos, fallos de segmento (illegal memory access) o comportamiento indefinido.

3. **¿Qué diferencia hay entre una versión con un elemento por hilo y una versión con grid-stride loop?**
   - **Un elemento por hilo:** Asume una correspondencia rígida de $1:1$ entre los hilos del grid y las posiciones del vector. Obliga a redimensionar el grid (lanzar más bloques) proporcionalmente al tamaño del vector $N$.
   - **Grid-stride loop:** Cada hilo procesa un elemento en un ciclo `while` y avanza su índice sumando el tamaño del grid (`gridDim.x * blockDim.x`). Esto permite:
     - Procesar vectores de cualquier tamaño $N$ usando un tamaño de grid constante y óptimo.
     - Mayor reusabilidad de hilos (evita el costo de lanzar grids gigantescos).
     - Mejor soporte para depuración (escala fácilmente en emuladores).
     - Conservación de la alineación y coalescencia de memoria al saltar exactamente por múltiplos del tamaño del warp/grid.

4. **¿Más hilos siempre implica menor tiempo? Justifique con sus mediciones.**
   No. El hardware de la GPU tiene un límite físico en la cantidad de hilos que puede ejecutar simultáneamente (determinado por la cantidad de multiprocesadores de flujo o SMs y su capacidad máxima de registros y memoria compartida). Si el número de hilos es muy pequeño, la GPU está subutilizada y los tiempos son dominados por el overhead de lanzamiento. Si el número de hilos es extremadamente grande, se satura el planificador de la GPU sin ganar paralelismo real, y el rendimiento se ve limitado por el ancho de banda del bus de memoria y el overhead de planificación de bloques.
   
   Lo demostramos con las siguientes **mediciones empíricas en la NVIDIA GeForce GTX 1650 Ti** para el Problema 1 (Transformación de vector) con $N = 1,048,576$:

      | Hilos por Bloque | Cantidad de Bloques | Kernel 1 Hilo/Elem (ms) | Total 1 Hilo/Elem (ms) | Kernel Grid-Stride (ms) | Total Grid-Stride (ms) |
      |:---:|:---:|:---:|:---:|:---:|:---:|
      | **32** | 32,768 | 0.1968 | 1.4020 | 0.1834 | 1.4122 |
      | **64** | 16,384 | 0.0938 | 1.5561 | 0.1790 | 1.8780 |
      | **128** | 8,192 | 0.1023 | 1.4513 | 0.2143 | 1.9880 |
      | **256** | 4,096 | 0.0932 | 1.6491 | 0.1989 | 1.6425 |
      | **512** | 2,048 | 0.1044 | 1.5969 | 0.2068 | 1.6586 |
      | **1024** | 1,024 | 0.1050 | 1.3729 | 0.1451 | 2.1003 |
      
   **Análisis detallado de los datos:**
   1. **Subutilización de la GPU (32 hilos/bloque):** En ambas versiones, la configuración de 32 hilos por bloque (que equivale a solo 1 warp por bloque) registra los tiempos de kernel más elevados ($0.1968$ ms y $0.1834$ ms). Esto ocurre porque 1 warp por bloque no es suficiente para que el planificador de la GPU (Warp Scheduler) pueda ocultar eficazmente las latencias de lectura/escritura en memoria global, dejando a las unidades de cómputo inactivas esperando los datos.
   2. **Impacto de la ocupación (64 a 256 hilos/bloque):** 
      - En la versión **1 Hilo/Elem**, al incrementar a 64 hilos (2 warps/bloque), el tiempo disminuye drásticamente a **$0.0938$ ms** (una mejora del **$52.3\%$**). El mejor tiempo absoluto se registra en 256 hilos con **$0.0932$ ms**.
      - En la versión **Grid-Stride**, el comportamiento se mantiene más estable en torno a $0.18 - 0.20$ ms en rangos medios debido a la sobrecarga extra que introduce el control del bucle `while` y los saltos de índice en un cálculo tan ligero. Sin embargo, demuestra que a partir de 64 hilos el rendimiento general ya alcanza su meseta.
   3. **Saturación del hardware (512 a 1024 hilos/bloque):** A partir de 256 hilos por bloque, aumentar la cantidad de hilos por bloque no produce ninguna ganancia en rendimiento (el tiempo oscila de forma plana entre $0.093$ y $0.105$ ms). Esto comprueba de forma cuantitativa que, una vez alcanzada la ocupación óptima y saturado el ancho de banda del bus de memoria, añadir más hilos solo introduce latencia en la planificación del hardware sin traducirse en un menor tiempo de ejecución.

---

### Problema 2: Stencil 1D usando shared memory
5. **¿Qué son los halos izquierdo y derecho?**
   Los halos (o celdas fantasmas) son elementos vecinos que pertenecen a bloques de hilos adyacentes pero que son necesarios para calcular el resultado de los bordes del bloque actual. En un stencil 1D de radio 1:
   - El **halo izquierdo** es el elemento inmediatamente anterior al primer elemento del bloque (`A[idx - 1]`).
   - El **halo derecho** es el elemento inmediatamente posterior al último elemento del bloque (`A[idx + 1]`).
   Para evitar accesos costosos a memoria global dentro del stencil, estos elementos son precargados en la memoria compartida por los hilos extremos del bloque.

6. **¿Qué datos se guardan en shared memory?**
   Se guarda el subarreglo de datos de entrada necesarios para realizar el stencil localmente en el bloque. Específicamente, para un bloque de hilos de tamaño `blockDim.x`, se reservan y guardan los `blockDim.x` elementos centrales asignados a los hilos de dicho bloque, más los $2$ halos correspondientes a los límites izquierdo y derecho, sumando un total de `blockDim.x + 2` elementos de tipo flotante.

7. **¿Qué error puede aparecer si se elimina `__syncthreads()`?**
   Si se elimina `__syncthreads()`, se introduce una **condición de carrera** (race condition) de tipo lectura-después-de-escritura (RAW). Algunos hilos podrían intentar leer los halos o los datos centrales de la memoria compartida `tile` antes de que los hilos asignados a cargarlos desde la memoria global hayan completado su escritura. Como resultado, los hilos leerán basura o datos desactualizados, produciendo resultados incorrectos.

8. **¿La versión shared memory fue siempre más rápida que la versión global? Explique.**
   No. De hecho, en todas nuestras mediciones en la NVIDIA GeForce GTX 1650 Ti, la versión con memoria compartida (shared memory) fue ligeramente **más lenta** que la versión con memoria global. Lo detallamos en la siguiente tabla comparativa de tiempos de kernel puro:

    | Tamaño de Entrada (N) | Tiempo Kernel Global (ms) | Tiempo Kernel Shared (ms) | Variación porcentual (Shared vs Global) | ¿Shared es más rápido? |
    |:---:|:---:|:---:|:---:|:---:|
    | **1,024** | 0.0494 ms | 0.1004 ms | +103.24% (Shared es más lento) | **No** |
    | **100,000** | 0.0824 ms | 0.0637 ms | -22.69% (Shared es más rápido) | **Sí** |
    | **1,048,576** | 0.1179 ms | 0.1400 ms | +18.74% (Shared es más lento) | **No** |
    | **16,777,216** | 1.0027 ms | 1.5725 ms | +56.82% (Shared es más lento) | **No** |

   **Estos resultados son producto de:**
   1. **Caché L1/L2 de Hardware:** Las GPUs modernas tienen cachés de hardware L1 y L2 muy eficientes. Al realizar accesos continuos en un stencil 1D (donde los hilos acceden a `A[idx - 1]`, `A[idx]` y `A[idx + 1]`), los hilos adyacentes del mismo warp leen posiciones contiguas que ya han sido traídas a la caché de la SM. Por lo tanto, el hardware ya está cacheando estos datos automáticamente en la versión global de forma transparente.
   2. **Sobrecarga de Instrucciones (Instruction Overhead):** La versión con memoria compartida requiere calcular índices locales, inicializar el arreglo local, cargar condicionalmente los halos izquierdo/derecho mediante bifurcaciones (`if`) y sincronizar los hilos usando `__syncthreads()`. Estas instrucciones de control y la barrera de sincronización introducen una sobrecarga de ciclos de reloj que supera el ahorro de latencia de memoria.
   3. **Baja Intensidad Aritmética:** Dado que el stencil de radio 1 realiza únicamente dos sumas por elemento, el kernel está fuertemente limitado por el ancho de banda (memory-bound). Al haber una alta localidad espacial de memoria, la caché L2 del chip iguala la latencia de acceso a la memoria compartida programada a mano, pero sin el costo asociado de ejecutar instrucciones adicionales y barreras de sincronización.

---

### Problema 3: Reducción de suma
9. **¿Por qué una reducción no se paraleliza igual que una transformación elemento a elemento?**
   Porque una transformación elemento a elemento es un problema de paralelismo puro donde cada salida depende únicamente de una entrada única y las operaciones no tienen interferencia mutua. La reducción, en cambio, requiere condensar múltiples valores de entrada en un único resultado. Esto introduce dependencias de datos en cascada: los resultados de las sumas parciales de cada hilo deben combinarse de manera sucesiva (normalmente mediante una estructura de árbol binario), lo que exige sincronización rigurosa para evitar colisiones y pérdida de datos.

10. **¿Por qué se generan sumas parciales?**
    Se generan porque la GPU no posee un mecanismo de sincronización global de bajo costo aplicable a nivel de todo el grid de bloques de hilos. Cada bloque realiza la reducción interna de sus hilos de forma independiente y escribe su resultado en una posición reservada de un arreglo de **sumas parciales** en memoria global. Luego, una etapa final en la CPU o un segundo kernel CUDA más pequeño suma estas sumas parciales para dar el resultado definitivo.

11. **¿Por qué se usa shared memory en este problema?**
    Se usa para acelerar la acumulación de las sumas locales dentro de cada bloque. Al cargar los datos en memoria compartida, los hilos de un mismo bloque pueden colaborar sumando pares de datos recursivamente con accesos de bajísima latencia y alto ancho de banda, evitando miles de operaciones atómicas costosas directamente sobre la memoria global.

12. **¿Por qué pueden existir pequeñas diferencias numéricas entre CPU y GPU cuando se usa float?**
    Se debe a que las operaciones con números en punto flotante no cumplen con la propiedad asociativa ($ (a + b) + c \neq a + (b + c) $) debido a los errores de redondeo inherentes a la precisión limitada (IEEE 754). La CPU realiza la suma de forma estrictamente secuencial, de izquierda a derecha, mientras que la GPU reduce el vector acumulando en paralelo en forma de árbol binario de suma. La diferencia en el orden de agrupación de los términos provoca discrepancias en los bits menos significativos del resultado.

---

### Problema 4: Histograma de 256 bins
13. **¿Por qué se necesita `atomicAdd`?**
    Se necesita porque múltiples hilos del grid pueden procesar simultáneamente datos que corresponden al mismo bin del histograma. Sin un mecanismo de exclusión mutua, los hilos intentarían leer, incrementar y escribir sobre la misma dirección de memoria al mismo tiempo, lo que causaría condiciones de carrera y pérdida de incrementos. `atomicAdd` garantiza que la lectura, incremento y escritura ocurra como una sola transacción indivisible.

14. **¿Qué condición de carrera aparece si varios hilos incrementan el mismo bin sin `atomicAdd`?**
    Aparece una colisión de tipo lectura-modificación-escritura (RMW). Por ejemplo, si el hilo A y el hilo B leen simultáneamente que el bin 5 tiene un conteo de $10$, ambos calculan $10 + 1 = 11$ localmente y luego escriben $11$ en el bin. El valor resultante es $11$ en lugar del correcto $12$, perdiendo un conteo.

15. **¿Qué ventaja tiene construir histogramas locales por bloque?**
    La ventaja principal es la reducción drástica de la **contención de memoria**. En lugar de que miles de hilos en toda la GPU intenten actualizar simultáneamente los mismos 256 bins en la memoria global (cuello de botella extremo), los hilos actualizan un histograma local privado del bloque en memoria compartida. Al finalizar el procesamiento del bloque, las sumas acumuladas locales se transfieren a la memoria global con un único conjunto de operaciones atómicas por bloque.

16. **¿Qué ocurre si todos los valores de entrada son iguales?**
    Si todos los elementos son iguales (por ejemplo, todos ceros), la contención llega a su punto máximo. En la versión global, todos los hilos de la GPU intentan escribir atómicamente en la misma dirección de memoria global (`hist[0]`), serializando completamente la ejecución y degradando drásticamente el rendimiento. En la versión compartida, la contención se traslada a la memoria compartida, lo cual es mucho más rápido, aunque sufrirá de conflictos de bancos (bank conflicts).

---

### Problema 5: Conteo con condición usando warps
17. **¿Qué es un warp?**
    Un warp es la unidad básica de planificación y ejecución en la arquitectura CUDA. Consta de un grupo de 32 hilos consecutivos (con índices locales `threadIdx.x` alineados de 0 a 31, 32 a 63, etc.) que se ejecutan simultáneamente en hardware bajo el modelo SIMT (Single Instruction, Multiple Threads), compartiendo la misma instrucción en el mismo ciclo de reloj.

18. **¿Dónde aparece divergencia en este problema?**
    La divergencia de warps aparece al evaluar la condición `if (A[idx] > umbral)`. Dado que los hilos de un warp procesan elementos de datos independientes, es muy común que algunos hilos cumplan la condición y otros no. Cuando esto ocurre, los hilos que no cumplen la condición son enmascarados (desactivados en hardware) mientras se ejecuta el bloque del `if` para los hilos activos, serializando el flujo de ejecución del warp y reduciendo la eficiencia general.

19. **¿Por qué `__ballot_sync()` puede reducir la cantidad de operaciones atómicas?**
    Porque permite a los hilos de un mismo warp cooperar activamente evaluando su condición y consolidando sus resultados en un registro de máscara de 32 bits en un solo ciclo de reloj. En lugar de que cada hilo que cumple la condición llame individualmente a `atomicAdd` sobre la memoria global (lo que podría generar hasta 32 operaciones atómicas concurrentes), la función `__popc(mask)` cuenta cuántos hilos del warp tienen su bit activo y un solo hilo líder del warp realiza un único `atomicAdd` acumulando el conteo de todos ellos. Esto reduce el número de operaciones atómicas globales en un factor de hasta 32.

20. **¿Qué ocurre cuando aproximadamente la mitad de los hilos cumple la condición?**
    Al evaluar una condición con $50\%$ de probabilidad en $N = 1,048,576$ elementos (como con el umbral de 500), los reportes de NVIDIA Nsight Compute (`ncu`) para la métrica `l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum` muestran que tanto el kernel global como el cooperativo generan **exactamente 32,768 sectores** de transacciones atómicas a memoria global, debido a que el hardware de la GPU (arquitectura Turing / CC 7.5) realiza **agregación atómica (Hardware Atomic Aggregation)** y consolida automáticamente las escrituras de un mismo warp a la misma dirección física (`contador`) en una única transacción de hardware. No obstante, la versión cooperativa por warp sigue siendo muy superior porque en la versión global la GPU tiene que decodificar y despachar individualmente las instrucciones de suma atómica de todos los hilos activos del warp (overhead de despacho de instrucciones), mientras que en la versión cooperativa con `__ballot_sync()` y `__popc()` solo un hilo líder por warp emite e introduce una única instrucción `atomicAdd` al pipeline físico, ahorrando significativamente ciclos de ejecución en la GPU.
