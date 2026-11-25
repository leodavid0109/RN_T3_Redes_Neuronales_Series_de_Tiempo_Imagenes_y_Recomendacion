# Cacería de la alucinación — Módulo 2

Errores cometidos por la herramienta de IA (Claude, Anthropic) durante el trabajo del Módulo 2. Se registran **todos**, también los menores, en el momento en que se detectan. Solo se documentan errores reales con evidencia verificable. Si en algún punto no se encuentra un error, se documentan las verificaciones hechas.

Cada hallazgo sigue los seis campos del enunciado: (1) prompt literal, (2) respuesta de la IA, (3) cómo se sospechó, (4) evidencia, (5) corrección y (6) lección.

| ID | Fecha | Categoría | Resumen | Gravedad |
|---|---|---|---|---|
| H2-1 | 2026-11-25 | Código (fuga de información) | Comparación asimétrica con imágenes volteadas: duplicados repartidos entre particiones | Alta: habría inflado la exactitud en prueba |
| H2-2 | 2026-11-25 | Código (preprocesamiento defectuoso) | El inventario suponía que todos los archivos eran imágenes válidas | Media: detenía el proceso; no producía resultados falsos |
| H2-3 | 2026-11-25 | Datos del mundo real (información técnica desactualizada) | Versión «estable» de PyTorch desactualizada y comando por defecto incompatible con la GPU | Media: ese comando instala una versión que no reconoce la RTX 5060 |
| H2-4 | 2026-11-25 | Datos del mundo real (características del conjunto) | Tamaño del .zip «corregido» por confusión de unidades GB/GiB | Baja |
| H2-5 | 2026-11-25 | Código / Conceptos | Umbral de coseno sobre características no negativas: la mitad de los pares «parecidos» | Alta: habría unido el conjunto en un solo grupo, o guardado 900 MB de pares inútiles |
| H2-6 | 2026-11-25 | Código / Conceptos | Umbral de pHash (Hamming ≤ 6) tomado como estándar, sin medirlo en estos datos | Media: decenas de miles de falsos duplicados, muchos entre clases distintas |
| H2-8 | 2026-11-25 | Código (fuga de información) | Agrupar las fotos de 2024 solo por similitud dejaba a la misma persona en dos particiones | Media: se detectó antes de entrenar |
| H2-7 | 2026-11-25 | Datos del mundo real (características del conjunto) | «Giro es una maniobra legítima, sin alerta», sin verificar qué contiene la clase | Alta: la política de alertas no alertaría el 89 % de esa clase, que es una distracción |

---

## H2-1. Comparación asimétrica con imágenes volteadas (fuga entre particiones)

**1. Prompt.** El código se generó en la conversación después de este mensaje, sin un pedido de código específico (la IA propuso escribir el paso de duplicados y partición a partir de la guía de trabajo): «seria en gpu propia debido a la reproducibilidad, puedo conectarte si quieres con la carpeta del github tambien».

**2. Respuesta de la IA (parte relevante).** Función de similitud en `modulo2/src/m2/duplicados.py`, primera versión:
```python
cos = np.maximum(a @ E[cols].T, a @ F[cols].T)          # A vs B y A vs volteo(B)
hm = np.minimum(ham(P[cols]), ham(PF[cols]))
...
cand &= np.arange(n)[None, :] > filas[:, None]          # cada par una sola vez (i < j)
```
La IA supuso que comparar A con el volteo de B equivale a comparar B con el volteo de A, y por eso evaluaba cada par en una sola dirección.

**3. Sospecha.** No hubo sospecha previa: el error lo detectó la verificación independiente de la partición, que también escribió la IA. En una prueba con 230 imágenes sintéticas y variantes plantadas, la verificación reportó 2 imágenes de validación o prueba con un duplicado en entrenamiento, aunque la agrupación por componentes conexas debía hacerlo imposible.

**4. Evidencia (experimento).** Mismo conjunto sintético, misma semilla:

| Versión | Violaciones validación→entrenamiento | Violaciones prueba→entrenamiento |
|---|---|---|
| Asimétrica (original) | 1 | 1 |
| Simétrica (corregida) | 0 | 0 |

Ni el pHash ni el embedding de una imagen volteada son simétricos: pHash(volteo(A)) no es el volteo bit a bit de pHash(A). Por eso sim(A, volteo(B)) ≠ sim(volteo(A), B), y la agrupación (que solo miraba i < j) y la verificación (que mira todas las combinaciones) obtenían números distintos para el mismo par.

**5. Corrección.** La similitud ahora es el máximo de las tres comparaciones, simétrico por construcción:
```python
cos = np.maximum.reduce([a @ E[cols].T, a @ F[cols].T, af @ E[cols].T])
hm  = np.minimum.reduce([ham(pa, P[cols]), ham(pa, PF[cols]), ham(paf, P[cols])])
```

**6. Lección.** Una partición sin fuga necesita una verificación independiente que recalcule las similitudes entre particiones y detenga el programa si encuentra un caso. Además, antes de evaluar los pares en una sola dirección (i < j) para ahorrar cómputo, hay que comprobar que la medida de similitud sea simétrica.

---

## H2-2. El inventario suponía que todos los archivos eran imágenes válidas

**1. Prompt.** Igual que H2-1 (código generado en la misma respuesta).

**2. Respuesta de la IA (parte relevante).** `modulo2/src/m2/inventario.py`, primera versión, sin ningún manejo de errores:
```python
with Image.open(p) as im:
    ancho_arch, alto_arch = im.size
im = abrir_rgb(p)
```
La IA tomó como cierta la descripción del artículo («7.286 imágenes JPEG»), sin prever archivos dañados.

**3. Sospecha.** Al ejecutarlo sobre los datos reales se detuvo en la imagen 111 de 7.286: `PIL.UnidentifiedImageError: cannot identify image file '...\other_activities\img_13318.jpg'`.

**4. Evidencia.** Con el inventario corregido: 10 de los 7.286 archivos (4 de `turning`, 6 de `other_activities`) tienen tamaño normal (39.889–51.288 bytes), pero su contenido son solo bytes en cero (`data/processed/archivos_ilegibles.csv`). Lista completa en el registro, sección 6.1.

**5. Corrección.** El inventario excluye y documenta los archivos ilegibles en lugar de detenerse, identifica el formato real por los primeros bytes (función `formato_real`) y fuerza la decodificación completa para detectar también archivos truncados. Se probó con un archivo vacío, uno con cabecera HEIC y un JPEG truncado plantados a propósito: los tres se detectan y se clasifican bien.

**6. Lección.** La documentación de un conjunto de datos no garantiza su contenido. El primer paso con datos reales debe ser defensivo: leer cada archivo completo, registrar las fallas y reportarlas, en vez de detenerse o descartarlas en silencio.

---

## H2-3. Versión de PyTorch desactualizada en la consulta web

**1. Prompt.** Del usuario: «dame el comando para que instale pytorch». La IA consultó https://pytorch.org/get-started/locally/ con una herramienta de lectura web y esta instrucción: «What is the current stable PyTorch version, and what are the exact pip install commands for Windows with each CUDA option (index-url)? List the CUDA versions offered.»

**2. Respuesta de la IA (parte relevante).** La herramienta de lectura web devolvió: «Stable version: 2.7.0», opciones de CUDA «11.8, 12.6, 12.8» y como comando por defecto `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`.

**3. Sospecha.** En el entorno de pruebas de la IA ya se había instalado torch 2.15.0 ese mismo día, así que «2.7.0 estable» no podía ser actual. La IA lo advirtió en la misma respuesta al usuario y recomendó revisar el selector y evitar CUDA 11.8 y 12.6.

**4. Evidencia.** La instalación del usuario el 2026-11-25 desde el índice `cu128` descargó `torch-2.11.0+cu128` y `torchvision-0.26.0+cu128` (salida de `pip install`). Además, la RTX 5060 (arquitectura Blackwell) necesita CUDA 12.8 o superior, así que el comando por defecto (`cu118`) habría instalado una versión que no la reconoce.

**5. Corrección.** Se usó el índice `cu128` y se verificó la GPU con `torch.cuda.is_available()` → `True`, `NVIDIA GeForce RTX 5060`. Versiones fijadas en `modulo2/requirements.txt`: torch 2.11.0+cu128 y torchvision 0.26.0+cu128.

**6. Lección.** Las versiones de software no se toman de un resumen de una página web, que puede venir de una copia en caché o de contenido generado con JavaScript. Hay que consultarlas en el propio índice de paquetes (la salida de `pip install` o `pip index versions`) y comprobar la compatibilidad con el hardware ejecutando código.

---

## H2-4. Tamaño del .zip «corregido» por confusión de unidades

**1. Prompt.** El usuario envió una captura del Explorador de Windows con el .zip descargado, sin texto. Antes había escrito: «el dataset pesa 2.8gb».

**2. Respuesta de la IA (parte relevante).** «Al registro le corrijo el tamaño: el .zip pesa 2,38 GB, no los 2,8 GB que anoté antes.» Y en el registro: «2,38 GB según el Explorador de Windows».

**3. Sospecha.** Al revisar el registro, la cifra no coincidía con los 2,6 GB que reporta la copia del conjunto en Zenodo.

**4. Evidencia.** El Explorador de Windows muestra los tamaños en unidades binarias aunque las rotule «GB»: 2,38 × 1024³ bytes = 2,56 × 10⁹ bytes ≈ 2,56 GB decimales, lo que concuerda con los 2,6 GB de Zenodo.

**5. Corrección.** El registro y el `.gitignore` ahora dicen «~2,56 GB» y explican la diferencia de unidades.

**6. Lección.** Antes de «corregir» una cifra con otra fuente, hay que comprobar que ambas usen las mismas unidades. Lo más seguro es comparar en bytes.

---

## H2-5. Umbral de coseno sobre características no negativas

**1. Prompt.** Igual que H2-1 (código y configuración generados en la misma respuesta).

**2. Respuesta de la IA (parte relevante).** En `configs/m2.yaml`: `registro_coseno: 0.85` y `umbral_coseno: 0.94`, con el comentario «Umbrales que DEFINEN un duplicado. Calibrar con figM2-5 antes de fijarlos». El código comparaba los embeddings de ResNet-18 normalizados, sin centrar. La IA fijó umbrales «provisionales» suponiendo que el coseno entre imágenes no relacionadas estaría lejos de 0,85.

**3. Sospecha.** En la primera corrida real, 13.659.949 pares superaron 0,85 (más de la mitad de los ~26,5 millones posibles) y `pares_candidatos.csv` pesó 899 MB. Un conjunto en el que la mitad de los pares son «casi duplicados» no tiene sentido.

**4. Evidencia (experimento con 200.000 pares aleatorios).** Coseno crudo: p1 = 0,704, mediana = 0,851, p99 = 0,933. Mínimo de las características: 0,0, porque salen de un *average pooling* después de una ReLU. Dos vectores con todas sus componentes ≥ 0 forman un ángulo de 90° como máximo, y en la práctica mucho menos, así que su coseno es alto aunque las imágenes no tengan relación. Al restar el vector promedio antes de normalizar (coseno centrado), la mediana entre pares aleatorios baja a 0,020 y el p99,9 a 0,827.

**5. Corrección.** Usar el coseno centrado y elegir el umbral a partir de su distribución en pares aleatorios y de una revisión visual de pares por franja (registro, sección 7). Corrección en el código: [pendiente].

**6. Lección.** Antes de fijar un umbral de similitud, hay que medir la distribución de la similitud entre pares no relacionados (la distribución nula). Los umbrales de coseno no se pueden trasladar de un espacio de características a otro, y las características no negativas, como las de las capas ReLU, deben centrarse.

---

## H2-6. Umbral de pHash tomado como estándar

**1. Prompt.** Igual que H2-1.

**2. Respuesta de la IA (parte relevante).** `umbral_hamming: 6` y `registro_hamming: 12` en `configs/m2.yaml`. En el registro, la tabla de detectores afirmaba que el pHash detecta «recompresión, volteos, cambios leves». Es un umbral habitual en la detección de casi duplicados con pHash de 64 bits, pero no se midió en estos datos.

**3. Sospecha.** Al estudiar H2-5 se cruzaron los pares de pHash con el coseno centrado.

**4. Evidencia.** En 200.000 pares aleatorios, el percentil 0,1 de Hamming ya es 4 y el percentil 1 es 10. Entre todos los pares con Hamming ≤ 6, solo el 42 % de los que tienen Hamming 6 son de la misma clase, frente a un ~21 % esperado por azar, y el 96 % tiene coseno centrado < 0,85. Incluso con Hamming 0 (hash idéntico), el 25 % de los pares tiene coseno centrado < 0,85 (registro, tabla 7.2). Las imágenes de este conjunto (mismo interior del vehículo, mismo encuadre) producen hashes perceptuales muy parecidos aunque la escena cambie.

**5. Corrección.** El pHash deja de ser un criterio para agrupar y queda solo como dato descriptivo.

**6. Lección.** Igual que H2-5: un umbral «estándar» de la literatura o de la práctica habitual se valida con la distribución nula del propio conjunto antes de usarlo.

---

## H2-7. «Giro» tratado como maniobra legítima sin verificar su contenido

**1. Prompt.** «la propuesta de distribucion fue aprobada, me toco ser la persona b entonces quiero indicaciones para que me ayudes a realizar mi parte».

**2. Respuesta de la IA (parte relevante).** En la guía de trabajo (documento del proyecto «Guia Persona B - Modulo 2 y matematica comun», sección 7): «Agrupación sugerida: *sin alerta* = conducción segura, giro (girar es una maniobra legítima); *con alerta* = escribir, hablar, otros riesgosos». En las hipótesis de dificultad de la sección 2 de la guía también se dice: «"giro" vs. "conducción segura" (diferencia sutil en las manos/volante)».

**3. Sospecha.** Al confirmar que el 91 % del conjunto viene de State Farm, se cruzaron las clases de ambos conjuntos.

**4. Evidencia.** Con `driver_imgs_list.csv` de State Farm, 1.187 de las 1.339 imágenes de `turning` (88,6 %) corresponden a la clase c7 *reaching behind* (estirarse hacia el asiento trasero), que es una distracción, no una maniobra de conducción (registro, sección 9). La IA infirió el significado de la clase a partir de su nombre y de la descripción del artículo, sin mirar las imágenes.

**5. Corrección.** La guía y la política de alertas se deben revisar: con el significado real, «giro» debe disparar alerta, o separarse en *reaching behind* frente a giro real si se quiere distinguir las 152 imágenes de otros orígenes. [pendiente: decisión en el registro]

**6. Lección.** El significado de una etiqueta no se deduce de su nombre ni de la descripción de los autores: hay que revisar una muestra de imágenes de cada clase (la Fig. M2-2 sirve para eso) antes de diseñar decisiones de negocio sobre ella.

---

## H2-8. Agrupar las fotos de 2024 solo por similitud

**1. Prompt.** El usuario eligió la opción «Por conductor + todo mezclado (Recomendado)», propuesta por la IA, para la partición. La IA implementó la partición a partir de esa decisión.

**2. Respuesta de la IA (parte relevante).** Primera versión de `particion.py`: las imágenes que no son de State Farm se agrupaban solo por pares con coseno centrado ≥ 0,75. La IA lo justificó así: «Un umbral de 0,75 solo une imágenes de la misma persona o sesión». Esa afirmación se había comprobado solo en State Farm, no en las fotos de 2024.

**3. Sospecha.** La verificación pasó, pero el coseno máximo entre particiones quedó en 0,743, justo debajo del umbral, y los nombres de los archivos de esos pares eran consecutivos (`IMG_3735`, `IMG_3740`) o de la misma tarde (`IMG_20240930_1326…`).

**4. Evidencia.** Revisión visual de los 8 pares entre particiones con mayor coseno (0,730–0,743): en varios aparece la misma persona en particiones distintas (p. ej. `IMG_3735` en entrenamiento y `IMG_3730` en validación; `IMG_3754` en entrenamiento e `IMG_3755` en validación). En otros son personas distintas en el mismo bus (registro, sección 10).

**5. Corrección.** Agrupación adicional por sesión de captura (fecha y hora del nombre del archivo, con cortes de 10 minutos). El coseno máximo entre particiones bajó a 0,734 / 0,626 / 0,566, y ninguna sesión queda repartida.

**6. Lección.** Una conclusión verificada en un subconjunto (aquí, State Farm) no se puede extender a otro de procedencia distinta. Que la verificación automática pase no basta: hay que mirar los casos que quedan justo debajo del umbral.

---

## Hipótesis de la IA descartadas (no son hallazgos, porque se plantearon como hipótesis)

| Fecha | Hipótesis | Resultado |
|---|---|---|
| 2026-11-25 | Las imágenes `img_N.jpg` con números consecutivos son ráfagas o fotogramas de video, y por tanto casi duplicadas | Descartada: con diferencia de 1, el coseno centrado mediano es 0,014, igual que entre imágenes al azar (registro, 7.4) |

## Errores de proceso (no son hallazgos de contenido)

| Fecha | Error | Corrección |
|---|---|---|
| 2026-11-25 | La IA dijo que escribiría el código de los modelos mientras se descargaban los datos y no lo hizo; el usuario no fue informado hasta más tarde. | Se informó explícitamente y se reprogramó para después de preparar los datos. |
