# Registro de trabajo — Módulo 2: clasificación de conducción distractiva

**Responsable:** Persona B · **Repositorio:** carpeta `modulo2/` · **Estado:** en curso

Este registro documenta, en orden cronológico, las decisiones tomadas en el Módulo 2, su justificación, las verificaciones realizadas y sus resultados. Cada cifra indica de dónde sale: del artículo de los autores, de nuestro código (con el archivo que la produce) o de una suposición explícita. Las entradas marcadas **[pendiente]** se completan cuando se ejecuta el paso correspondiente.

**Política de errores de la IA.** Todo error cometido por la herramienta de IA durante el trabajo (código, datos, cifras, conceptos o referencias) se documenta, aunque sea menor, en `docs/alucinaciones/modulo2.md`, con la plantilla de seis campos del enunciado. En este registro se enlaza el hallazgo (H2-n) desde la sección donde ocurrió.

## Resumen de estado

| Etapa | Estado | Evidencia |
|---|---|---|
| Entorno y hardware | Listo: PyTorch reconoce la GPU | Sección 1 |
| Descripción del conjunto según los autores | Hecho | Sección 2 |
| Código de inventario, duplicados y partición | Escrito y probado con datos sintéticos | `modulo2/src/m2/`, sección 4 |
| Inventario sobre los datos reales | Hecho: 7.276 imágenes legibles, 10 archivos dañados; formatos y resoluciones distintos a los del artículo | Sección 6, `data/processed/resumen_inventario.json` |
| Origen de los 10 archivos dañados (¿.zip o descompresión?) | Hecho: ya venían en ceros en el .zip publicado | Sección 6.5 |
| Detección de duplicados (primera corrida) | Hecha; el umbral de coseno del diseño no servía (H2-5) y el pHash no es confiable | Sección 7 |
| Procedencia de las imágenes | **Confirmado:** 6.629 imágenes (91,1 %) son de State Farm (26 conductores); «giro» es en su mayoría *reaching behind* | Secciones 8 y 9 |
| Calibración del umbral y partición | **Hecha y confirmada** (por conductor + sesión + coseno centrado ≥ 0,75; 5.076 / 1.067 / 1.133; 0 fugas) | Sección 10, `splits.csv`, `verificacion_fuga.json` |
| Preprocesamiento | Hecho: 224 px, estirado, 121,4 MB | Secciones 11 y 12 |
| Modelo de referencia (logística sobre características congeladas) | Hecho: F1 macro en prueba 0,698 (State Farm 0,747; fotos de 2024 0,237) | Sección 12 |
| CNN desde cero | Hecho: F1 prueba 0,656 ± 0,102 (30 semillas) | Sección 15 |
| Transfer learning | **Hecho: F1 prueba 0,855 ± 0,021 (30 semillas); mejor modelo** | Sección 15 |
| Grad-CAM, aciertos y errores, política de alertas | [pendiente] | |

---

## 2026-11-25

### 1. Entorno de cómputo

**Decisión:** entrenar en GPU propia y no en servicios en la nube (Colab o Kaggle).
**Justificación:** reproducibilidad. En la nube cambian sin aviso la GPU asignada, las versiones de CUDA y de las librerías y el tiempo disponible, y eso afecta tanto a los resultados como a los tiempos de cómputo que el enunciado pide reportar.

| Elemento | Valor | Fuente |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5060, 8 GB (8151 MiB) | `nvidia-smi`, 2026-11-25 |
| Controlador | 610.88; CUDA UMD 13.3 | `nvidia-smi` |
| Sistema operativo | Windows | |
| Python | 3.14 (64 bits) | salida de `pip install` |
| PyTorch, torchvision | 2.11.0+cu128, 0.26.0+cu128 (índice `https://download.pytorch.org/whl/cu128`) | salida de `pip install`, 2026-11-25 |
| Reconocimiento de la GPU por PyTorch | Sí: `2.11.0+cu128 12.8 True NVIDIA GeForce RTX 5060` | `torch.cuda.is_available()`, 2026-11-25 |
| Lanzador de Python | `py` (el alias `python` de Windows abre la Microsoft Store; todos los comandos de este módulo se ejecutan con `py`) | |

**Consecuencias:**
- La RTX 5060 tiene arquitectura Blackwell y requiere una compilación de PyTorch con CUDA 12.8 o superior; las compilaciones con CUDA más antiguas no la reconocen. La versión exacta instalada se fija en `modulo2/requirements.txt`.
- 8 GB de memoria limitan el tamaño de lote en el ajuste fino. Se registrará el tamaño de lote usado y si se usa precisión mixta.
- Determinismo: `set_seed` (`modulo2/src/m2/utils.py`) fija las semillas de Python, NumPy y PyTorch (CPU y GPU), desactiva `cudnn.benchmark`, activa `cudnn.deterministic` y `torch.use_deterministic_algorithms` y define `CUBLAS_WORKSPACE_CONFIG`. Semilla global: 42; corridas múltiples: semillas 0 a 29.
- Los datos crudos (archivo `Multi-Class Driver Behavior Image Dataset.zip`; el Explorador de Windows muestra «2,38 GB», que en realidad son GiB, es decir unos 2,56 GB decimales, lo que concuerda con los 2,6 GB que reporta la copia en Zenodo) no se versionan en Git: superan con creces los límites de GitHub y el enunciado pide versionar solo los datos procesados. Se guardan fuera de la carpeta sincronizada con OneDrive, para evitar conflictos de sincronización con Git y con la lectura de miles de archivos. La ruta se indica en `configs/m2.yaml` (`datos.raw_dir`).

### 2. El conjunto de datos según sus autores

Fuentes: registro en Mendeley Data (DOI 10.17632/mzb4b6dff3.1, versión 1, publicado el 15-11-2024) y artículo en Data in Brief (DOI 10.1016/j.dib.2025.111529).

| Característica | Según el artículo |
|---|---|
| Total de imágenes | 7.286 |
| Por clase | Conducción segura 1.679 · Giro 1.343 · Hablar por teléfono 1.513 · Escribir en el teléfono 1.561 · Otros 1.190 |
| Resolución | 400 × 600 píxeles; formato JPEG |
| Captura | Dos celulares (iPhone 11 y Moto Edge 50 Fusion), en vehículos particulares y buses públicos |
| Conductores | 178, de 20 a 50 años; unos 30 días de captura, con 1–2 horas diarias por conductor |
| Lugar y fecha | Octubre de 2024, Dhaka (Bangladesh) |
| Organización | Una carpeta por clase |

**Contraste con la descarga (2026-11-25).** El .zip se descomprimió en `C:/datos/m2_raw` con `tar -xf`. Contiene una carpeta raíz `Multi-Class Driver Behavior Image Dataset` con cinco subcarpetas. Los conteos de archivos coinciden exactamente con el artículo:

| Carpeta | Clase canónica | Archivos | Artículo |
|---|---|---|---|
| `safe_driving` | conduccion_segura | 1.679 | 1.679 |
| `turning` | giro | 1.343 | 1.343 |
| `talking_phone` | hablar_telefono | 1.513 | 1.513 |
| `texting_phone` | escribir_telefono | 1.561 | 1.561 |
| `other_activities` | otros | 1.190 | 1.190 |
| **Total** | | **7.286** | **7.286** |

Los nombres de carpeta se asignan a las clases con `datos.alias_clases` de `configs/m2.yaml`, sin cambios. Resolución, formato y rotaciones EXIF: [pendiente: inventario].

**Inconsistencias encontradas en el artículo** (se resolverán con los datos, sección 3):
1. **Aumento de datos.** La sección 3.3 dice que el aumento (rotación de hasta 15°, volteo horizontal, ajuste de brillo y recorte aleatorio; la sección 3.4 menciona además zoom) se aplicó en línea durante el entrenamiento. Las secciones 3.1 y 3.4 describen imágenes expandidas guardadas en las carpetas de cada clase. No se dice cuántas fotos son originales ni cuántas copias aumentadas hay por foto.
2. **Lugar.** La tabla de especificaciones dice Savar (Dhaka) y la sección 2 dice Ashulia (Dhaka).

**Implicación para la partición:** si el conjunto publicado contiene versiones aumentadas de la misma foto, una partición aleatoria pondría copias de una misma foto en entrenamiento y en prueba, y la exactitud en prueba quedaría inflada. Como el artículo no lo aclara, se asume el peor caso y se detectan las copias empíricamente.

**Limitación que no se puede eliminar:** el conjunto no trae identificador de conductor. Con 178 conductores, un mismo conductor casi seguro aparece en varias particiones en fotos distintas (fuga de identidad). La detección de casi duplicados agrupa las tomas muy parecidas (misma sesión, misma pose), pero no garantiza separar conductores. Se discutirá como limitación de la evaluación y en la pregunta de generalización.

### 3. Diseño del inventario, la detección de duplicados y la partición

Código: `modulo2/run.py` con los pasos `inventario`, `duplicados` y `particion` (o `datos` para correr los tres).

**Inventario** (`src/m2/inventario.py`): una fila por imagen con clase, tamaño, orientación EXIF, MD5 del archivo y brillo medio.
- Decisión: todas las imágenes se leen aplicando la orientación EXIF (`ImageOps.exif_transpose`). Las fotos de celular guardan a menudo la rotación en los metadatos y no en los píxeles. PIL no la aplica por defecto, así que sin esto una misma foto podría verse girada en un paso y derecha en otro. El inventario registra cuántas imágenes traen rotación EXIF.

**Detección de duplicados** (`src/m2/duplicados.py`): tres detectores complementarios.

| Detector | Qué detecta | Qué no detecta |
|---|---|---|
| MD5 del archivo | Copias exactas byte a byte | Cualquier modificación |
| pHash de 64 bits, de la imagen y de su volteo horizontal; distancia de Hamming | Recompresión, volteos, cambios leves | Rotaciones, recortes y zoom; cambios de brillo grandes (sección 4) |
| Coseno entre embeddings de ResNet-18 preentrenada en ImageNet, de la imagen y de su volteo | Rotación, recorte, zoom y brillo | Puede unir fotos distintas pero muy parecidas (mismo conductor y pose) |

- Los pares se guardan con umbrales laxos (coseno ≥ 0,85 o Hamming ≤ 12). El umbral definitivo se aplica en la partición, así que se puede recalibrar sin recalcular nada.
- **Calibración del umbral:** la Fig. M2-5 muestra pares de ejemplo por franja de coseno. El umbral se fija en la franja más baja donde los pares todavía son la misma foto. Umbrales provisionales: coseno ≥ 0,94 o Hamming ≤ 6. **[pendiente: valor final y justificación con la figura]**

**Partición** (`src/m2/particion.py`):
1. Las imágenes unidas por algún par duplicado forman un grupo (componentes conexas con *union-find*). Así, dos aumentos distintos de una foto quedan juntos aunque no se parezcan entre sí directamente.
2. Partición por grupo y estratificada por clase con `StratifiedGroupKFold` (20 pliegues, `shuffle=True`, semilla 42): 14 pliegues para entrenamiento, 3 para validación y 3 para prueba (aprox. 70/15/15). Ningún grupo queda repartido.
3. **Verificación independiente:** se recalculan las similitudes entre particiones (validación–entrenamiento, prueba–entrenamiento y prueba–validación) y el programa se detiene con error si alguna imagen supera el umbral o si se comparte un MD5. Resultado en `verificacion_fuga.json` y Fig. M2-7.
4. La partición queda fija en `splits.csv` y la usan todos los modelos (regla de comparación justa). Las 30 semillas cambian la inicialización, el orden de los lotes y el aumento de datos, pero no la partición.
5. Los grupos con imágenes de más de una clase se exportan a `grupos_mixtos.csv` para revisarlos a mano: pueden ser errores de etiqueta.

### 4. Prueba del código con datos sintéticos

Antes de usar los datos reales, el código se probó con un conjunto sintético de 230 imágenes de 400 × 600 en cinco carpetas con nombres como los del conjunto real. Se plantaron variantes conocidas de 30 fotos base por clase: copias exactas, volteos horizontales y aumentos de brillo del 20 %. Como en el entorno de prueba no había pesos preentrenados, se probó solo la vía de MD5 y pHash.

**Hallazgo 1. Error en el código detectado por la verificación.** En la primera versión, la verificación de fuga encontró 2 imágenes de validación o prueba con un duplicado en entrenamiento, aunque la agrupación debía impedirlo.
- Causa: la similitud entre A y B se calculaba solo como `máx(sim(A, B), sim(A, volteo(B)))` y cada par se evaluaba en una sola dirección (i < j). Ni el pHash ni el embedding de una imagen volteada son simétricos: sim(A, volteo(B)) ≠ sim(volteo(A), B). Por eso la agrupación y la verificación veían números distintos para el mismo par.
- Corrección: la similitud ahora es el máximo de las tres comparaciones (A–B, A–volteo(B), volteo(A)–B), igual en ambas direcciones (`similitudes_bloque` en `duplicados.py`). Tras corregir, la verificación da 0 violaciones.
- Lección: la verificación independiente de la partición no es un trámite, porque fue lo que detectó el error.
- Este código lo generó una IA (Claude) durante el trabajo. Documentado como hallazgo **H2-1** (categoría Código: fuga de información).

**Hallazgo 2. El pHash solo no basta.** Distancia de Hamming entre cada foto base y sus variantes:

| Variante | n | Hamming medio | Mínimo–máximo |
|---|---|---|---|
| Copia exacta | 15 | 0,0 | 0–0 |
| Volteo horizontal | 40 | 4,0 | 0–6 |
| Brillo +20 % | 25 | 9,2 | 4–14 |

Con el umbral de Hamming ≤ 6, el pHash agrupa todas las copias y todos los volteos, pero se le escapan la mayoría de los cambios de brillo. Esto justifica combinarlo con embeddings preentrenados. Matiz: las imágenes sintéticas tienen poca textura, así que en fotos reales el pHash puede ser más robusto al brillo. La cifra real se verá en la Fig. M2-4.

**Protección añadida:** si un solo grupo reúne más del 5 % de las imágenes, el programa se detiene. Indica un umbral demasiado laxo o un backbone sin pesos preentrenados; con pesos aleatorios, todos los embeddings salen casi idénticos.

### 5. Qué datos procesados se versionarán

Las imágenes crudas (.zip de unos 2,56 GB) no entran al repositorio. Para que la aplicación y las figuras funcionen sin internet se versionarán:
- `inventario.csv`, `splits.csv` y los JSON de verificación (pocos MB);
- **[pendiente]** una copia reducida de las imágenes (p. ej. lado corto de 256 px en JPEG), cuyo tamaño se medirá y se registrará aquí. Si pasa de unos cientos de MB, se evaluará guardar solo la partición de prueba más un subconjunto para el modo demostración;
- los modelos entrenados.

### 6. Inventario sobre los datos reales

Comando: `py run.py inventario --set datos.raw_dir=C:/datos/m2_raw` (77 s en la máquina de la sección 1). Salidas: `data/processed/inventario.csv`, `resumen_inventario.json`, `archivos_ilegibles.csv` y figuras M2-1 a M2-3.

**6.1 Archivos ilegibles.** La primera ejecución se detuvo con `UnidentifiedImageError` en `other_activities/img_13318.jpg`: el código, generado por IA, suponía que todos los archivos eran imágenes válidas (hallazgo **H2-2**). Se cambió el inventario para que excluya y documente los archivos ilegibles en lugar de detenerse. Se identifica su formato real por los primeros bytes y no por la extensión, y se fuerza la decodificación completa de cada imagen para detectar también archivos truncados (`formato_real` en `inventario.py`). Resultado: **10 archivos ilegibles; los 10 tienen tamaño normal (39.889–51.288 bytes) pero su contenido son solo bytes en cero**.

| Clase | Archivos |
|---|---|
| giro (`turning`) | `img_101434.jpg`, `img_62337.jpg`, `img_67523.jpg`, `img_70552.jpg` |
| otros (`other_activities`) | `img_13318.jpg`, `img_13396.jpg`, `img_13541.jpg`, `img_13625.jpg`, `img_20398.jpg`, `img_22266.jpg` |

Quedan **7.276 imágenes legibles**: conducción segura 1.679 · giro 1.339 · escribir 1.561 · hablar 1.513 · otros 1.184. Razón entre la clase mayor y la menor: 1,42.
**[pendiente]** Determinar si los ceros ya vienen así en el .zip publicado o si los produjo la descompresión con `tar`. Herramienta: `tools/verificar_zip.py`, que lee esas entradas directamente del .zip y valida su CRC-32.

**6.2 Formatos y resoluciones: no coinciden con el artículo.** El artículo afirma que todas las imágenes miden 400 × 600 píxeles, que el formato es JPEG y que las fotos se tomaron con dos celulares. En los datos publicados **no hay ninguna imagen de 400 × 600** y conviven al menos seis orígenes, que se distinguen por el patrón del nombre del archivo (columna `origen` de `inventario.csv`, función `origen_archivo`):

| Origen (patrón del nombre) | Formato | Resolución mostrada | Imágenes | Interpretación (hipótesis) |
|---|---|---|---|---|
| `img_N.jpg` | JPEG | 640 × 480 | 6.629 (91,1 %) | Imágenes ya redimensionadas por los autores; la numeración sugiere una secuencia mayor (ver 6.4) |
| `IMG_N_N_HDR(_AE).jpg` | JPEG | 3072 × 4096, 2460 × 3280 y otras | 404 | Fotos originales de celular (posiblemente el Moto Edge 50 Fusion) |
| `IMG_N.JPG` | JPEG | 3024 × 4032 | 32 | Fotos originales de celular (12 MP; coincide con el iPhone 11) |
| `N-N-N-N-N.png` | PNG | 683 × 384 | 183 | Origen desconocido (¿capturas de video?) |
| `Screenshot_N.png` | PNG | ≈ 620 × 900 (irregular) | 17 | Capturas de pantalla |
| `photo_fecha.jpg` | JPEG | 720 × 1280 y otras | 11 | Nombres típicos de imágenes exportadas de una aplicación de mensajería |

Distribución por clase:

| Origen | Segura | Giro | Escribir | Hablar | Otros |
|---|---|---|---|---|---|
| img_640x480 | 1.531 | 1.187 | 1.505 | 1.434 | 972 |
| foto_IMG_fecha_HDR | 96 | 84 | 42 | 50 | 132 |
| foto_IMG_N | 9 | 6 | 0 | 5 | 12 |
| png_683x384 | 24 | 61 | 14 | 24 | 60 |
| captura_pantalla | 11 | 1 | 0 | 0 | 5 |
| photo_fecha | 8 | 0 | 0 | 0 | 3 |

Otros datos del inventario:
- 113 imágenes traen rotación EXIF (todas fotos de alta resolución: el archivo guarda 4096 × 3072 y la imagen se muestra a 3072 × 4096). La decisión de aplicar la orientación EXIF (sección 3) era necesaria.
- **No hay duplicados exactos** (0 MD5 repetidos). Si existen versiones aumentadas guardadas, no son copias byte a byte: tendrán que aparecer como casi duplicados en el paso siguiente.

**6.3 ¿Delata el origen la clase? (riesgo de atajo).** Si un formato o una resolución apareciera casi solo en una clase, la red podría aprender a reconocer el formato en lugar de la conducta. Prueba exploratoria: regresión logística que predice la clase **solo con metadatos**, sin ver los píxeles. Validación cruzada estratificada de 5 pliegues, semilla 42, sobre las 7.276 imágenes.

| Predictor | Exactitud |
|---|---|
| Siempre la clase mayoritaria (azar informado) | 0,231 |
| Origen + relación de aspecto + número de píxeles | 0,238 |
| Lo anterior + tamaño del archivo + brillo medio + rotación EXIF | 0,262 |

Conclusión: en conjunto, los metadatos casi no informan la clase, porque el 91 % de las imágenes tiene el mismo origen y resolución. El riesgo de atajo es bajo a nivel global, pero no nulo dentro de los orígenes minoritarios: el 33 % de los PNG de 683 × 384 son de giro, frente al 18 % de giro en el total. Consecuencias:
- Preprocesamiento: **[pendiente decidir]** cómo llevar todas las imágenes a la misma geometría sin que la deformación o las franjas de relleno delaten el origen (las relaciones de aspecto son 4:3 horizontal, 3:4 vertical y 16:9).
- Evaluación: además de las métricas globales, se reportarán métricas por origen en el conjunto de prueba.

**6.4 Numeración de `img_N.jpg`.** Los números van de 0 a 102.129, pero solo hay 6.629 archivos, y ningún número se repite entre clases. Esto indica una numeración global de una colección mayor de la que se seleccionaron imágenes. Hay números consecutivos (diferencia de 1) dentro de cada clase: entre 22 (giro) y 63 (escribir, hablar). La mediana de la separación entre números consecutivos va de 10 (otros) a 33 (giro). **Hipótesis:** las imágenes numeradas de forma consecutiva o casi consecutiva son tomas de una misma ráfaga o fotogramas de un mismo video, y por tanto casi duplicadas. El artículo no menciona video. Se comprobará en el paso de duplicados comparando la similitud con la diferencia de numeración.

**Inconsistencias del conjunto acumuladas** (para el reporte y para la sección de ética, sesgos de los datos):
1. Momento del aumento de datos, contradictorio en el artículo (sección 2).
2. Lugar de captura: Savar o Ashulia (sección 2).
3. Resolución y formato: el artículo dice 400 × 600 y JPEG; los datos son 640 × 480 en su mayoría, con fotos de hasta 3120 × 4160, PNG y capturas de pantalla (6.2).
4. Origen: el artículo dice dos celulares; hay capturas de pantalla e imágenes con nombres de exportación de mensajería (6.2).
5. Diez archivos con contenido solo de ceros (6.1).

**6.5 Los archivos dañados ya venían así en la publicación.** `tools/verificar_zip.py` leyó las 10 entradas directamente del .zip: CRC-32 correcto y contenido solo de ceros en el propio .zip (`data/processed/verificacion_zip.csv`). No los dañó la descompresión: es un defecto del archivo publicado. Se excluyen.

### 7. Detección de duplicados sobre los datos reales

Comando: `py run.py duplicados --set datos.raw_dir=C:/datos/m2_raw`. Duró 175 s: 98 s para los embeddings de ResNet-18 (pesos `IMAGENET1K_V1`) en la GPU y 14 s para los pares.

**7.1 El umbral de coseno del diseño original no servía (hallazgo H2-5).** Con el coseno crudo, 13.659.949 pares superaron 0,85, más de la mitad de los ~26,5 millones de pares posibles, y 14.758 superaron 0,98. Además, `pares_candidatos.csv` ocupó 899 MB. Diagnóstico, sobre 200.000 pares aleatorios (análisis en la sesión, con `caracteristicas.npz`):

| Medida | p1 | p50 | p99 | p99,9 |
|---|---|---|---|---|
| Coseno crudo, pares aleatorios | 0,704 | 0,851 | 0,933 | |
| Coseno **centrado**, pares aleatorios | −0,235 | 0,020 | 0,557 | 0,827 |

Causa: las características de ResNet salen de un *average pooling* después de una ReLU, así que todas sus componentes son ≥ 0 (mínimo observado: 0,0). Dos vectores no negativos nunca forman ángulos grandes, de modo que el coseno es alto aunque las imágenes no tengan nada que ver. Si a cada vector se le resta el vector promedio del conjunto antes de normalizar (**coseno centrado**), dos imágenes no relacionadas quedan cerca de 0 y la medida sí discrimina. Corrección en el código: **[pendiente]**.

**7.2 El pHash no es confiable en este conjunto.** Pares por distancia de Hamming (mínimo entre original y volteo), con el coseno centrado correspondiente:

| Hamming | Pares | Misma clase | Coseno centrado mediano | Fracción con coseno < 0,85 |
|---|---|---|---|---|
| 0 | 2.747 | 0,97 | 0,92 | 0,25 |
| 2 | 14.931 | 0,81 | 0,82 | 0,59 |
| 4 | 36.469 | 0,60 | 0,69 | 0,84 |
| 6 | 68.522 | 0,42 | 0,58 | 0,96 |

En pares aleatorios, el percentil 0,1 de Hamming ya es 4. Con el umbral provisional de Hamming ≤ 6 habría decenas de miles de falsos duplicados entre imágenes de clases distintas. Decisión: el pHash deja de ser un criterio para agrupar y queda solo como dato descriptivo.

**7.3 Con el coseno centrado, el conjunto está lleno de casi duplicados.** La imagen típica tiene un vecino con coseno centrado de 0,894 (percentiles 5/50/95: 0,734 / 0,894 / 0,972), cuando entre pares aleatorios el percentil 99,9 es 0,827.

| Umbral de coseno centrado | Pares | Imágenes con algún par | Fracción de pares de la misma clase | Grupos (componentes conexas) | Grupo mayor |
|---|---|---|---|---|---|
| 0,85 | 18.544 | 5.213 | 0,98 | 2.830 | 95 |
| 0,90 | 8.320 | 3.390 | 0,99 | 4.718 | 71 |
| 0,94 | — | 1.476 | — | 6.244 | 36 |

Revisión visual de 18 pares al azar, 3 por franja de coseno, de 0,80 a 1,00: **en todos los pares, incluso en la franja de 0,80 a 0,85, aparece la misma persona, en el mismo vehículo y en la misma sesión, con la pose casi idéntica**. Por lo tanto, cualquier umbral ≥ 0,80 deja casi duplicados sin agrupar, y la partición tiene que agrupar de forma más amplia (sección 8).

**7.4 Hipótesis de la numeración consecutiva: descartada.** La hipótesis de 6.4 (números consecutivos = ráfaga o video) no se cumple. Los pares con números consecutivos dentro de una clase se parecen igual que dos imágenes al azar: coseno centrado mediano 0,014 con diferencia de 1 y 0,050 con diferencias mayores de 100; Hamming mediano 24 en todas las franjas. La numeración no refleja el orden de captura.

### 8. Procedencia de las imágenes: evidencia de que el 94 % no se recolectó en Bangladesh en 2024

Al revisar los pares de 7.3 se observaron rasgos que contradicen el artículo: los vehículos tienen el volante a la izquierda (en Bangladesh se conduce por la izquierda y el volante va a la derecha), la cámara está fija en el lado del pasajero, el interior del vehículo se repite en todas las imágenes y hay luz exterior intensa. Se reunió la siguiente evidencia:

| Evidencia | Observación | Fuente |
|---|---|---|
| Fechas de modificación de los archivos dentro del .zip | `img_N.jpg` (6.639, incluidos los 10 dañados) y los PNG de 683 × 384 (183): **todos de octubre de 2019**. Fotos de celular (436), capturas de pantalla (17) y `photo_fecha` (11): septiembre y octubre de 2024 | Listado de la carpeta descomprimida, 2026-11-25 |
| Nombres de los PNG | Marcas de tiempo del 24-04-2019 (p. ej. `2019-04-2416-01-33.png`) | Nombres de archivo |
| Resolución y nombres de `img_N.jpg` | 640 × 480 y el patrón `img_<número>.jpg`, con números de 0 a 102.129 | Inventario (6.2, 6.4) |
| Conjunto State Farm (Kaggle, 2016) | Más de 22.000 imágenes anotadas y más de 79.000 de prueba (≈ 101.000 en total); 26 participantes, 10 clases, 22.424 imágenes de entrenamiento; las imágenes son fotogramas de video y muchas son parecidas; uso limitado a los fines de la competencia | debuggercafe.com; CEUR-WS Vol-3027 paper 43; arXiv 2006.10955 |
| El propio artículo de Afridi et al. | Cita el conjunto de State Farm como antecedente y lo describe como un conjunto de escenarios montados, frente al suyo, que presenta como más auténtico | PMC12019831 |

**Hipótesis (confirmada en la sección 9):** las 6.639 imágenes `img_N.jpg` provienen del conjunto *State Farm Distracted Driver Detection*, y los 183 PNG de 2019 de otra fuente anterior no identificada. Solo **464 imágenes (6,4 %)** serían compatibles con la recolección descrita en el artículo.

**Cómo confirmarla:** el archivo `driver_imgs_list.csv` de la competencia de State Farm (Kaggle) asigna a cada imagen de entrenamiento su conductor (`subject`) y su clase (c0–c9). Si los nombres `img_N.jpg` aparecen ahí con clases coherentes (p. ej. `texting_phone` ↔ c1/c3 y `talking_phone` ↔ c2/c4), la hipótesis queda confirmada y además se obtiene el identificador del conductor. **[pendiente]**

**Consecuencias si se confirma:**
1. **Partición:** con unos 26 conductores, cada uno presente en todas las clases, una partición por imagen o por casi duplicado deja al mismo conductor en entrenamiento y en prueba (fuga de identidad), y las métricas quedarían infladas. Habría que partir por conductor.
2. **Generalización (pregunta de negocio 2):** el modelo aprendería sobre todo de vehículos estadounidenses con volante a la izquierda y escenas montadas, no de conductores de Bangladesh ni de Colombia.
3. **Ética y licencia:** el uso de los datos de State Farm está limitado a la competencia; se debe declarar en la sección de ética y consultar con el profesor.
4. **Reporte:** se debe explicar cómo se construyó realmente el conjunto, como pide el enunciado.

**Inconsistencias del conjunto acumuladas:** se agregan a la lista de la sección 6:
6. Los 10 archivos de ceros vienen así en el .zip publicado (6.5).
7. El 94 % de los archivos tiene fecha de 2019, anterior a la recolección declarada (octubre de 2024), con indicios de que proviene de State Farm (8).

### 9. Confirmación: el 91 % de las imágenes es del conjunto State Farm

Fuente: `driver_imgs_list.csv` (22.424 filas: 26 conductores y 10 clases), descargado de la página de datos de la competencia *State Farm Distracted Driver Detection* en Kaggle el 2026-11-25, tras aceptar sus reglas. Se cruzó por nombre de archivo con `inventario.csv`.

- Las **6.629 imágenes legibles `img_N.jpg`, y también los 10 archivos de ceros, están en la lista de entrenamiento de State Farm** (6.639 de 6.639). Ninguna imagen de los demás orígenes aparece en la lista.
- La correspondencia entre clases es exacta: ninguna imagen cae fuera de estas equivalencias.

| Clase en Afridi et al. | Clase de State Farm | Imágenes (legibles) |
|---|---|---|
| conducción segura (`safe_driving`) | c0 *safe driving* | 1.531 |
| escribir en el teléfono (`texting_phone`) | c1 *texting – right* + c3 *texting – left* | 751 + 754 |
| hablar por teléfono (`talking_phone`) | c2 *talking on the phone – right* + c4 *talking on the phone – left* | 743 + 691 |
| **giro (`turning`)** | **c7 *reaching behind*** | 1.187 |
| otros (`other_activities`) | c5 *operating the radio* + c6 *drinking* + c8 *hair and makeup* + c9 *talking to passenger* | 211 + 195 + 275 + 291 |

- **26 conductores**, todos presentes en las cinco clases, con entre 127 y 384 imágenes cada uno (mediana 249,5).

**Composición real del conjunto (7.276 imágenes legibles):**

| Procedencia | Imágenes | % |
|---|---|---|
| State Farm (Kaggle, EE. UU.; escenas montadas; 26 conductores) | 6.629 | 91,1 |
| PNG de 683 × 384 con fecha de 2019, de procedencia no identificada | 183 | 2,5 |
| Fotos de celular, capturas de pantalla e imágenes de mensajería de 2024 (compatibles con lo que describe el artículo) | 464 | 6,4 |

**Consecuencias:**
1. **Significado de la clase «giro».** El artículo la describe como conductores que mueven la cabeza o el cuerpo para cambiar de dirección, es decir, una maniobra normal. Pero el 88,6 % de sus imágenes (1.187 de 1.339) son *reaching behind* de State Farm: estirarse hacia el asiento trasero, que es una distracción. La clase mezcla dos conductas distintas. Esto invalida la agrupación de alertas que proponía la guía de trabajo («giro» sin alerta) (hallazgo **H2-7**). La política de alertas se debe redefinir con el significado real.
2. **Partición.** Con 26 conductores presentes en todas las clases, partir por imagen o por casi duplicado deja al mismo conductor en entrenamiento y prueba. La partición de las imágenes de State Farm debe hacerse **por conductor** (`subject`). Para las 647 imágenes de otros orígenes no hay identificador de conductor: se agrupan por similitud (coseno centrado).
3. **Generalización.** El modelo aprenderá sobre todo de 26 conductores en escenas montadas en EE. UU. Las 464 imágenes de 2024 son las únicas cercanas al caso de uso real (celular, otros vehículos), así que conviene reportar su desempeño por separado.
4. **Licencia y ética.** El conjunto de State Farm se distribuyó bajo las reglas de una competencia de Kaggle que limitan su uso. Afridi et al. lo redistribuyen dentro de su conjunto sin declararlo. Va a la sección de ética, y conviene consultarlo con el profesor.
5. **Reporte.** La sección «cómo se construyó el conjunto» debe contar esta composición real, con esta evidencia.

### 10. Decisiones y partición definitiva

**Decisiones de la Persona B (2026-11-25):**
1. **Partición por conductor, con todas las procedencias mezcladas.** Las imágenes de State Farm se parten por conductor. Las demás se parten por grupos (definidos abajo), por separado dentro de cada procedencia, para que entrenamiento, validación y prueba tengan imágenes de ambas. Se reportarán métricas por procedencia. Alternativas descartadas: (a) reservar todas las fotos de 2024 como prueba externa, porque el modelo no vería fotos de celular al entrenar; (b) usar solo State Farm, porque se pierden las imágenes más parecidas al caso de uso.
2. **Se mantienen las 5 clases publicadas.** «Giro» dispara alerta (política: alerta = todo lo que no sea conducción segura), y en el reporte se explica su significado real (sección 9).

**Cambios en el código:**
- `duplicados.py`: coseno **centrado** (se resta el vector promedio antes de normalizar); pHash solo descriptivo; pares guardados comprimidos (`pares_candidatos.csv.gz`, 1,2 MB con coseno ≥ 0,60, frente a los 899 MB anteriores); distribución nula de 200.000 pares aleatorios en `resumen_duplicados.json`.
- `particion.py`: grupos = conductor de State Farm ∪ sesión de captura ∪ pares con coseno centrado ≥ 0,75. Partición por grupos dentro de cada procedencia, eligiendo entre 2.000 órdenes aleatorios (semilla 42) el más cercano a 70/15/15 y a la distribución de clases.

**Calibración (análisis con las características reales, centradas):**

| Comprobación | Resultado |
|---|---|
| Distribución nula (200.000 pares aleatorios): percentiles 50 / 99 / 99,9 | 0,020 / 0,554 / 0,821 |
| Coseno centrado máximo entre imágenes de **conductores distintos** de State Farm | 0,656. Percentiles 50 / 99 del máximo por imagen: 0,483 / 0,616. Ningún par ≥ 0,75 |
| Coseno centrado máximo entre una imagen de State Farm y una de otra procedencia | 0,529 |
| Revisión visual de pares de 2024 en particiones distintas, con coseno 0,73–0,74 (primera versión, agrupando solo por similitud) | Varios pares son **la misma persona** (p. ej. `IMG_3735`/`IMG_3730`, `IMG_3754`/`IMG_3755`) y otros son personas distintas en el mismo bus |

Conclusiones:
- En State Farm, partir por conductor basta: ningún par entre conductores distintos supera 0,656.
- En las fotos de 2024, la similitud no separa a las personas: la misma persona puede quedar por debajo del umbral, y personas distintas en el mismo vehículo pueden quedar por encima. Por eso estas imágenes se agrupan además por **sesión de captura**: fotos del mismo origen con menos de 10 minutos entre una y la siguiente, según la fecha y la hora del nombre del archivo. Los orígenes sin fecha en el nombre (`IMG_N.JPG`, capturas de pantalla) forman una sesión cada uno. La primera versión, solo por similitud, habría dejado a la misma persona en dos particiones (hallazgo **H2-8**).
- En las fotos de 2024, dos sesiones HDR quedan unidas por similitud en un grupo de 232 de 647 imágenes (36 %). Por eso `max_fraccion_grupo` sube de 0,35 a 0,45. El grupo queda entero en entrenamiento.

**Partición resultante** (`py run.py particion`; corrida de verificación con las características reales, 2026-11-25):

| | Entrenamiento | Validación | Prueba |
|---|---|---|---|
| Imágenes | 5.076 (69,8 %) | 1.067 (14,7 %) | 1.133 (15,6 %) |
| De State Farm | 4.633 | 999 | 997 |
| De otras procedencias | 443 | 68 | 136 |
| Conductores de State Farm | 17 | 5 (p041, p042, p064, p066, p072) | 4 (p012, p039, p051, p075) |
| Grupos de otras procedencias | 5 (capturas de pantalla; HDR s1+s3; `photo_fecha` s0 y s1; PNG) | 2 (`IMG_N`; HDR s2) | 1 (HDR s0, 136 fotos) |
| Proporción de conducción segura / giro / escribir / hablar / otros | 0,224 / 0,185 / 0,217 / 0,210 / 0,164 | 0,238 / 0,175 / 0,210 / 0,217 / 0,159 | 0,252 / 0,187 / 0,207 / 0,192 / 0,162 |

Verificación independiente: 0 grupos, 0 MD5 y 0 imágenes con un par ≥ 0,75 compartidos entre particiones. Coseno centrado máximo entre particiones: validación→entrenamiento 0,734; prueba→entrenamiento 0,626; prueba→validación 0,566.

**Limitaciones de esta partición:**
- La prueba de otras procedencias es una sola sesión de 2024 (136 fotos): mide la generalización a esa sesión, con poca precisión estadística.
- En validación, de otras procedencias, hay solo 2 imágenes de «escribir en el teléfono».
- Con 4 conductores de State Farm en prueba, las métricas dependen de qué conductores tocaron. Las 30 semillas no capturan esta variabilidad, porque la partición es fija.
**Confirmación en la máquina de la sección 1 (2026-11-25).** Se borró el `pares_candidatos.csv` antiguo y se corrieron `py run.py duplicados` (105,6 s: 91 s de embeddings en la GPU; 189.815 pares registrados con coseno ≥ 0,60; `pares_candidatos.csv.gz`) y `py run.py particion` (6,8 s). Los resultados coinciden **exactamente** con la corrida de verificación: los mismos conductores por partición, los mismos conteos, las mismas proporciones y la misma verificación (0 fugas; máximos 0,734 / 0,626 / 0,566). La distribución nula y los percentiles del vecino más cercano también coinciden con los de la sección 7 (nula: 0,0196 / 0,554 / 0,821; vecino más cercano: 0,734 / 0,894 / 0,972).

Pares registrados por franja de coseno centrado: [0,60–0,65) 63.803 · [0,65–0,70) 41.988 · [0,70–0,75) 27.935 · [0,75–0,80) 21.483 · [0,80–0,85) 16.054 · [0,85–0,90) 10.222 · [0,90–1,00] 8.330.

La Persona B revisó las figuras M2-5 y M2-7 sin observaciones, e hizo los commits de la etapa de datos (2026-11-25).

### 11. Preprocesamiento y modelo de referencia: diseño

**Preprocesamiento** (`py run.py preproceso`, `src/m2/preproceso.py`): una sola versión de las imágenes para todos los modelos, guardada en `data/processed/imagenes/` (JPEG, calidad 90) con la misma estructura de carpetas que el conjunto. La columna `ruta_proc` se agrega a `splits.csv`.
- **Tamaño 224 × 224**, el de entrada de las redes preentrenadas en ImageNet.
- **Geometría: estirar** (redimensionar sin conservar la proporción). Las proporciones originales son 4:3 horizontal (State Farm), 3:4 vertical (fotos de celular) y 16:9 (PNG). Alternativas consideradas:
  - recortar al centro: pierde los bordes, donde en State Farm suelen quedar las manos y el teléfono;
  - rellenar con franjas: las franjas delatan la proporción, y por tanto el origen.
  Estirar no pierde contenido, y la deformación distinta por origen es un riesgo bajo porque el origen casi no informa la clase (6.3: 0,238 frente a 0,231 al azar). Las tres opciones quedan en `preproceso.geometria` para un análisis de sensibilidad.
- Tamaño en disco: 121,4 MB; se versionan completas (sección 12).

**Modelo de referencia** (`py run.py baseline`, `src/m2/baseline.py`):
- **Red congelada: EfficientNet-B0** preentrenada en ImageNet (pesos por defecto de torchvision), la misma que se ajustará en transfer learning. Justificación: es liviana (unos 5,3 millones de parámetros), adecuada para el cómputo embarcado de la sección «caso de uso», y permite comparar «misma red, congelada frente a ajustada».
- Características: salida de 1.280 dimensiones de la penúltima capa, extraída una sola vez y sin aumento de datos. Estandarizadas con la media y la desviación del entrenamiento.
- Regresión logística multinomial (scikit-learn, `lbfgs`, máx. 3.000 iteraciones). **Presupuesto de búsqueda: 11 valores de C** (de 10⁻⁴ a 10, en escala logarítmica), elegidos por F1 macro en validación. El modelo final se ajusta solo con entrenamiento, como las redes, que usan validación para la parada temprana.
- Parámetros entrenables: 1.280 × 5 + 5 = 6.405.
- **Variabilidad:** el modelo es determinista; se reportan 1.000 remuestras *bootstrap* del conjunto de prueba (semilla 42), para compararlas en la misma figura con las 30 semillas de las redes. Es una aproximación declarada: el bootstrap mide la incertidumbre del conjunto de prueba, mientras que las semillas miden la variabilidad del entrenamiento.
- Salidas: `models/baseline_logreg.joblib`, `resultados/baseline_*.csv|json` y Fig. M2-8.
- Prueba del código con datos sintéticos (230 imágenes, red sin pesos): el flujo completo corre (preproceso → características → búsqueda → métricas → bootstrap → figura). Las métricas de esa prueba no tienen significado.

### 12. Resultados del preprocesamiento y del modelo de referencia (2026-11-25)

**Preprocesamiento** (`py run.py preproceso`): 7.276 imágenes en 106 s; **121,4 MB** en total (unos 17 KB por imagen). Ningún archivo se acerca al límite de 100 MB por archivo de GitHub, así que las imágenes procesadas **se versionan completas**: cumplen el requisito de que la ejecución no dependa de los datos crudos ni de internet.

**Modelo de referencia** (`py run.py baseline`; 95 s en total: 33 s de extracción en la GPU y 26 s de búsqueda; pesos `efficientnet_b0_rwightman-7f5810bc`):

| C | 0,0001 | 0,0003 | 0,001 | 0,003 | 0,01 | **0,03** | 0,1 | 0,3 | 1 | 3 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 macro (validación) | 0,542 | 0,556 | 0,564 | 0,560 | 0,559 | **0,571** | 0,568 | 0,559 | 0,562 | 0,555 | 0,537 |

La curva es casi plana (0,54–0,57): la regularización influye poco. Con el C elegido (0,03), la exactitud en entrenamiento es 1,00: el modelo memoriza el entrenamiento.

| Conjunto | Exactitud | F1 macro |
|---|---|---|
| Validación | 0,580 | 0,571 |
| **Prueba** | **0,705** | **0,698** |
| Prueba, State Farm (997) | 0,760 | 0,747 |
| Prueba, fotos de 2024 (136) | 0,301 | 0,237 |
| Prueba, bootstrap de 1.000 remuestras: media ± desviación (mín.–máx.) | 0,705 ± 0,013 (0,657–0,750) | 0,697 ± 0,013 (0,653–0,745) |

Por clase, en prueba (precisión / sensibilidad): conducción segura 0,81 / 0,83 · giro 0,72 / 0,76 · escribir 0,85 / 0,55 · hablar 0,76 / 0,73 · otros 0,44 / 0,62. Los errores más frecuentes: escribir confundido con otros (44) y con giro (39); hablar confundido con otros (49).

**Análisis** (a partir de `resultados/baseline_predicciones.csv` y de `splits.csv`):
1. **La variabilidad entre conductores domina.** Exactitud por conductor: en prueba, p012 0,76 · p039 0,79 · p051 0,90 · p075 0,57; en validación, p041 0,55 · p042 0,89 · p064 0,59 · p066 0,49 · p072 0,54. La diferencia entre validación (0,58) y prueba (0,71) se explica porque en validación quedaron conductores más difíciles. Consecuencias:
   - el bootstrap por imagen (desviación 0,013) **subestima** la incertidumbre, porque las imágenes de un mismo conductor no son independientes. **[pendiente]** Agregar un bootstrap por conductor o grupo;
   - las 30 semillas de las redes tampoco capturan esta variabilidad, porque la partición es fija. Se declara como limitación.
2. **Generalización a fotos reales: muy baja.** En la sesión de 2024 de prueba, la exactitud es 0,30, cerca del azar (0,20 con 5 clases), y el modelo **nunca predice «conducción segura»** (0 de 28). En validación (dos sesiones de 2024) la exactitud es 0,25. Las características de ImageNet con una capa lineal entrenada sobre el 91 % de State Farm no se trasladan a fotos de celular en buses de Bangladesh. Es evidencia directa para la pregunta de negocio 2.
3. La clase «otros» (radio, beber, peinarse o maquillarse, hablar con el pasajero) tiene la precisión más baja (0,44): agrupa conductas heterogéneas y recibe errores de todas las demás. Esto confirma la hipótesis de dificultad.

### 13. Redes neuronales: diseño del código

Archivos: `src/m2/modelos.py`, `entrenamiento.py` y `redes.py`. Comandos: `py run.py entrenar | busqueda | semillas`, con `--set red.familia=cnn|transfer`.

**Arquitecturas:**
- **CNN desde cero** (`CNNDesdeCero`): b bloques [Conv 3×3 – BatchNorm – ReLU] × 2 + MaxPool 2×2; los filtros empiezan en `filtros_base` y se duplican en cada bloque. Con 5 bloques, la entrada de 224 × 224 queda en un mapa de 7 × 7. Cabeza: *Global Average Pooling* → Dropout → Lineal(5). Inicialización de Kaiming (He) para las convoluciones.
- **Transfer learning:** EfficientNet-B0 preentrenada (la misma red del modelo de referencia), con la cabeza cambiada por Dropout → Lineal(1.280 → 5). **Dos etapas:** durante `epocas_cabeza` épocas solo se entrena la cabeza; después se descongelan los últimos `bloques_descongelados` bloques de `features` (de 0 a 9), con tasa de aprendizaje `lr × factor_lr_base`. Los bloques congelados mantienen su BatchNorm en modo evaluación, para conservar las estadísticas de ImageNet.

**Entrenamiento** (común a las dos familias):
- Entropía cruzada.
- AdamW (weight decay desacoplado) o SGD con momento de Nesterov.
- Calentamiento lineal de 1 época y decaimiento coseno por paso.
- Precisión mixta bfloat16 en la GPU.
- Parada temprana por F1 macro en validación (`paciencia` épocas sin mejora), conservando los pesos de la mejor época.
- Aumento de datos solo en entrenamiento y en línea, con tres intensidades (leve, medio, fuerte): recorte aleatorio con reescalado, rotación y brillo/contraste/saturación. El volteo horizontal se usa solo en la intensidad fuerte: no cambia la etiqueta, porque las clases «derecha» e «izquierda» de State Farm están unidas.
- La semilla controla la inicialización, el orden de los lotes, el dropout y el aumento de datos; la partición es fija.

**Búsqueda** (`py run.py busqueda`): búsqueda aleatoria con un **presupuesto fijo de 16 configuraciones por familia**, semilla 42, máximo 12 épocas y paciencia 4; criterio: F1 macro en validación. Justificación de la búsqueda aleatoria frente a la rejilla: con varios hiperparámetros de importancia desigual, explora más valores distintos de cada uno con el mismo presupuesto (Bergstra y Bengio, 2012). Espacio (`busqueda.espacio` en `configs/m2.yaml`):
- CNN: optimizador y lr {AdamW 3·10⁻⁴, 10⁻³, 3·10⁻³; SGD 0,01, 0,03}; weight decay {10⁻⁴, 10⁻³, 0,01, 0,05}; dropout {0,2, 0,4, 0,5}; bloques {4, 5}; filtros base {16, 32}; aumento {leve, medio, fuerte}.
- Transfer: los mismos optimizadores y lr; weight decay {10⁻⁴, 0,01, 0,05}; dropout {0,2, 0,4}; bloques descongelados {0, 2, 4, 9}; factor de lr de la base {0,05, 0,1, 0,3}; aumento {leve, medio, fuerte}.

La búsqueda se puede interrumpir y reanudar sin repetir configuraciones. La mejor queda en `resultados/mejor_config_<familia>.yaml`, con `epocas_max` 30 y paciencia 6 para la corrida final.

**30 semillas** (`py run.py semillas`): la mejor configuración con las semillas 0 a 29. También se reanuda. El modelo de la **semilla 0** se guarda como modelo final (`models/<familia>_final.pt`); se eligió de antemano, no por su resultado, para no favorecer una semilla afortunada. Se guardan las probabilidades en prueba de las 30 semillas (`prob_prueba_<familia>.npz`).

**`py run.py entrenar`:** una corrida con la configuración `red` del YAML más los cambios de la CLI, por ejemplo `--set red.lr=0.003`. Es la que se usa para la modificación en vivo de la sustentación.

**Pruebas del código** (datos sintéticos, CPU):
- Las dos familias entrenan, evalúan y guardan resultados.
- La búsqueda y las semillas se reanudan sin repetir trabajo.
- Prueba de la congelación en dos etapas, comparando el `state_dict` antes y después de cada época: en la etapa 1 solo cambia la cabeza (bloques 0–8 y sus estadísticas de BatchNorm intactos); en la etapa 2 cambian los bloques 7–8 descongelados y la cabeza, y los bloques 0–6 siguen intactos.
- Se corrigió un problema de diseño: las claves del espacio de búsqueda con «.» no se podían cambiar con `--set` (hallazgo **H2-9**). Ahora se escriben con «/» (p. ej. `cnn/bloques`).

### 14. Costo de cómputo medido (2026-11-25)

`py run.py entrenar --set red.familia=<familia> --set red.epocas_max=2` con la configuración `red` por defecto (AdamW, lr 10⁻³, aumento medio, lote de 64):

| Familia | Época 1 | Época 2 | F1 macro val. (2 épocas) | F1 macro prueba | F1 prueba, fotos de 2024 | Parámetros entrenables |
|---|---|---|---|---|---|---|
| Transfer (EfficientNet-B0, 4 bloques; las 2 épocas solo entrenan la cabeza) | 29 s | 8 s | 0,377 | 0,485 | 0,191 | 3.705.293 de 4.013.953 |
| CNN desde cero (5 bloques, 32 filtros base) | 31 s | 11 s | 0,624 | 0,547 | 0,100 | 4.716.773 |

La primera época incluye el arranque de los procesos de carga y el calentamiento de la GPU (unos 20 s). Estimaciones:
- búsqueda: unos 16 × (30 s + 11 × 12 s) ≈ 45 min por familia como máximo;
- 30 semillas: hasta unas 3 h por familia con 30 épocas, menos con la parada temprana.
Son corridas indicativas de 2 épocas, no resultados.

### 15. Búsqueda de hiperparámetros y 30 semillas: resultados (2026-11-25)

**Búsqueda** (`py run.py busqueda`; 16 configuraciones por familia; semilla 42; máx. 12 épocas; paciencia 4). Fig. M2-9 y `resultados/busqueda_<familia>.csv`.

| | Transfer learning | CNN desde cero |
|---|---|---|
| Cómputo de la búsqueda | 31,8 min | 33,6 min |
| F1 macro en validación: mejor / mediana / peor | **0,860** / 0,666 / 0,433 | **0,766** / 0,728 / 0,183 |
| Configuración elegida | #5: AdamW, lr 0,003, weight decay 10⁻⁴, dropout 0,2, **4 bloques descongelados**, factor de lr de la base 0,3, aumento leve | #10: SGD con Nesterov, lr 0,03, weight decay 0,01, dropout 0,4, **5 bloques, 16 filtros base**, aumento leve |
| Parámetros entrenables de la elegida | 3.705.293 de 4.013.953 (cabeza 6.405) | 1.181.045 |

Patrones de la búsqueda:
- **El aumento fuerte perjudica en ambas familias:** en la CNN, todas las configuraciones con aumento fuerte quedaron por debajo de 0,61, y en transfer, por debajo de 0,68. 6 de las 7 mejores configuraciones de la CNN (la excepción, la #4, usa aumento medio) y las 4 mejores de transfer usan aumento leve. Interpretación: en State Farm la cámara está fija, así que recortes del 60 % y rotaciones de 15° alejan las imágenes de entrenamiento de las de evaluación sin aportar una variación realista.
- **Transfer con 0 bloques descongelados** (solo se entrena la cabeza, con aumento) obtuvo entre 0,43 y 0,57, el mismo nivel que el modelo de referencia (0,571). Lo que aporta el transfer learning es **ajustar los bloques de la base**, no la cabeza.

**30 semillas** (`py run.py semillas`; semillas 0 a 29; máx. 30 épocas; paciencia 6; la partición es la misma en todas). Fig. M2-10 (curvas de la semilla 0), Fig. M2-11 (cajas), Fig. M2-12 (matriz de confusión de transfer, semilla 0) y `resultados/comparacion_modelos.csv`.

| Modelo | Variabilidad | F1 macro prueba: media ± desv. (mín.–máx.) | F1 macro val. (media) | F1 prueba State Farm | F1 prueba fotos de 2024 | Parámetros entrenables | Búsqueda | 30 semillas |
|---|---|---|---|---|---|---|---|---|
| Referencia (logística sobre EfficientNet-B0 congelada) | bootstrap de prueba (1.000) | 0,697 ± 0,013 (0,653–0,745) | 0,571 | 0,747 | 0,237 | 6.405 | 11 config., 0,4 min | — |
| CNN desde cero | 30 semillas | 0,656 ± 0,102 (0,419–0,813) | 0,690 | 0,703 | 0,138 | 1.181.045 | 16 config., 33,6 min | 65,8 min |
| **Transfer learning (EfficientNet-B0)** | 30 semillas | **0,855 ± 0,021 (0,778–0,879)** | 0,829 | **0,913** | **0,387** | 3.705.293 | 16 config., 31,8 min | 70,5 min |

Mediana de la mejor época: transfer 8,5; CNN 7. Correlación entre el F1 de validación y el de prueba a lo largo de las semillas: transfer 0,49; CNN 0,84.

**Análisis:**
1. **El transfer learning supera claramente a la referencia:** +0,16 de F1 macro en prueba. Incluso su peor semilla (0,778) supera la mejor remuestra del bootstrap de la referencia (0,745). Además es estable entre semillas (desviación 0,021). El costo es unas 580 veces más parámetros entrenables y unos 100 minutos de GPU entre búsqueda y semillas, frente a menos de un minuto de la referencia.
2. **La CNN desde cero no supera a la referencia en promedio** (0,656 frente a 0,697) y es **muy inestable** (desviación 0,102; de 0,42 a 0,81). Varias semillas se detienen pronto (mejor época entre 2 y 5) con F1 bajo. Con SGD y lr 0,03, el F1 de validación oscila mucho de una época a otra (Fig. M2-10), y la parada temprana puede quedarse con un máximo temprano.
3. **Optimismo de la búsqueda (*winner's curse*).** La configuración de la CNN se eligió con F1 de validación 0,766 en una sola semilla (42), pero la media de las 30 semillas con la misma configuración es 0,690. Elegir el máximo entre 16 configuraciones con una sola semilla favorece a las que tuvieron suerte. En transfer la diferencia es menor (0,860 frente a 0,829). Por eso la comparación se hace con las 30 semillas y no con el valor de la búsqueda.
4. **Generalización a fotos reales:** todos los modelos caen con fuerza en la sesión de 2024 de prueba (136 fotos de celular en buses). Transfer llega a 0,387 de F1, por encima del azar (≈ 0,2) pero lejos de su 0,913 en State Farm. La CNN desde cero (0,138) queda incluso por debajo de la referencia (0,237): sin conocimiento previo de ImageNet aprende rasgos específicos de State Farm. Es evidencia directa para la pregunta de negocio 2.
5. **Errores por clase** (transfer, semilla 0, Fig. M2-12): «otros» es el sumidero de errores. Recibe el 12 % de conducción segura, el 11 % de escribir y el 17 % de hablar, y tiene la menor precisión (0,55). Escribir y hablar casi no se confunden entre sí (≤ 1 %).

**Limitación ya señalada (sección 12):** las 30 semillas miden la variabilidad del entrenamiento con la partición fija. No miden la variabilidad por la elección de conductores en prueba, que en la referencia va de 0,49 a 0,90 de exactitud por conductor. **[pendiente]** Bootstrap por conductor o grupo.

**Revisión de las figuras:** al revisar la Fig. M2-9 se encontró que los dos paneles compartían el eje Y, y las etiquetas de la CNN mostraban las configuraciones de transfer. Se corrigió (hallazgo **H2-10**).

### 16. Evaluación del modelo final: diseño (`py run.py evaluacion`, `src/m2/evaluacion.py`)

Usa los modelos guardados (`models/<familia>_final.pt`, semilla 0), sin reentrenar. El modelo interpretado es **transfer learning** (`evaluacion.familia`).

1. **Grad-CAM** (Selvaraju et al., 2017), implementado con *hooks* de PyTorch sobre la última capa convolucional: `features[8]` en EfficientNet-B0 y la salida del bloque 5 en la CNN, ambas de 7 × 7. Mapa: ReLU(Σₖ αₖᶜ Aᵏ), con αₖᶜ = media espacial de ∂yᶜ/∂Aᵏ, para la clase predicha; se interpola a 224 × 224 y se normaliza a [0, 1].
   - **10 aciertos y 10 errores** de prueba (Figs. M2-13 y M2-14; listados en `resultados/gradcam_aciertos.csv` y `gradcam_errores.csv`). Selección con semilla 42: una imagen por clase real y el resto al azar; entre los errores se reservan hasta 3 fotos de 2024.
   - **Grad-CAM promedio por clase** sobre los aciertos de State Farm en prueba (Fig. M2-15). Como la cámara de State Farm es fija, el promedio muestra en qué zonas de la escena se apoya el modelo para cada clase.
2. **Política de alertas:**
   - Puntaje = 1 − P(conducción segura); alerta si puntaje ≥ umbral. «Giro» dispara alerta (sección 10).
   - Umbrales de 0,02 a 0,98 (paso 0,02).
   - El umbral se **elige en validación**, minimizando el costo esperado = falsas alarmas + r × no detectadas por cada 1.000 imágenes, con cocientes de costo r ∈ {1, 5, 10, 20}. Se **reporta en prueba**.
   - Dos prevalencias de distracción: la del conjunto de prueba (≈ 75 %, irreal) y una **supuesta del 10 %** para la operación (`alertas.prevalencia_supuesta`). Por cada 1.000 imágenes: falsas alarmas = 1.000 (1 − π) TFA; no detectadas = 1.000 π TND.
   - Fig. M2-16 y `resultados/alertas_umbral_prueba.csv`.
3. **Exactitud por grupo de prueba** (4 conductores de State Farm y la sesión de 2024) para la referencia, las 30 semillas de la CNN y las 30 de transfer (Fig. M2-17; `resultados/metricas_por_grupo_prueba.csv`). Muestra la variabilidad por conductor que las semillas no capturan. Reemplaza al bootstrap por conductor planeado en la sección 12: con solo 5 grupos en prueba, un bootstrap de grupos sería muy poco informativo.
4. **Diagrama de arquitectura** (Fig. M2-18): capas, dimensiones de salida (pasada con una imagen de 224 × 224), tipo y activación, y parámetros por capa. Comprobación: EfficientNet-B0 suma 4.013.953 parámetros, y los bloques 5 a 8 descongelados suman 3.698.888, igual que en el entrenamiento (sección 15).

**Pruebas** con datos sintéticos y modelos mínimos: el paso completo corre. Se corrigió un error: el *hook* de Grad-CAM quedaba registrado y fallaba en una pasada posterior sin gradientes; ahora solo actúa con gradientes y se retira al terminar.

Resultados reales: **[pendiente]**.

### Próximos pasos

1. Verificar con A y C que todo el equipo usa Python 3.14 y PyTorch 2.11.0, como se acordó en el día 0.
2. Informar a A, C y al profesor sobre la procedencia del conjunto.
3. Correr `py run.py evaluacion`, revisar los Grad-CAM y explicar los 10 errores.
4. `api.py`, página de la app y modo demo.
