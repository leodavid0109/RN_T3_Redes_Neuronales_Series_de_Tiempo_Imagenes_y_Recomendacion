# Uso de inteligencia artificial — Módulo 2

**Herramienta:** Claude (Anthropic), usado como asistente de programación y análisis desde la aplicación de escritorio, con acceso de lectura y escritura a la carpeta del repositorio y acceso de lectura a la carpeta de datos (`C:/datos`). La IA no ejecutó entrenamientos en la GPU: escribió y probó el código en su propio entorno (con datos sintéticos o con archivos derivados de los reales), y la Persona B lo ejecutó en su máquina y le devolvió las salidas.

Los prompts se presentan editados para mayor claridad, con fines educativos.

## 1. Interacciones principales

### Etapa 1. Planeación y entorno

| # | Fecha | Prompt | Resultado de la IA | Impacto en el trabajo |
|---|---|---|---|---|
| 1 | 2026-11-25 | Con base en la distribución de trabajo aprobada (rol de Persona B: Módulo 2, fundamentación matemática común y marco de ética), elaborar una hoja de ruta técnica con las tareas, los métodos y los entregables de cada etapa. | Guía de trabajo por etapas (datos, partición sin fuga, modelos, evaluación, preguntas de negocio, matemática, ética). | Sirvió de plan. Una de sus recomendaciones («giro» sin alerta) resultó incorrecta (H2-7). |
| 2 | 2026-11-25 | Configurar el proyecto para entrenar en la GPU local por reproducibilidad, conectar la carpeta del repositorio e implementar el primer bloque del flujo de datos (inventario, detección de duplicados y partición). | Código de `inventario.py`, `duplicados.py` y `particion.py`, con determinismo y semillas fijas; pruebas con datos sintéticos. | La verificación de fuga incluida en el código detectó un error del propio código (H2-1). |
| 3 | 2026-11-25 | Mantener una bitácora técnica del Módulo 2, actualizada en cada paso, que se pueda anexar sin cambios a la documentación del equipo. | `docs/registro/registro_modulo2.md`. | Fuente principal para redactar `modulo_2.md`. |
| 4 | 2026-11-25 | Indicar el comando de instalación de PyTorch compatible con la RTX 5060 (arquitectura Blackwell). | Comando con el índice CUDA 12.8; aviso de que la información consultada en la web podía estar desactualizada. | La consulta web devolvió una versión desactualizada (H2-3). |

### Etapa 2. Datos y procedencia del conjunto

| # | Fecha | Prompt | Resultado de la IA | Impacto en el trabajo |
|---|---|---|---|---|
| 5 | 2026-11-25 | Diagnosticar el fallo del inventario con un archivo que no se puede leer y hacerlo robusto frente a archivos dañados. | Inventario que identifica el formato real por los primeros bytes y documenta los archivos ilegibles. | 10 archivos con contenido solo de ceros (H2-2). |
| 6 | 2026-11-25 | Interpretar el resumen del inventario y contrastarlo con la descripción del artículo de los autores. | Análisis de resoluciones, orígenes por patrón de nombre, numeración y riesgo de atajo por metadatos. | Se descubrió que los datos no coinciden con el artículo (400 × 600 px, solo JPEG). |
| 7 | 2026-11-25 | Establecer como norma que todo error de la IA se documente como hallazgo, con evidencia y con la plantilla del enunciado. | `docs/alucinaciones/modulo2.md` y política de errores en el registro. | Revisión retrospectiva que encontró H2-4. |
| 8 | 2026-11-25 | Analizar los resultados de la verificación del .zip y de la detección de duplicados, y calibrar el umbral de similitud. | Diagnóstico del coseno sin centrar, distribución nula, revisión visual de pares y análisis de fechas de archivo. | H2-5 y H2-6; indicios de que el 94 % de los archivos son anteriores a 2024. |
| 9 | 2026-11-25 | Guiar la obtención de la lista de conductores de la competencia de State Farm en Kaggle para confirmar la procedencia de las imágenes. | Cruce de nombres con `driver_imgs_list.csv`. | **Confirmación de que el 91 % del conjunto es State Farm (26 conductores)** y de que «giro» es en su mayoría *reaching behind*. |
| 10 | 2026-11-25 | Partición del Módulo 2: separar por conductor y mezclar todas las procedencias en las tres particiones. Mantener las 5 clases publicadas y que «giro» dispare alerta. | Partición por conductor, sesión de captura y similitud; verificación visual. | Partición sin fuga y confirmada; H2-8. |

### Etapa 3. Modelos

| # | Fecha | Prompt | Resultado de la IA | Impacto en el trabajo |
|---|---|---|---|---|
| 11 | 2026-11-25 | Definir una práctica de control de versiones: cuándo y con qué granularidad hacer commits. | Pauta: un commit por avance lógico, separando código, resultados y documentos, en ramas `m2/…`. | Historial de commits que refleja el trabajo por etapas. |
| 12 | 2026-11-25 | Continuar con el preprocesamiento y el modelo de referencia (regresión logística sobre una red preentrenada congelada). | `preproceso.py`, `baseline.py` y `metricas.py`. | Modelo de referencia: F1 macro en prueba de 0,698; casi azar en las fotos de 2024. |
| 13 | 2026-11-25 | Implementar la CNN desde cero y el transfer learning con búsqueda de hiperparámetros, 30 semillas y un modo de entrenamiento para la modificación en vivo. | `modelos.py`, `entrenamiento.py` y `redes.py`, con pruebas de la congelación en dos etapas y de la reanudación. | H2-9 y un error de proceso en una prueba. |
| 14 | 2026-11-25 | Estimar el costo de cómputo de la búsqueda y de las semillas a partir de una medición de tiempo por época. | Estimación de tiempos (registro, sección 14). | Planeación de las corridas largas. |
| 15 | 2026-11-25 | Analizar los resultados de la búsqueda de hiperparámetros y de las 30 semillas de cada red, generar las figuras comunes de comparación (búsqueda, curvas de aprendizaje, variabilidad entre semillas, matriz de confusión) y comparar las redes con el modelo de referencia. | Script `figuras_redes.py`, tabla `comparacion_modelos.csv` y análisis (registro, sección 15). | Transfer learning es el mejor modelo (F1 0,855 ± 0,021); la CNN desde cero no supera a la referencia en promedio. H2-10. |

## 2. Discusión: impacto de la IA en el Módulo 2

**Dónde aportó más.**
- **Velocidad de implementación:** el flujo completo (inventario, duplicados, partición, preprocesamiento, modelo de referencia y redes) quedó escrito, probado y documentado en una jornada.
- **Análisis de datos:** el hallazgo de mayor consecuencia del módulo, que el 91 % del conjunto proviene de State Farm, surgió de la combinación de varios análisis propuestos por la IA (fechas de los archivos, nombres, revisión visual de pares y orientación del volante) y se confirmó con una fuente primaria (`driver_imgs_list.csv`).
- **Verificación automática:** las comprobaciones que la IA incluyó en el código (verificación de fuga, protección contra grupos degenerados, pruebas de congelación) detectaron varios de sus propios errores.

**Dónde falló.** Al momento de esta versión hay 10 hallazgos documentados (H2-1 a H2-10). El patrón común es dar por cierto algo sin medirlo en estos datos:
- umbrales tomados de la práctica habitual (H2-5, H2-6);
- la documentación de los autores tomada como verdad (H2-2, H2-7);
- una conclusión verificada en un subconjunto extendida a otro (H2-8);
- información técnica desactualizada (H2-3).
Dos de ellos (H2-1 y H2-8) habrían producido fuga de información y métricas infladas sin que el programa fallara.

**Qué verificó la Persona B.**
- Ejecutó todo el código en su máquina y comparó los resultados con los de la IA (la partición coincidió exactamente).
- Tomó las decisiones de diseño con consecuencias (partición, tratamiento de «giro»).
- Revisó las figuras.
- Obtuvo de forma independiente la fuente primaria (Kaggle) que confirmó la procedencia del conjunto.

**Lecciones para el uso de IA:**
- Exigir que cada afirmación sobre los datos venga con una medición sobre los datos.
- Revisar a ojo los casos cercanos a los umbrales, aunque la verificación automática pase.
- No aceptar información de versiones ni de documentación sin consultar la fuente primaria.
