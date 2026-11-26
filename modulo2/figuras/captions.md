# Pies de figura — Módulo 2

Todas las figuras se regeneran con `py run.py datos` desde `modulo2/`. Fuente de los datos: Afridi et al. (2024); el 91 % de las imágenes proviene del conjunto State Farm Distracted Driver Detection (registro, sección 9).

| Figura | Archivo | Título | Fuente |
|---|---|---|---|
| Fig. M2-1 | `figM2-1_imagenes_por_clase.png` | Número de imágenes por clase | Elaboración propia con datos de Afridi et al. (2024) |
| Fig. M2-2 | `figM2-2_ejemplos_por_clase.png` | Ejemplos aleatorios de cada clase (semilla 42) | Imágenes de Afridi et al. (2024) |
| Fig. M2-3 | `figM2-3_brillo_por_clase.png` | Brillo medio de las imágenes por clase | Elaboración propia con datos de Afridi et al. (2024) |
| Fig. M2-4 | `figM2-4_distribucion_similitud.png` | Coseno centrado con el vecino más cercano frente a pares aleatorios (distribución nula) | Elaboración propia |
| Fig. M2-5 | `figM2-5_calibracion_duplicados.png` | Pares de imágenes por franja de coseno centrado, usados para calibrar el umbral | Elaboración propia con imágenes de Afridi et al. (2024) |
| Fig. M2-6 | `figM2-6_particion_por_clase.png` | Imágenes por clase, partición y procedencia | Elaboración propia |
| Fig. M2-8 | `figM2-8_busqueda_baseline.png` | Modelo de referencia: F1 macro en validación según la regularización C | Elaboración propia |
| Fig. M2-9 | `figM2-9_busqueda_hiperparametros.png` | Búsqueda aleatoria de hiperparámetros: F1 macro en validación de las 16 configuraciones de cada red | Elaboración propia |
| Fig. M2-10 | `figM2-10_curvas_aprendizaje.png` | Curvas de aprendizaje de las redes finales (semilla 0), con la mejor época y la parada temprana | Elaboración propia |
| Fig. M2-11 | `figM2-11_comparacion_semillas.png` | F1 macro en prueba: 30 semillas por red y bootstrap del modelo de referencia; desempeño por procedencia | Elaboración propia |
| Fig. M2-12 | `figM2-12_matriz_confusion.png` | Matriz de confusión en prueba del mejor modelo (transfer learning, semilla 0) | Elaboración propia |
| Fig. M2-13 | `figM2-13_gradcam_aciertos.png` | Diez imágenes de prueba bien clasificadas por el modelo final y su mapa Grad-CAM | Elaboración propia con imágenes de Afridi et al. (2024); método de Selvaraju et al. (2017) |
| Fig. M2-14 | `figM2-14_gradcam_errores.png` | Diez imágenes de prueba mal clasificadas por el modelo final y su mapa Grad-CAM | Elaboración propia con imágenes de Afridi et al. (2024); método de Selvaraju et al. (2017) |
| Fig. M2-15 | `figM2-15_gradcam_promedio.png` | Grad-CAM promedio por clase sobre los aciertos de State Farm en prueba | Elaboración propia; método de Selvaraju et al. (2017) |
| Fig. M2-16 | `figM2-16_alertas_umbral.png` | Falsas alarmas y distracciones no detectadas por cada 1.000 imágenes según el umbral de alerta | Elaboración propia |
| Fig. M2-17 | `figM2-17_exactitud_por_grupo.png` | Exactitud en prueba por conductor o sesión, para cada modelo | Elaboración propia |
| Fig. M2-18 | `figM2-18_arquitecturas.png` | Arquitectura de las redes finales: capas, dimensiones y parámetros | Elaboración propia |
| Fig. M2-7 | `figM2-7_verificacion_fuga.png` | Coseno centrado máximo entre particiones frente al umbral de agrupación | Elaboración propia |
