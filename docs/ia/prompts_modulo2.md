# Registro de prompts — Módulo 2

Prompts principales usados con herramientas de IA en el Módulo 2, copiados literalmente, con un resumen de la respuesta y de su efecto en el trabajo. Herramienta: Claude (Anthropic), en la aplicación de escritorio con acceso a la carpeta del repositorio.

| # | Fecha | Prompt (literal) | Qué produjo la IA | Qué se usó / qué se corrigió |
|---|---|---|---|---|
| 1 | 2026-11-25 | «la propuesta de distribucion fue aprobada, me toco ser la persona b entonces quiero indicaciones para que me ayudes a realizar mi parte» | Guía de trabajo del Módulo 2 y de la matemática común (documento en el proyecto) | Se usó como hoja de ruta |
| 2 | 2026-11-25 | «seria en gpu propia debido a la reproducibilidad, puedo conectarte si quieres con la carpeta del github tambien» (más la salida de `nvidia-smi`) | Código de inventario, detección de duplicados y partición (`modulo2/src/m2/`) | Al probarlo con datos sintéticos, la verificación de fuga detectó un error de simetría en la comparación con volteos; corregido (registro, sección 4, hallazgo 1) |
| 3 | 2026-11-25 | «ve realizando un documento de registro de modulo 2 que sera anexado tal cual a la documentacion general lo vas editando a medida que vamos trabajando» | Este registro y `docs/registro/registro_modulo2.md` | — |
| 4 | 2026-11-25 | «dame el comando para que instale pytorch» | Comando con el índice `cu128`, tras consultar pytorch.org | La consulta web devolvió una versión desactualizada (hallazgo H2-3) |
| 5 | 2026-11-25 | Error de `UnidentifiedImageError` pegado en el chat | Inventario tolerante a archivos ilegibles, con identificación del formato real | El código original no lo contemplaba (H2-2) |
| 6 | 2026-11-25 | Resumen JSON del inventario pegado en el chat | Análisis de orígenes, numeración y riesgo de atajo; `tools/verificar_zip.py` | — |
| 7 | 2026-11-25 | «cuando realices procesos cuadno te equivoques tomalos en cuenta para los hallazgos de alucinacion y documentalo» | Este archivo de hallazgos y la política de errores del registro | Al revisar se encontró H2-4 |
