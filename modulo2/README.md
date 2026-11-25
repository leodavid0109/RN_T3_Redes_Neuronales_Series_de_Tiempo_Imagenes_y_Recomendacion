# Módulo 2 — Clasificación de conducción distractiva

Responsable: Persona B. Registro detallado de decisiones: [`docs/registro/registro_modulo2.md`](../docs/registro/registro_modulo2.md).

## Datos

Multi-Class Driver Behavior Image Dataset (Afridi et al., 2024), Mendeley Data, DOI [10.17632/mzb4b6dff3.1](https://doi.org/10.17632/mzb4b6dff3.1). Artículo: Data in Brief, DOI [10.1016/j.dib.2025.111529](https://doi.org/10.1016/j.dib.2025.111529).

1. Descargar el .zip desde Mendeley Data y descomprimirlo **fuera de OneDrive** (p. ej. `C:/datos/m2_raw`).
2. Indicar la ruta en `configs/m2.yaml` (`datos.raw_dir`) o con `--set datos.raw_dir=C:/datos/m2_raw`.

Las clases se reconocen por el nombre de la carpeta (`datos.alias_clases`).

3. **Procedencia:** el 91 % de las imágenes proviene del conjunto *State Farm Distracted Driver Detection* (registro, sección 9). La partición por conductor necesita `driver_imgs_list.csv` de esa competencia de Kaggle (pestaña Data; hay que aceptar las reglas). Colocarlo en `C:/datos/driver_imgs_list.csv` o indicar la ruta en `datos.state_farm_lista`. No se incluye en el repositorio.

## Entorno

```bash
# 1) PyTorch con CUDA >= 12.8 desde https://pytorch.org/get-started/locally/
# 2) Resto de dependencias
pip install -r modulo2/requirements.txt
```

## Ejecución

Desde `modulo2/` (en Windows, si `python` abre la Microsoft Store, usar `py` en su lugar):

```bash
python run.py inventario   # conteos, resoluciones, brillo, ejemplos  -> figM2-1..3
python run.py duplicados   # pHash + embeddings, pares candidatos     -> figM2-4..5
python run.py particion    # grupos, partición sin fuga y verificación -> figM2-6..7, splits.csv
python run.py datos        # los tres en orden
```

Cualquier parámetro se cambia con `--set`, p. ej. `python run.py particion --set particion.umbral_coseno=0.93`.
La configuración efectiva y las versiones quedan en `data/processed/config_efectiva_*.json`.

## Salidas versionadas

| Archivo | Contenido |
|---|---|
| `data/processed/inventario.csv` | una fila por imagen: clase, tamaño, EXIF, MD5, brillo |
| `data/processed/splits.csv` | partición fija usada por **todos** los modelos |
| `data/processed/verificacion_fuga.json` | prueba de que no hay duplicados entre particiones |
| `figuras/figM2-*.png` | figuras del reporte (pies en `figuras/captions.md`) |
