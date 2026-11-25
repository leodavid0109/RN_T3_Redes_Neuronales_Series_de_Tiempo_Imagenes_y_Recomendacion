"""Paso 4: preprocesamiento único para todos los modelos.

Lleva cada imagen (con la orientación EXIF aplicada) a un tamaño fijo y la guarda
como JPEG en data/processed/imagenes/, con la misma estructura de carpetas que el
conjunto original. Todos los modelos (referencia, CNN desde cero y transfer
learning) leen estas imágenes: mismo preprocesamiento y mismas particiones.

Geometría (preproceso.geometria):
  estirar   redimensiona a tamano x tamano sin conservar la proporción (no pierde
            contenido; deforma distinto según el origen).          <- por defecto
  recortar  lado corto a `tamano` y recorte central cuadrado (pierde bordes).
  rellenar  lado largo a `tamano` y relleno gris hasta el cuadrado (franjas
            visibles cuya forma delata la proporción original).
Ver registro, sección 11.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

from .imagenes import abrir_rgb
from .utils import resolve


def ruta_procesada(ruta: str) -> str:
    return str(Path(ruta).with_suffix(".jpg").as_posix())


def transformar(im: Image.Image, tamano: int, geometria: str) -> Image.Image:
    if geometria == "estirar":
        return im.resize((tamano, tamano), Image.BICUBIC)
    if geometria == "recortar":
        return ImageOps.fit(im, (tamano, tamano), Image.BICUBIC, centering=(0.5, 0.5))
    if geometria == "rellenar":
        return ImageOps.pad(im, (tamano, tamano), Image.BICUBIC, color=(128, 128, 128))
    raise ValueError(f"geometria desconocida: {geometria}")


def ejecutar(cfg: dict) -> dict:
    pcfg = cfg["preproceso"]
    proc = resolve(cfg["datos"]["processed_dir"])
    raiz = resolve(cfg["datos"]["raw_dir"])
    splits = pd.read_csv(proc / "splits.csv")
    destino = proc / pcfg["carpeta"]
    rutas = splits["ruta"].map(ruta_procesada)
    if rutas.duplicated().any():
        raise ValueError(f"Nombres repetidos al cambiar la extensión: {rutas[rutas.duplicated()].tolist()[:5]}")

    t0 = time.time()
    total_bytes = 0
    for ruta, rp in tqdm(list(zip(splits["ruta"], rutas)), desc=f"Preproceso ({pcfg['geometria']}, {pcfg['tamano']} px)"):
        salida = destino / rp
        salida.parent.mkdir(parents=True, exist_ok=True)
        im = transformar(abrir_rgb(raiz / ruta), pcfg["tamano"], pcfg["geometria"])
        im.save(salida, "JPEG", quality=pcfg["calidad_jpeg"], optimize=True)
        total_bytes += salida.stat().st_size

    splits["ruta_proc"] = rutas
    splits.to_csv(proc / "splits.csv", index=False)
    res = {
        "imagenes": int(len(splits)),
        "tamano": pcfg["tamano"],
        "geometria": pcfg["geometria"],
        "calidad_jpeg": pcfg["calidad_jpeg"],
        "megabytes_totales": round(total_bytes / 1e6, 1),
        "tiempo_s": round(time.time() - t0, 1),
    }
    with open(proc / "resumen_preproceso.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return res
