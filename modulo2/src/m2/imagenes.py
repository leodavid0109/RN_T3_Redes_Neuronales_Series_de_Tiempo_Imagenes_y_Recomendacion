"""Lectura de imágenes con la orientación EXIF aplicada.

Las fotos de celular suelen guardar la rotación en la etiqueta EXIF y no en los
píxeles. PIL no la aplica por defecto, así que una misma foto puede verse
acostada o de pie según quién la abra. Todo el módulo lee las imágenes con esta
función para que el inventario, los duplicados y el entrenamiento vean lo mismo.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def abrir_rgb(ruta: str | Path) -> Image.Image:
    with Image.open(ruta) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def orientacion_exif(ruta: str | Path) -> int:
    """Valor de la etiqueta EXIF Orientation (1 = sin rotación; 0 = no existe)."""
    try:
        with Image.open(ruta) as im:
            return int(im.getexif().get(0x0112, 0))
    except Exception:
        return 0
