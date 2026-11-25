"""Paso 1: inventario del conjunto de datos.

Recorre la carpeta cruda, asigna la clase según la carpeta de cada imagen y
registra tamaño, orientación EXIF, hash MD5 del archivo y brillo medio.
Salidas:
  data/processed/inventario.csv
  data/processed/resumen_inventario.json
  figuras/figM2-1_imagenes_por_clase.png
  figuras/figM2-2_ejemplos_por_clase.png
  figuras/figM2-3_brillo_por_clase.png
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .imagenes import EXTENSIONES, abrir_rgb, orientacion_exif
from .utils import resolve

# Orden fijo de clases (índice = etiqueta numérica en todo el módulo)
CLASES = ["conduccion_segura", "giro", "escribir_telefono", "hablar_telefono", "otros"]
NOMBRES = {
    "conduccion_segura": "Conducción segura",
    "giro": "Giro",
    "escribir_telefono": "Escribir en el teléfono",
    "hablar_telefono": "Hablar por teléfono",
    "otros": "Otros riesgosos",
}


def clase_de_carpeta(ruta: Path, raiz: Path, alias: dict[str, str]) -> str | None:
    """Busca, desde la carpeta inmediata hacia arriba, un nombre que contenga un alias."""
    for parte in reversed(ruta.relative_to(raiz).parts[:-1]):
        nombre = parte.lower()
        for clave, clase in alias.items():
            if clave.lower() in nombre:
                return clase
    return None


FIRMAS = [
    (b"\xff\xd8\xff", "JPEG"),
    (b"\x89PNG", "PNG"),
    (b"GIF8", "GIF"),
    (b"BM", "BMP"),
]


def formato_real(ruta: Path) -> tuple[str, str]:
    """Identifica el formato por los primeros bytes (no por la extensión)."""
    with open(ruta, "rb") as f:
        cab = f.read(32)
    if not cab:
        return "vacío", ""
    if cab[:4] == b"RIFF" and cab[8:12] == b"WEBP":
        return "WEBP", cab.hex()
    if cab[4:8] == b"ftyp":
        marca = cab[8:12].decode("ascii", "replace")
        return {"heic": "HEIC", "heix": "HEIC", "mif1": "HEIF", "avif": "AVIF"}.get(marca, f"ftyp:{marca}"), cab.hex()
    for firma, nombre in FIRMAS:
        if cab.startswith(firma):
            return nombre, cab.hex()
    if set(cab) == {0}:
        return "solo ceros", cab.hex()
    return "desconocido", cab.hex()


ORIGENES = [
    # (expresión regular sobre el nombre del archivo, etiqueta). El orden importa.
    (r"^img_\d+\.jpg$", "img_640x480"),
    (r"^IMG_\d+\.JPG$", "foto_IMG_N"),
    (r"^IMG_\d+_\d+.*\.jpg$", "foto_IMG_fecha_HDR"),
    (r"^\d+-\d+-\d+-\d+-\d+\.png$", "png_683x384"),
    (r"^Screenshot_\d+\.png$", "captura_pantalla"),
    (r"^photo_.*\.jpg$", "photo_fecha"),
]


def origen_archivo(nombre: str) -> str:
    """Grupo de origen según el patrón del nombre: en este conjunto los patrones
    coinciden con resoluciones y formatos distintos (ver registro, sección 2)."""
    import re

    for patron, etiqueta in ORIGENES:
        if re.match(patron, nombre):
            return etiqueta
    return "otro"


def md5(ruta: Path) -> str:
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def construir_inventario(cfg: dict) -> pd.DataFrame:
    raiz = resolve(cfg["datos"]["raw_dir"])
    if not raiz.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de datos crudos: {raiz}\n"
            "Descarga el conjunto y ajusta datos.raw_dir en configs/m2.yaml "
            "o pásalo con --set datos.raw_dir=RUTA"
        )
    alias = cfg["datos"]["alias_clases"]
    archivos = sorted(p for p in raiz.rglob("*") if p.suffix.lower() in EXTENSIONES)
    if not archivos:
        raise FileNotFoundError(f"No hay imágenes en {raiz}")

    filas, sin_clase, ilegibles = [], [], []
    for p in tqdm(archivos, desc="Inventario"):
        clase = clase_de_carpeta(p, raiz, alias)
        if clase is None:
            sin_clase.append(str(p.relative_to(raiz)))
            continue
        try:
            with Image.open(p) as im:
                ancho_arch, alto_arch = im.size
            im = abrir_rgb(p)  # fuerza la decodificación completa (detecta archivos truncados)
        except Exception as e:
            fmt, cab = formato_real(p)
            ilegibles.append(
                dict(ruta=p.relative_to(raiz).as_posix(), clase=clase, bytes=p.stat().st_size,
                     formato_real=fmt, cabecera_hex=cab, error=f"{type(e).__name__}: {e}")
            )
            continue
        ancho, alto = im.size
        brillo = float(np.asarray(im.convert("L").resize((64, 64))).mean())
        filas.append(
            dict(
                ruta=p.relative_to(raiz).as_posix(),
                clase=clase,
                etiqueta=CLASES.index(clase),
                origen=origen_archivo(p.name),
                ancho=ancho,
                alto=alto,
                ancho_archivo=ancho_arch,
                alto_archivo=alto_arch,
                orientacion_exif=orientacion_exif(p),
                bytes=p.stat().st_size,
                md5=md5(p),
                brillo=brillo,
            )
        )
    if sin_clase:
        print(f"AVISO: {len(sin_clase)} imágenes sin clase reconocida (ignoradas). Ejemplos: {sin_clase[:5]}")
    ilegibles = pd.DataFrame(ilegibles, columns=["ruta", "clase", "bytes", "formato_real", "cabecera_hex", "error"])
    if len(ilegibles):
        print(f"AVISO: {len(ilegibles)} archivos no se pudieron leer (excluidos). Ver archivos_ilegibles.csv")
        print(ilegibles.groupby(["clase", "formato_real"]).size().to_string())
    return pd.DataFrame(filas), sin_clase, ilegibles


def resumen(df: pd.DataFrame, sin_clase: list[str], ilegibles: pd.DataFrame) -> dict:
    conteo = df["clase"].value_counts().reindex(CLASES).fillna(0).astype(int)
    res = df.groupby(["ancho", "alto"]).size().sort_values(ascending=False)
    grupos_md5 = df.groupby("md5").size()
    return {
        "total_imagenes": int(len(df)),
        "imagenes_por_clase": conteo.to_dict(),
        "proporcion_por_clase": (conteo / conteo.sum()).round(4).to_dict(),
        "razon_desbalance_max_min": round(float(conteo.max() / max(conteo.min(), 1)), 3),
        "resoluciones": {f"{a}x{b}": int(n) for (a, b), n in res.items()},
        "imagenes_con_rotacion_exif": int((df["orientacion_exif"] > 1).sum()),
        "origen_por_clase": pd.crosstab(df["origen"], df["clase"]).reindex(columns=CLASES, fill_value=0).to_dict(orient="index"),
        "duplicados_exactos_md5": {
            "archivos_en_grupos_repetidos": int(grupos_md5[grupos_md5 > 1].sum()),
            "grupos_repetidos": int((grupos_md5 > 1).sum()),
        },
        "imagenes_sin_clase": len(sin_clase),
        "archivos_ilegibles": {
            "total": int(len(ilegibles)),
            "por_clase": ilegibles["clase"].value_counts().to_dict(),
            "por_formato_real": ilegibles["formato_real"].value_counts().to_dict(),
        },
    }


def figuras(df: pd.DataFrame, cfg: dict) -> None:
    out = resolve(cfg["salidas"]["figuras"])
    out.mkdir(parents=True, exist_ok=True)
    raiz = resolve(cfg["datos"]["raw_dir"])
    rng = np.random.default_rng(cfg["seed"])
    etiquetas = [NOMBRES[c] for c in CLASES]

    # Fig. M2-1: imágenes por clase
    conteo = df["clase"].value_counts().reindex(CLASES)
    fig, ax = plt.subplots(figsize=(8, 4))
    barras = ax.bar(etiquetas, conteo.values, color="#4C72B0")
    ax.bar_label(barras)
    ax.set_ylabel("Número de imágenes")
    ax.set_title("Imágenes por clase")
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(out / "figM2-1_imagenes_por_clase.png", dpi=150)
    plt.close(fig)

    # Fig. M2-2: ejemplos por clase (4 aleatorios con la semilla global)
    n = 4
    fig, axes = plt.subplots(len(CLASES), n, figsize=(2.4 * n, 2.6 * len(CLASES)))
    for i, c in enumerate(CLASES):
        sub = df[df["clase"] == c]
        idx = rng.choice(len(sub), size=min(n, len(sub)), replace=False)
        for j in range(n):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(idx):
                ax.imshow(abrir_rgb(raiz / sub.iloc[idx[j]]["ruta"]))
        axes[i, 0].set_title(NOMBRES[c], loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "figM2-2_ejemplos_por_clase.png", dpi=120)
    plt.close(fig)

    # Fig. M2-3: brillo medio por clase (variabilidad de iluminación)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([df.loc[df["clase"] == c, "brillo"] for c in CLASES], tick_labels=etiquetas)
    ax.set_ylabel("Brillo medio (escala de grises, 0–255)")
    ax.set_title("Iluminación por clase")
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(out / "figM2-3_brillo_por_clase.png", dpi=150)
    plt.close(fig)


def ejecutar(cfg: dict) -> dict:
    df, sin_clase, ilegibles = construir_inventario(cfg)
    proc = resolve(cfg["datos"]["processed_dir"])
    proc.mkdir(parents=True, exist_ok=True)
    df.to_csv(proc / "inventario.csv", index=False)
    ilegibles.to_csv(proc / "archivos_ilegibles.csv", index=False)
    res = resumen(df, sin_clase, ilegibles)
    with open(proc / "resumen_inventario.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    figuras(df, cfg)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return res
