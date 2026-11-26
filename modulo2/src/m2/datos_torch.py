"""Conjunto de datos de PyTorch sobre las imágenes preprocesadas y la partición fija."""
from __future__ import annotations

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .utils import resolve

MEDIA_IMAGENET = [0.485, 0.456, 0.406]
DESV_IMAGENET = [0.229, 0.224, 0.225]


class ImagenesM2(Dataset):
    def __init__(self, df: pd.DataFrame, raiz, tf):
        self.rutas = df["ruta_proc"].tolist()
        self.y = df["etiqueta"].to_numpy()
        self.raiz, self.tf = raiz, tf

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, i):
        with Image.open(self.raiz / self.rutas[i]) as im:
            x = self.tf(im.convert("RGB"))
        return x, int(self.y[i])


def transformacion_evaluacion(tamano: int = 224):
    return transforms.Compose(
        [transforms.Resize((tamano, tamano)), transforms.ToTensor(), transforms.Normalize(MEDIA_IMAGENET, DESV_IMAGENET)]
    )


def transformacion_entrenamiento(aumento: dict, tamano: int = 224):
    """Aumento de datos aplicado solo al entrenamiento (en línea, en cada época)."""
    pasos = []
    if aumento.get("recorte_escala_min", 1.0) < 1.0:
        pasos.append(transforms.RandomResizedCrop(tamano, scale=(aumento["recorte_escala_min"], 1.0), ratio=(0.9, 1.1)))
    else:
        pasos.append(transforms.Resize((tamano, tamano)))
    if aumento.get("rotacion", 0) > 0:
        pasos.append(transforms.RandomRotation(aumento["rotacion"]))
    if aumento.get("brillo_contraste", 0) > 0:
        b = aumento["brillo_contraste"]
        pasos.append(transforms.ColorJitter(brightness=b, contrast=b, saturation=b / 2))
    if aumento.get("volteo_h", 0) > 0:
        pasos.append(transforms.RandomHorizontalFlip(aumento["volteo_h"]))
    pasos += [transforms.ToTensor(), transforms.Normalize(MEDIA_IMAGENET, DESV_IMAGENET)]
    return transforms.Compose(pasos)


def cargar_splits(cfg: dict) -> pd.DataFrame:
    proc = resolve(cfg["datos"]["processed_dir"])
    df = pd.read_csv(proc / "splits.csv")
    if "ruta_proc" not in df.columns:
        raise FileNotFoundError("splits.csv no tiene 'ruta_proc': corre primero 'py run.py preproceso'.")
    return df


def cargador(cfg: dict, df: pd.DataFrame, tf, batch_size: int, barajar: bool = False, generador=None) -> DataLoader:
    raiz = resolve(cfg["datos"]["processed_dir"]) / cfg["preproceso"]["carpeta"]
    return DataLoader(
        ImagenesM2(df.reset_index(drop=True), raiz, tf),
        batch_size=batch_size,
        shuffle=barajar,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.get("num_workers", 0) > 0,
        generator=generador,
    )
