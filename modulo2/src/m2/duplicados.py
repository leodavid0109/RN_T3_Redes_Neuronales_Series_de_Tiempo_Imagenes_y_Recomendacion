"""Paso 2: similitud entre imágenes para detectar casi duplicados.

Criterio: coseno CENTRADO entre embeddings de una red preentrenada (ImageNet),
tomando el máximo entre la imagen y su volteo horizontal en ambas direcciones.

Por qué centrado: las características de ResNet salen de un average pooling
después de una ReLU (todas >= 0), y entre vectores no negativos el coseno es alto
aunque las imágenes no tengan relación (mediana 0,85 en pares aleatorios de este
conjunto). Al restar el vector promedio antes de normalizar, dos imágenes no
relacionadas quedan cerca de 0. Ver registro, sección 7, y hallazgo H2-5.

El pHash (64 bits) se calcula solo como dato descriptivo: en este conjunto no
discrimina (registro 7.2, hallazgo H2-6).

Salidas (data/processed/duplicados/):
  caracteristicas.npz         embeddings centrados (imagen y volteo), pHash, vector medio
  pares_candidatos.csv.gz     pares con coseno centrado >= registro_coseno
  vecino_mas_cercano.csv      similitud con el vecino más cercano de cada imagen
  resumen_duplicados.json     distribución nula (pares aleatorios) y conteos
Figuras:
  figM2-4_distribucion_similitud.png
  figM2-5_calibracion_duplicados.png
"""
from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import imagehash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from .imagenes import abrir_rgb
from .utils import get_device, resolve, set_seed


class _ImgDS(Dataset):
    def __init__(self, raiz, rutas, size):
        self.raiz, self.rutas = raiz, rutas
        self.tf = transforms.Compose(
            [
                # Se redimensiona sin recortar para que el embedding vea la foto completa
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, i):
        im = abrir_rgb(self.raiz / self.rutas[i])
        ph = imagehash.phash(im).hash.flatten()
        ph_flip = imagehash.phash(ImageOps.mirror(im)).hash.flatten()
        return self.tf(im), torch.from_numpy(ph.astype(np.uint8)), torch.from_numpy(ph_flip.astype(np.uint8))


def _backbone(nombre: str, usar_pesos: bool = True):
    # usar_pesos=False solo sirve para probar el código sin internet: sin pesos
    # preentrenados los embeddings no detectan duplicados de forma útil.
    pesos = models.get_model_weights(nombre).DEFAULT if usar_pesos else None
    m = models.get_model(nombre, weights=pesos)
    if hasattr(m, "fc"):
        m.fc = torch.nn.Identity()
    elif hasattr(m, "classifier"):
        m.classifier = torch.nn.Identity()
    else:
        raise ValueError(f"No sé quitar la cabeza de {nombre}")
    return m.eval(), str(pesos)


def _normalizar(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


@torch.no_grad()
def calcular_caracteristicas(cfg: dict, inv: pd.DataFrame) -> dict:
    dcfg = cfg["duplicados"]
    dev = get_device(cfg.get("dispositivo", "auto"))
    modelo, pesos = _backbone(dcfg["backbone"], dcfg.get("preentrenado", True))
    modelo = modelo.to(dev)
    ds = _ImgDS(resolve(cfg["datos"]["raw_dir"]), inv["ruta"].tolist(), dcfg["img_size"])
    dl = DataLoader(ds, batch_size=dcfg["batch_size"], num_workers=cfg.get("num_workers", 0), shuffle=False)
    E, F, H, HF = [], [], [], []
    for x, h, hf in tqdm(dl, desc=f"Embeddings ({dcfg['backbone']}, {dev})"):
        x = x.to(dev)
        E.append(modelo(x).float().cpu())
        F.append(modelo(torch.flip(x, dims=[3])).float().cpu())
        H.append(h)
        HF.append(hf)
    E = _normalizar(torch.cat(E).numpy())
    F = _normalizar(torch.cat(F).numpy())
    # Centrado: se resta el promedio de todos los vectores (originales y volteados)
    media = np.concatenate([E, F]).mean(0, keepdims=True)
    return dict(
        emb=_normalizar(E - media).astype(np.float32),
        emb_flip=_normalizar(F - media).astype(np.float32),
        media=media.astype(np.float32),
        min_caracteristica=np.float32(min(E.min(), F.min())),
        phash=torch.cat(H).numpy().astype(np.uint8),
        phash_flip=torch.cat(HF).numpy().astype(np.uint8),
        pesos=pesos,
    )


def similitudes_bloque(car: dict, filas: np.ndarray, columnas: np.ndarray | None = None):
    """Coseno centrado máximo y Hamming mínimo entre `filas` y `columnas` (todas si es None).

    Se compara en ambas direcciones (A vs. volteo de B y volteo de A vs. B):
    ni el pHash ni el embedding de un volteo son simétricos, y sin esto la
    similitud de (i, j) no coincide con la de (j, i) (hallazgo H2-1).
    """
    E, F = car["emb"], car["emb_flip"]
    P = car["phash"].astype(np.float32)
    PF = car["phash_flip"].astype(np.float32)
    cols = np.arange(len(E)) if columnas is None else columnas
    a, af = E[filas], F[filas]
    cos = np.maximum.reduce([a @ E[cols].T, a @ F[cols].T, af @ E[cols].T])

    def ham(pa, pb):
        # Hamming = bits distintos = n - (coincidencias de unos + coincidencias de ceros)
        return pa.shape[1] - (pa @ pb.T + (1 - pa) @ (1 - pb).T)

    pa, paf = P[filas], PF[filas]
    hm = np.minimum.reduce([ham(pa, P[cols]), ham(pa, PF[cols]), ham(paf, P[cols])]).round().astype(np.int16)
    return cos, hm


def distribucion_nula(car: dict, n_pares: int, seed: int) -> dict:
    """Similitud entre pares de imágenes elegidos al azar (referencia para el umbral)."""
    rng = np.random.default_rng(seed)
    n = len(car["emb"])
    i, j = rng.integers(0, n, n_pares), rng.integers(0, n, n_pares)
    m = i != j
    i, j = i[m], j[m]
    E, F = car["emb"], car["emb_flip"]
    cos = np.maximum.reduce([(E[i] * E[j]).sum(1), (E[i] * F[j]).sum(1), (F[i] * E[j]).sum(1)])
    P, PF = car["phash"], car["phash_flip"]
    hm = np.minimum.reduce([(P[i] != P[j]).sum(1), (P[i] != PF[j]).sum(1), (PF[i] != P[j]).sum(1)])
    q = [1, 5, 50, 95, 99, 99.9]
    return {
        "n_pares": int(m.sum()),
        "coseno_centrado_percentiles": {str(k): round(float(v), 4) for k, v in zip(q, np.percentile(cos, q))},
        "hamming_percentiles": {str(k): float(v) for k, v in zip([0.1, 1, 5, 50], np.percentile(hm, [0.1, 1, 5, 50]))},
        "_cos": cos,
    }


def pares_candidatos(car: dict, md5: np.ndarray, umbral_registro: float, bloque: int = 512):
    n = len(car["emb"])
    pares, nn_cos, nn_ham = [], np.full(n, -1.0, np.float32), np.full(n, 99, np.int16)
    for ini in tqdm(range(0, n, bloque), desc="Pares"):
        filas = np.arange(ini, min(ini + bloque, n))
        cos, hm = similitudes_bloque(car, filas)
        cos[np.arange(len(filas)), filas] = -9  # excluir la propia imagen
        hm[np.arange(len(filas)), filas] = 99
        nn_cos[filas] = cos.max(1)
        nn_ham[filas] = hm.min(1)
        cand = (cos >= umbral_registro) & (np.arange(n)[None, :] > filas[:, None])  # i < j
        ii, jj = np.nonzero(cand)
        pares.append(
            pd.DataFrame(
                dict(i=filas[ii], j=jj, coseno=cos[ii, jj].round(4), hamming=hm[ii, jj], md5_igual=md5[filas[ii]] == md5[jj])
            )
        )
    return pd.concat(pares, ignore_index=True), nn_cos, nn_ham


def figuras(cfg, inv, pares, nn_cos, nula_cos):
    out = resolve(cfg["salidas"]["figuras"])
    out.mkdir(parents=True, exist_ok=True)
    umbral = cfg["particion"]["umbral_coseno"]
    raiz = resolve(cfg["datos"]["raw_dir"])

    # Fig. M2-4: vecino más cercano frente a pares aleatorios
    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.linspace(-0.4, 1.0, 120)
    ax.hist(nula_cos, bins=bins, density=True, alpha=0.5, label="Pares aleatorios (distribución nula)")
    ax.hist(nn_cos, bins=bins, density=True, alpha=0.5, label="Vecino más cercano de cada imagen")
    ax.axvline(umbral, color="red", ls="--", label=f"umbral de agrupación = {umbral}")
    ax.set_xlabel("Coseno centrado (máximo entre imagen y volteo)")
    ax.set_ylabel("Densidad")
    ax.set_title("Similitud entre imágenes: vecino más cercano frente a pares aleatorios")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figM2-4_distribucion_similitud.png", dpi=150)
    plt.close(fig)

    # Fig. M2-5: pares de ejemplo por franja de coseno, para calibrar el umbral a ojo
    rng = np.random.default_rng(cfg["seed"])
    franjas = cfg["duplicados"]["franjas_calibracion"]
    n_ej = 3
    fig, axes = plt.subplots(len(franjas), 2 * n_ej, figsize=(2.2 * 2 * n_ej, 2.4 * len(franjas)))
    axes = np.atleast_2d(axes)
    for r, (lo, hi) in enumerate(franjas):
        sub = pares[(pares["coseno"] >= lo) & (pares["coseno"] < hi)]
        idx = rng.choice(len(sub), size=min(n_ej, len(sub)), replace=False) if len(sub) else []
        for k in range(n_ej):
            for s in range(2):
                a = axes[r, 2 * k + s]
                a.axis("off")
                if k < len(idx):
                    fila = sub.iloc[idx[k]]
                    i = int(fila["i"] if s == 0 else fila["j"])
                    a.imshow(abrir_rgb(raiz / inv.iloc[i]["ruta"]))
                    a.set_title(inv.iloc[i]["clase"][:12] + (f"\ncos={fila['coseno']:.3f}" if s == 0 else "\n"), fontsize=7)
        axes[r, 0].text(-0.1, 0.5, f"[{lo}, {hi})\nn={len(sub)}", transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=9)
    fig.suptitle("Pares de imágenes por franja de coseno centrado (cada par = 2 imágenes contiguas)")
    fig.tight_layout()
    fig.savefig(out / "figM2-5_calibracion_duplicados.png", dpi=110)
    plt.close(fig)


def ejecutar(cfg: dict) -> None:
    set_seed(cfg["seed"])
    dcfg = cfg["duplicados"]
    proc = resolve(cfg["datos"]["processed_dir"])
    inv = pd.read_csv(proc / "inventario.csv")
    dest = proc / "duplicados"
    dest.mkdir(parents=True, exist_ok=True)
    viejo = dest / "pares_candidatos.csv"
    if viejo.exists():
        print(f"AVISO: existe {viejo} de una versión anterior (sin centrar, ~900 MB). Bórralo; ya no se usa.")

    t0 = time.time()
    car = calcular_caracteristicas(cfg, inv)
    np.savez_compressed(dest / "caracteristicas.npz", **{k: v for k, v in car.items() if k != "pesos"})
    nula = distribucion_nula(car, dcfg["muestra_nula"], cfg["seed"])
    pares, nn_cos, nn_ham = pares_candidatos(car, inv["md5"].to_numpy(), dcfg["registro_coseno"])
    pares["clase_i"] = inv["clase"].to_numpy()[pares["i"]]
    pares["clase_j"] = inv["clase"].to_numpy()[pares["j"]]
    pares.to_csv(dest / "pares_candidatos.csv.gz", index=False, compression="gzip")
    pd.DataFrame({"ruta": inv["ruta"], "clase": inv["clase"], "nn_coseno": nn_cos, "nn_hamming": nn_ham}).to_csv(
        dest / "vecino_mas_cercano.csv", index=False
    )
    figuras(cfg, inv, pares, nn_cos, nula.pop("_cos"))

    q = [5, 50, 95]
    resumen = {
        "backbone": dcfg["backbone"],
        "pesos": car["pesos"],
        "minimo_caracteristica_sin_centrar": float(car["min_caracteristica"]),
        "distribucion_nula": nula,
        "vecino_mas_cercano_percentiles": {str(k): round(float(v), 4) for k, v in zip(q, np.percentile(nn_cos, q))},
        "pares_por_franja": {
            f"[{lo}, {hi})": int(((pares.coseno >= lo) & (pares.coseno < hi)).sum()) for lo, hi in dcfg["franjas_calibracion"]
        },
        "pares_registrados": int(len(pares)),
        "tiempo_s": round(time.time() - t0, 1),
    }
    with open(dest / "resumen_duplicados.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
