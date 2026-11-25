"""Métricas de clasificación comunes a todos los modelos del Módulo 2."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .inventario import CLASES


def calcular(y: np.ndarray, prob: np.ndarray) -> dict:
    pred = prob.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(len(CLASES)), zero_division=0)
    return {
        "n": int(len(y)),
        "exactitud": round(float(accuracy_score(y, pred)), 4),
        "f1_macro": round(float(f1_score(y, pred, average="macro", labels=range(len(CLASES)), zero_division=0)), 4),
        "por_clase": {
            c: {"precision": round(float(p[k]), 4), "sensibilidad": round(float(r[k]), 4), "f1": round(float(f[k]), 4), "n": int(s[k])}
            for k, c in enumerate(CLASES)
        },
        "matriz_confusion": confusion_matrix(y, pred, labels=range(len(CLASES))).tolist(),
    }


def por_grupo(df: pd.DataFrame, prob: np.ndarray, columna: str) -> dict:
    """Exactitud y F1 macro dentro de cada valor de `columna` (p. ej. procedencia)."""
    res = {}
    y = df["etiqueta"].to_numpy()
    for valor in sorted(df[columna].unique()):
        m = (df[columna] == valor).to_numpy()
        res[str(valor)] = {k: v for k, v in calcular(y[m], prob[m]).items() if k in ("n", "exactitud", "f1_macro")}
    return res


def bootstrap(y: np.ndarray, prob: np.ndarray, n: int, seed: int) -> pd.DataFrame:
    """Remuestreo del conjunto de prueba: variabilidad de un modelo determinista."""
    rng = np.random.default_rng(seed)
    pred = prob.argmax(1)
    filas = []
    for b in range(n):
        i = rng.integers(0, len(y), len(y))
        filas.append(
            dict(
                remuestra=b,
                exactitud=accuracy_score(y[i], pred[i]),
                f1_macro=f1_score(y[i], pred[i], average="macro", labels=range(len(CLASES)), zero_division=0),
            )
        )
    return pd.DataFrame(filas)
