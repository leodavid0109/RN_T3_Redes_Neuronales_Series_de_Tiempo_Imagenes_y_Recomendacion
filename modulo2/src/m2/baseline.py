"""Paso 5: modelo de referencia.

Regresión logística multinomial sobre características extraídas por una red
preentrenada congelada (la misma red que después se ajusta en transfer learning).

1. Se extraen una sola vez las características de la penúltima capa para todas
   las imágenes preprocesadas (sin aumento de datos).
2. Se estandarizan con media y desviación del ENTRENAMIENTO.
3. Búsqueda en rejilla de la regularización C (presupuesto = número de valores),
   eligiendo por F1 macro en VALIDACIÓN. El modelo final se ajusta solo con
   entrenamiento, igual que las redes (que usan validación para la parada temprana).
4. Métricas en validación y prueba, también por procedencia.
5. Variabilidad: el modelo es determinista, así que se reporta un bootstrap del
   conjunto de prueba (para compararlo con las 30 semillas de las redes).

Salidas:
  data/processed/caracteristicas_<backbone>.npz     (no se versiona)
  models/baseline_logreg.joblib
  resultados/baseline_busqueda.csv, baseline_metricas.json,
  resultados/baseline_predicciones.csv, baseline_bootstrap.csv
  figuras/figM2-8_busqueda_baseline.png
"""
from __future__ import annotations

import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torchvision import models
from tqdm import tqdm

from . import metricas
from .datos_torch import cargador, cargar_splits, transformacion_evaluacion
from .inventario import CLASES
from .utils import get_device, resolve, set_seed


def extractor(nombre: str, usar_pesos: bool = True):
    pesos = models.get_model_weights(nombre).DEFAULT if usar_pesos else None
    m = models.get_model(nombre, weights=pesos)
    if hasattr(m, "fc"):
        dim = m.fc.in_features
        m.fc = torch.nn.Identity()
    else:
        ult = [c for c in m.classifier.modules() if isinstance(c, torch.nn.Linear)][0]
        dim = ult.in_features
        m.classifier = torch.nn.Identity()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval(), dim, str(pesos)


@torch.no_grad()
def extraer(cfg: dict, df: pd.DataFrame) -> tuple[np.ndarray, str]:
    bcfg = cfg["baseline"]
    proc = resolve(cfg["datos"]["processed_dir"])
    cache = proc / f"caracteristicas_{bcfg['backbone']}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["ruta_proc"]) == df["ruta_proc"].tolist():
            print(f"Características en caché: {cache.name}")
            return z["X"], str(z["pesos"])
    dev = get_device(cfg.get("dispositivo", "auto"))
    modelo, _, pesos = extractor(bcfg["backbone"], bcfg.get("preentrenado", True))
    modelo = modelo.to(dev)
    dl = cargador(cfg, df, transformacion_evaluacion(cfg["preproceso"]["tamano"]), bcfg["batch_size"])
    X = [modelo(x.to(dev)).float().cpu().numpy() for x, _ in tqdm(dl, desc=f"Características ({bcfg['backbone']}, {dev})")]
    X = np.concatenate(X)
    np.savez_compressed(cache, X=X, ruta_proc=np.array(df["ruta_proc"]), pesos=pesos)
    return X, pesos


def ejecutar(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    bcfg = cfg["baseline"]
    df = cargar_splits(cfg)
    out_res = resolve(cfg["salidas"]["resultados"])
    out_mod = resolve(cfg["salidas"]["modelos"])
    out_fig = resolve(cfg["salidas"]["figuras"])
    for d in (out_res, out_mod, out_fig):
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    X, pesos = extraer(cfg, df)
    t_extraccion = time.time() - t0
    y = df["etiqueta"].to_numpy()
    m = {p: (df["particion"] == p).to_numpy() for p in ["entrenamiento", "validacion", "prueba"]}
    esc = StandardScaler().fit(X[m["entrenamiento"]])
    Xs = esc.transform(X)

    # Búsqueda de C con presupuesto explícito
    filas = []
    t1 = time.time()
    for C in bcfg["valores_C"]:
        ti = time.time()
        clf = LogisticRegression(C=C, max_iter=bcfg["max_iter"], random_state=cfg["seed"])
        clf.fit(Xs[m["entrenamiento"]], y[m["entrenamiento"]])
        pv = clf.predict(Xs[m["validacion"]])
        filas.append(
            dict(
                C=C,
                f1_macro_validacion=f1_score(y[m["validacion"]], pv, average="macro"),
                exactitud_validacion=float((pv == y[m["validacion"]]).mean()),
                exactitud_entrenamiento=float((clf.predict(Xs[m["entrenamiento"]]) == y[m["entrenamiento"]]).mean()),
                tiempo_s=round(time.time() - ti, 2),
            )
        )
        print(f"  C={C:<8g} F1 macro val={filas[-1]['f1_macro_validacion']:.4f}")
    busqueda = pd.DataFrame(filas)
    busqueda.to_csv(out_res / "baseline_busqueda.csv", index=False)
    t_busqueda = time.time() - t1

    mejor_C = float(busqueda.loc[busqueda["f1_macro_validacion"].idxmax(), "C"])
    clf = LogisticRegression(C=mejor_C, max_iter=bcfg["max_iter"], random_state=cfg["seed"]).fit(
        Xs[m["entrenamiento"]], y[m["entrenamiento"]]
    )
    prob = clf.predict_proba(Xs)
    joblib.dump({"escalador": esc, "modelo": clf, "backbone": bcfg["backbone"], "pesos": pesos, "clases": CLASES},
                out_mod / "baseline_logreg.joblib")

    pred_df = df[["ruta", "ruta_proc", "clase", "etiqueta", "procedencia", "particion"]].copy()
    for k, c in enumerate(CLASES):
        pred_df[f"p_{c}"] = prob[:, k].round(6)
    pred_df["prediccion"] = [CLASES[k] for k in prob.argmax(1)]
    pred_df.to_csv(out_res / "baseline_predicciones.csv", index=False)

    boot = metricas.bootstrap(y[m["prueba"]], prob[m["prueba"]], bcfg["bootstrap"], cfg["seed"])
    boot.to_csv(out_res / "baseline_bootstrap.csv", index=False)

    res = {
        "modelo": "Regresión logística multinomial sobre características congeladas",
        "backbone": bcfg["backbone"],
        "pesos": pesos,
        "dimension_caracteristicas": int(X.shape[1]),
        "parametros_entrenables": int(clf.coef_.size + clf.intercept_.size),
        "presupuesto_busqueda": {"configuraciones": len(bcfg["valores_C"]), "criterio": "F1 macro en validación"},
        "mejor_C": mejor_C,
        "tiempo_s": {"extraccion": round(t_extraccion, 1), "busqueda": round(t_busqueda, 1)},
        "validacion": metricas.calcular(y[m["validacion"]], prob[m["validacion"]]),
        "prueba": metricas.calcular(y[m["prueba"]], prob[m["prueba"]]),
        "prueba_por_procedencia": metricas.por_grupo(df[m["prueba"]], prob[m["prueba"]], "procedencia"),
        "prueba_bootstrap": {
            "remuestras": bcfg["bootstrap"],
            "f1_macro": {k: round(float(v), 4) for k, v in boot["f1_macro"].describe()[["mean", "std", "min", "max"]].items()},
            "exactitud": {k: round(float(v), 4) for k, v in boot["exactitud"].describe()[["mean", "std", "min", "max"]].items()},
        },
    }
    with open(out_res / "baseline_metricas.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    # Fig. M2-8: búsqueda de C
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(busqueda["C"], busqueda["f1_macro_validacion"], "o-", label="F1 macro (validación)")
    ax.semilogx(busqueda["C"], busqueda["exactitud_entrenamiento"], "s--", alpha=0.6, label="Exactitud (entrenamiento)")
    ax.axvline(mejor_C, color="red", ls=":", label=f"C elegido = {mejor_C:g}")
    ax.set_xlabel("C (inverso de la regularización L2)")
    ax.set_ylabel("Métrica")
    ax.set_title(f"Modelo de referencia: búsqueda de C ({len(busqueda)} configuraciones)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_fig / "figM2-8_busqueda_baseline.png", dpi=150)
    plt.close(fig)

    resumen = {k: res[k] for k in ("mejor_C", "parametros_entrenables", "tiempo_s", "prueba_por_procedencia", "prueba_bootstrap")}
    resumen["validacion"] = {k: res["validacion"][k] for k in ("exactitud", "f1_macro")}
    resumen["prueba"] = {k: res["prueba"][k] for k in ("exactitud", "f1_macro")}
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    return res
