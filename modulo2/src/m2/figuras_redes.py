"""Figuras de comparación entre modelos (se regeneran desde resultados/, sin reentrenar).

  py run.py figuras

  figM2-9_busqueda_hiperparametros.png   F1 macro en validación de cada configuración probada
  figM2-10_curvas_aprendizaje.png        pérdida y F1 por época de las redes finales (semilla 0)
  figM2-11_comparacion_semillas.png      F1 macro en prueba: 30 semillas por red + bootstrap del baseline
  figM2-12_matriz_confusion.png          matriz de confusión en prueba del mejor modelo (semilla 0)
  resultados/comparacion_modelos.csv     tabla resumen de la comparación justa
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .inventario import CLASES, NOMBRES
from .utils import resolve

FAMILIAS = {"cnn": "CNN desde cero", "transfer": "Transfer learning (EfficientNet-B0)"}
COLOR = {"baseline": "#8C8C8C", "cnn": "#DD8452", "transfer": "#4C72B0"}


def _aumento(texto: str) -> str:
    a = json.loads(texto)
    return {5: "leve", 10: "medio", 15: "fuerte"}.get(a.get("rotacion"), "?")


def fig_busqueda(res, fig_dir):
    # Sin sharey: cada panel tiene sus propias etiquetas (con eje compartido, las del
    # segundo panel sobrescribían las del primero; ver hallazgo H2-10)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    marcas = {"leve": "o", "medio": "s", "fuerte": "^"}
    for ax, fam in zip(axes, FAMILIAS):
        b = res / f"busqueda_{fam}.csv"
        if not b.exists():
            ax.set_visible(False)
            continue
        t = pd.read_csv(b)
        t["aum"] = t["hp_aumento"].map(_aumento)
        t["opt"] = t["hp__optimizacion"].map(lambda s: "{optimizador} {lr:g}".format(**json.loads(s)))
        t = t.sort_values("f1_macro_validacion").reset_index(drop=True)
        for aum, g in t.groupby("aum"):
            ax.scatter(g["f1_macro_validacion"], g.index, marker=marcas.get(aum, "x"), s=60, color=COLOR[fam], label=f"aumento {aum}")
        ax.set_yticks(t.index, [f"#{c} {o}" for c, o in zip(t["config"], t["opt"])], fontsize=7)
        mejor = t["f1_macro_validacion"].idxmax()
        ax.scatter(t.loc[mejor, "f1_macro_validacion"], mejor, s=200, facecolors="none", edgecolors="red", label="elegida")
        ax.set_title(f"{FAMILIAS[fam]}\n{len(t)} configuraciones, {t['tiempo_s'].sum() / 60:.0f} min de cómputo", fontsize=10)
        ax.set_xlabel("F1 macro en validación")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(axis="x", alpha=0.3)
    base = res / "baseline_metricas.json"
    if base.exists():
        f1b = json.load(open(base, encoding="utf-8"))["validacion"]["f1_macro"]
        for ax in axes:
            ax.axvline(f1b, color=COLOR["baseline"], ls="--", lw=1)
            ax.text(f1b, ax.get_ylim()[1], " referencia", color=COLOR["baseline"], fontsize=7, va="top")
    fig.suptitle("Búsqueda aleatoria de hiperparámetros (semilla 42, máx. 12 épocas)")
    fig.tight_layout()
    fig.savefig(fig_dir / "figM2-9_busqueda_hiperparametros.png", dpi=150)
    plt.close(fig)


def fig_curvas(res, fig_dir):
    fams = [f for f in FAMILIAS if (res / "curvas" / f"final_{f}_semilla0.csv").exists()]
    if not fams:
        return
    fig, axes = plt.subplots(2, len(fams), figsize=(6 * len(fams), 7), squeeze=False)
    for k, fam in enumerate(fams):
        c = pd.read_csv(res / "curvas" / f"final_{fam}_semilla0.csv")
        mejor = int(c.loc[c["f1_macro_validacion"].idxmax(), "epoca"])
        ax = axes[0, k]
        ax.plot(c["epoca"], c["perdida_entrenamiento"], "o-", label="entrenamiento")
        ax.plot(c["epoca"], c["perdida_validacion"], "s-", label="validación")
        ax.set_ylabel("Pérdida (entropía cruzada)")
        ax.set_title(FAMILIAS[fam], fontsize=10)
        ax2 = axes[1, k]
        ax2.plot(c["epoca"], c["exactitud_entrenamiento"], "o-", label="exactitud entrenamiento")
        ax2.plot(c["epoca"], c["f1_macro_validacion"], "s-", label="F1 macro validación")
        ax2.set_ylabel("Métrica")
        ax2.set_xlabel("Época")
        for a in (ax, ax2):
            a.axvline(mejor, color="red", ls=":", label=f"mejor época ({mejor}); se restauran sus pesos")
            a.axvspan(mejor + 0.5, c["epoca"].max() + 0.5, color="red", alpha=0.06,
                      label="épocas sin mejora (parada temprana)")
            a.legend(fontsize=7)
            a.grid(alpha=0.3)
        if fam == "transfer":
            mf = json.load(open(res / "final_transfer_metricas.json", encoding="utf-8"))["config"]["transfer"]["epocas_cabeza"]
            for a in (ax, ax2):
                a.axvspan(0.5, mf + 0.5, color="green", alpha=0.08)
            ax.text(mf / 2 + 0.5, ax.get_ylim()[1], "solo\ncabeza", ha="center", va="top", fontsize=7, color="green")
    fig.suptitle("Curvas de aprendizaje de las redes finales (semilla 0)")
    fig.tight_layout()
    fig.savefig(fig_dir / "figM2-10_curvas_aprendizaje.png", dpi=150)
    plt.close(fig)


def tabla_comparacion(res) -> pd.DataFrame:
    filas = []
    if (res / "baseline_metricas.json").exists():
        m = json.load(open(res / "baseline_metricas.json", encoding="utf-8"))
        bt = pd.read_csv(res / "baseline_bootstrap.csv")["f1_macro"]
        filas.append(dict(
            modelo="Referencia: logística sobre EfficientNet-B0 congelada", clave="baseline",
            variabilidad="bootstrap de prueba (1.000)", f1_prueba_media=bt.mean(), f1_prueba_desv=bt.std(),
            f1_prueba_min=bt.min(), f1_prueba_max=bt.max(), f1_validacion_media=m["validacion"]["f1_macro"],
            f1_prueba_state_farm=m["prueba_por_procedencia"]["state_farm"]["f1_macro"],
            f1_prueba_otra=m["prueba_por_procedencia"]["otra"]["f1_macro"],
            parametros_entrenables=m["parametros_entrenables"],
            configuraciones_busqueda=m["presupuesto_busqueda"]["configuraciones"],
            minutos_busqueda=round(m["tiempo_s"]["busqueda"] / 60, 1), minutos_30_semillas=np.nan,
        ))
    for fam in FAMILIAS:
        s, b = res / f"semillas_{fam}.csv", res / f"busqueda_{fam}.csv"
        if not s.exists():
            continue
        s, b = pd.read_csv(s), pd.read_csv(b)
        filas.append(dict(
            modelo=FAMILIAS[fam], clave=fam, variabilidad=f"{len(s)} semillas",
            f1_prueba_media=s["f1_macro_prueba"].mean(), f1_prueba_desv=s["f1_macro_prueba"].std(),
            f1_prueba_min=s["f1_macro_prueba"].min(), f1_prueba_max=s["f1_macro_prueba"].max(),
            f1_validacion_media=s["f1_macro_validacion"].mean(),
            f1_prueba_state_farm=s["f1_prueba_state_farm"].mean(), f1_prueba_otra=s["f1_prueba_otra"].mean(),
            parametros_entrenables=int(s["parametros_entrenables"].iloc[0]), configuraciones_busqueda=len(b),
            minutos_busqueda=round(b["tiempo_s"].sum() / 60, 1), minutos_30_semillas=round(s["tiempo_s"].sum() / 60, 1),
        ))
    t = pd.DataFrame(filas)
    num = t.select_dtypes("number").columns
    t[num] = t[num].round(4)
    t.to_csv(res / "comparacion_modelos.csv", index=False)
    return t


def fig_semillas(res, fig_dir):
    series, etiquetas, colores = [], [], []
    if (res / "baseline_bootstrap.csv").exists():
        series.append(pd.read_csv(res / "baseline_bootstrap.csv")["f1_macro"].to_numpy())
        etiquetas.append("Referencia\n(bootstrap de prueba)")
        colores.append(COLOR["baseline"])
    for fam in FAMILIAS:
        if (res / f"semillas_{fam}.csv").exists():
            series.append(pd.read_csv(res / f"semillas_{fam}.csv")["f1_macro_prueba"].to_numpy())
            etiquetas.append(FAMILIAS[fam].replace(" (", "\n(") + "\n(30 semillas)")
            colores.append(COLOR[fam])
    if not series:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    bp = ax.boxplot(series, tick_labels=etiquetas, patch_artist=True, widths=0.5)
    for p, c in zip(bp["boxes"], colores):
        p.set_facecolor(c)
        p.set_alpha(0.5)
    rng = np.random.default_rng(0)
    for k, s in enumerate(series[1:] if len(series) > 2 else [], start=2):
        ax.scatter(k + rng.uniform(-0.08, 0.08, len(s)), s, s=12, color="k", alpha=0.6, zorder=3)
    ax.set_ylabel("F1 macro en prueba")
    ax.set_title("Variabilidad entre corridas")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=8)

    # Por procedencia: medias y desviaciones
    t = tabla_comparacion(res)
    ax = axes[1]
    x = np.arange(len(t))
    for j, (col, nombre) in enumerate([("f1_prueba_state_farm", "State Farm (997)"), ("f1_prueba_otra", "Fotos de 2024 (136)")]):
        ax.bar(x + (j - 0.5) * 0.35, t[col], width=0.35, label=nombre, color=[COLOR[c] for c in t["clave"]],
               alpha=0.9 if j == 0 else 0.45, hatch="" if j == 0 else "//", edgecolor="k")
    ax.axhline(0.2, color="k", ls=":", lw=1, label="azar (5 clases, F1 ≈ 0,2)")
    ax.set_xticks(x, [e.split("\n")[0] for e in etiquetas], fontsize=8)
    ax.set_ylabel("F1 macro en prueba (media)")
    ax.set_title("Desempeño por procedencia")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1)
    fig.suptitle("Comparación en prueba: mismos datos, misma partición y mismo preprocesamiento")
    fig.tight_layout()
    fig.savefig(fig_dir / "figM2-11_comparacion_semillas.png", dpi=150)
    plt.close(fig)


def fig_matriz(res, fig_dir, familia="transfer"):
    ruta = res / f"final_{familia}_metricas.json"
    if not ruta.exists():
        return
    m = json.load(open(ruta, encoding="utf-8"))
    cm = np.array(m["prueba"]["matriz_confusion"])
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASES)):
        for j in range(len(CLASES)):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.6 else "black")
    nombres = [NOMBRES[c] for c in CLASES]
    ax.set_xticks(range(len(CLASES)), nombres, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(CLASES)), nombres, fontsize=8)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(f"{FAMILIAS[familia]}, semilla {m['semilla']}: prueba\n"
                 f"exactitud {m['prueba']['exactitud']:.3f}, F1 macro {m['prueba']['f1_macro']:.3f}", fontsize=10)
    fig.colorbar(im, fraction=0.046, label="proporción de la fila")
    fig.tight_layout()
    fig.savefig(fig_dir / "figM2-12_matriz_confusion.png", dpi=150)
    plt.close(fig)


def ejecutar(cfg: dict) -> None:
    res = resolve(cfg["salidas"]["resultados"])
    fig_dir = resolve(cfg["salidas"]["figuras"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_busqueda(res, fig_dir)
    fig_curvas(res, fig_dir)
    fig_semillas(res, fig_dir)
    fig_matriz(res, fig_dir)
    print(tabla_comparacion(res).drop(columns=["clave"]).to_string(index=False))
