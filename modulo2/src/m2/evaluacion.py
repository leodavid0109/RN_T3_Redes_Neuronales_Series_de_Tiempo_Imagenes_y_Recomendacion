"""Paso 9: evaluación del modelo final, interpretabilidad y política de alertas.

  py run.py evaluacion

Usa los modelos guardados (models/<familia>_final.pt, semilla 0), sin reentrenar.

1. Grad-CAM (Selvaraju et al., 2017) sobre la última capa convolucional:
   - 10 aciertos y 10 errores del mejor modelo en prueba     -> figM2-13, figM2-14
   - mapa promedio por clase en las imágenes de State Farm   -> figM2-15
     (cámara fija: el promedio muestra en qué zonas se apoya el modelo)
2. Política de alertas: puntaje = 1 - P(conducción segura). El umbral se elige en
   VALIDACIÓN minimizando el costo esperado para varios cocientes de costo
   (distracción no detectada / falsa alarma) y se reporta en PRUEBA, por cada 1.000
   imágenes, con la prevalencia del conjunto y con una prevalencia supuesta.   -> figM2-16
3. Desempeño por grupo de prueba (conductor de State Farm o sesión) de los tres
   modelos (variabilidad que las 30 semillas no capturan).                    -> figM2-17
4. Diagrama de arquitectura de las dos redes con dimensiones y parámetros.     -> figM2-18
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

from .datos_torch import cargador, cargar_splits, transformacion_evaluacion
from .entrenamiento import predecir
from .inventario import CLASES, NOMBRES
from .modelos import construir
from .utils import get_device, resolve

FAMILIAS = {"cnn": "CNN desde cero", "transfer": "Transfer learning (EfficientNet-B0)"}


# ---------------------------------------------------------------- modelos
def cargar_modelo(cfg: dict, familia: str, dev):
    ruta = resolve(cfg["salidas"]["modelos"]) / f"{familia}_final.pt"
    ck = torch.load(ruta, map_location=dev, weights_only=True)
    rcfg = dict(ck["rcfg"])
    rcfg.setdefault("transfer", {})["preentrenado"] = False  # los pesos vienen del archivo
    m = construir(rcfg)
    m.load_state_dict(ck["estado"])
    return m.to(dev).eval(), rcfg


def capa_objetivo(modelo: nn.Module, familia: str) -> nn.Module:
    """Última capa convolucional (mapa de 7 x 7) sobre la que se calcula Grad-CAM."""
    return modelo.features[-1] if familia == "transfer" else modelo.features


class GradCAM:
    """Grad-CAM: L^c = ReLU( sum_k alpha_k^c A^k ),  alpha_k^c = media espacial de dy^c / dA^k."""

    def __init__(self, modelo: nn.Module, capa: nn.Module):
        self.modelo, self.A, self.G = modelo, None, None
        self.handle = capa.register_forward_hook(self._guardar)

    def _guardar(self, _m, _entrada, salida):
        self.A = salida
        if salida.requires_grad:  # en pasadas sin gradiente (torch.no_grad) no hay nada que guardar
            salida.register_hook(lambda g: setattr(self, "G", g))

    def cerrar(self):
        self.handle.remove()

    def __call__(self, x: torch.Tensor, clase: torch.Tensor | None = None) -> tuple[np.ndarray, np.ndarray]:
        self.modelo.eval()
        self.modelo.zero_grad(set_to_none=True)
        z = self.modelo(x)
        if clase is None:
            clase = z.argmax(1)
        z.gather(1, clase[:, None]).sum().backward()
        alfa = self.G.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((alfa * self.A).sum(1, keepdim=True))
        cam = nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        mx = cam.flatten(1).max(1).values.clamp(min=1e-8)
        return (cam / mx[:, None, None]).detach().cpu().numpy(), torch.softmax(z, 1).detach().cpu().numpy()


def _cams(cfg, modelo, familia, df, dev, lote=32):
    tf = transformacion_evaluacion(cfg["preproceso"]["tamano"])
    gc = GradCAM(modelo, capa_objetivo(modelo, familia))
    raiz = resolve(cfg["datos"]["processed_dir"]) / cfg["preproceso"]["carpeta"]
    cams, probs = [], []
    for i in range(0, len(df), lote):
        x = torch.stack([tf(Image.open(raiz / r).convert("RGB")) for r in df["ruta_proc"].iloc[i : i + lote]]).to(dev)
        c, p = gc(x)
        cams.append(c)
        probs.append(p)
    gc.cerrar()
    return np.concatenate(cams), np.concatenate(probs)


# ---------------------------------------------------------------- figuras de ejemplos
def _seleccion(dpr: pd.DataFrame, pred: np.ndarray, n: int, aciertos: bool, seed: int) -> pd.DataFrame:
    """n ejemplos repartidos entre clases reales; en los errores se reservan hasta 3 de las fotos de 2024."""
    rng = np.random.default_rng(seed)
    ok = pred == dpr["etiqueta"].to_numpy()
    cand = dpr[ok] if aciertos else dpr[~ok]
    elegidos = []
    if not aciertos:
        otra = cand[cand["procedencia"] == "otra"]
        elegidos += list(rng.choice(otra.index, size=min(3, len(otra)), replace=False))
    for c in CLASES:  # una ronda por clase, luego se completa al azar
        sub = cand[(cand["clase"] == c) & ~cand.index.isin(elegidos)]
        if len(sub) and len(elegidos) < n:
            elegidos.append(rng.choice(sub.index))
    resto = cand[~cand.index.isin(elegidos)]
    elegidos += list(rng.choice(resto.index, size=min(n - len(elegidos), len(resto)), replace=False))
    return dpr.loc[elegidos[:n]]


def fig_ejemplos(cfg, sel, cams, probs, titulo, archivo):
    raiz = resolve(cfg["datos"]["processed_dir"]) / cfg["preproceso"]["carpeta"]
    fig, axes = plt.subplots(4, 5, figsize=(15, 13))
    for k, (_, fila) in enumerate(sel.iterrows()):
        im = np.asarray(Image.open(raiz / fila["ruta_proc"]).convert("RGB"))
        for j, mapa in enumerate([False, True]):
            ax = axes[2 * (k // 5) + j, k % 5]
            ax.imshow(im)
            if mapa:
                ax.imshow(cams[k], cmap="jet", alpha=0.45, vmin=0, vmax=1)
            ax.axis("off")
            if not mapa:
                p = probs[k]
                ax.set_title(f"#{k + 1} real: {NOMBRES[fila['clase']]}\npred: {NOMBRES[CLASES[p.argmax()]]} ({p.max():.2f})\n"
                             f"{'State Farm ' + fila['grupo'][3:] if fila['procedencia'] == 'state_farm' else 'foto 2024'}",
                             fontsize=7.5)
    fig.suptitle(titulo + "\nFila superior: imagen; fila inferior: Grad-CAM de la clase predicha (rojo = mayor peso)", fontsize=11)
    fig.tight_layout()
    fig.savefig(archivo, dpi=110)
    plt.close(fig)


def fig_cam_promedio(cfg, cams_sf, dsf, pred_sf, archivo):
    raiz = resolve(cfg["datos"]["processed_dir"]) / cfg["preproceso"]["carpeta"]
    fondo = np.mean([np.asarray(Image.open(raiz / r).convert("L"), dtype=np.float32)
                     for r in dsf["ruta_proc"].iloc[:: max(1, len(dsf) // 200)]], axis=0)
    fig, axes = plt.subplots(1, len(CLASES), figsize=(16, 3.8))
    for k, c in enumerate(CLASES):
        m = (dsf["clase"] == c).to_numpy() & (pred_sf == k)
        ax = axes[k]
        ax.imshow(fondo, cmap="gray")
        if m.any():
            ax.imshow(cams_sf[m].mean(0), cmap="jet", alpha=0.5, vmin=0, vmax=1)
        ax.set_title(f"{NOMBRES[c]}\n(aciertos de State Farm, n={m.sum()})", fontsize=9)
        ax.axis("off")
    fig.suptitle("Grad-CAM promedio por clase sobre el promedio de las imágenes de State Farm en prueba (cámara fija)")
    fig.tight_layout()
    fig.savefig(archivo, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- alertas
def tabla_umbral(y: np.ndarray, p_seg: np.ndarray, umbrales: np.ndarray) -> pd.DataFrame:
    alerta_real = y != CLASES.index("conduccion_segura")
    s = 1 - p_seg
    filas = []
    for u in umbrales:
        alerta = s >= u
        fp = int((alerta & ~alerta_real).sum())
        fn = int((~alerta & alerta_real).sum())
        filas.append(dict(umbral=round(float(u), 3), falsas_alarmas=fp, no_detectadas=fn,
                          tasa_falsa_alarma=fp / max((~alerta_real).sum(), 1), tasa_no_deteccion=fn / max(alerta_real.sum(), 1)))
    return pd.DataFrame(filas)


def por_mil(t: pd.DataFrame, prevalencia: float) -> pd.DataFrame:
    t = t.copy()
    t["falsas_alarmas_x1000"] = 1000 * (1 - prevalencia) * t["tasa_falsa_alarma"]
    t["no_detectadas_x1000"] = 1000 * prevalencia * t["tasa_no_deteccion"]
    return t


def politica_alertas(cfg, y_va, p_va, y_pr, p_pr, res_dir, fig_dir) -> dict:
    acfg = cfg["alertas"]
    umbrales = np.round(np.arange(0.02, 0.99, 0.02), 2)
    seg = CLASES.index("conduccion_segura")
    tv, tp = tabla_umbral(y_va, p_va[:, seg], umbrales), tabla_umbral(y_pr, p_pr[:, seg], umbrales)
    prev_conj = float((y_pr != seg).mean())
    prevs = {"conjunto_prueba": prev_conj, "supuesta": acfg["prevalencia_supuesta"]}
    elegidos = {}
    for prev_nombre, prev in prevs.items():
        v = por_mil(tv, prev)
        for r in acfg["cocientes_costo"]:
            costo = v["falsas_alarmas_x1000"] + r * v["no_detectadas_x1000"]
            u = float(v.loc[costo.idxmin(), "umbral"])
            fila = por_mil(tp, prev).set_index("umbral").loc[u]
            elegidos[f"{prev_nombre}_costo{r}"] = dict(
                prevalencia=round(prev, 4), cociente_costo_FN_FP=r, umbral_elegido_en_validacion=u,
                prueba_falsas_alarmas_x1000=round(float(fila["falsas_alarmas_x1000"]), 1),
                prueba_no_detectadas_x1000=round(float(fila["no_detectadas_x1000"]), 1),
                prueba_tasa_falsa_alarma=round(float(fila["tasa_falsa_alarma"]), 4),
                prueba_tasa_no_deteccion=round(float(fila["tasa_no_deteccion"]), 4),
            )
    out = pd.concat([por_mil(tp, prev_conj).assign(prevalencia="conjunto_prueba"),
                     por_mil(tp, acfg["prevalencia_supuesta"]).assign(prevalencia="supuesta")])
    out.to_csv(res_dir / "alertas_umbral_prueba.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (prev_nombre, prev) in zip(axes, prevs.items()):
        t = por_mil(tp, prev)
        ax.plot(t["umbral"], t["falsas_alarmas_x1000"], "o-", ms=3, label="falsas alarmas (conducción segura con alerta)")
        ax.plot(t["umbral"], t["no_detectadas_x1000"], "s-", ms=3, label="distracciones no detectadas")
        for r, est in zip(acfg["cocientes_costo"], [":", "--", "-.", (0, (1, 3))]):
            u = elegidos[f"{prev_nombre}_costo{r}"]["umbral_elegido_en_validacion"]
            ax.axvline(u, color="k", ls=est, lw=1, label=f"umbral elegido, costo FN/FP = {r}: {u:.2f}")
        ax.set_xlabel("Umbral sobre el puntaje de alerta = 1 − P(conducción segura)")
        ax.set_ylabel("Casos por cada 1.000 imágenes")
        ax.set_title(f"Prevalencia de distracción {prev:.0%} ({'conjunto de prueba' if prev_nombre == 'conjunto_prueba' else 'supuesta'})",
                     fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Política de alertas (transfer learning, semilla 0): umbral elegido en validación, resultados en prueba")
    fig.tight_layout()
    fig.savefig(fig_dir / "figM2-16_alertas_umbral.png", dpi=150)
    plt.close(fig)
    return {"prevalencias": prevs, "elegidos": elegidos}


# ---------------------------------------------------------------- por grupo
def metricas_por_grupo(dpr: pd.DataFrame, res_dir, fig_dir) -> pd.DataFrame:
    y = dpr["etiqueta"].to_numpy()
    modelos = {}
    bp = res_dir / "baseline_predicciones.csv"
    if bp.exists():
        b = pd.read_csv(bp)
        b = b[b["particion"] == "prueba"].set_index("ruta_proc").loc[dpr["ruta_proc"]]
        modelos["Referencia"] = [b[[f"p_{c}" for c in CLASES]].to_numpy()]
    for fam, nombre in [("cnn", "CNN desde cero"), ("transfer", "Transfer learning")]:
        f = res_dir / f"prob_prueba_{fam}.npz"
        if f.exists():
            z = np.load(f)
            modelos[nombre] = [z[k] for k in z.files]
    filas = []
    for nombre, lista in modelos.items():
        for s, p in enumerate(lista):
            ok = p.argmax(1) == y
            for g, idx in dpr.groupby("grupo").indices.items():
                filas.append(dict(modelo=nombre, corrida=s, grupo=g, n=len(idx), exactitud=ok[idx].mean()))
    t = pd.DataFrame(filas)
    t.to_csv(res_dir / "metricas_por_grupo_prueba.csv", index=False)
    resumen = t.groupby(["modelo", "grupo"])["exactitud"].agg(["mean", "std", "min", "max"]).round(4)

    grupos = sorted(t["grupo"].unique())
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colores = {"Referencia": "#8C8C8C", "CNN desde cero": "#DD8452", "Transfer learning": "#4C72B0"}
    for k, nombre in enumerate(modelos):
        sub = t[t["modelo"] == nombre]
        for j, g in enumerate(grupos):
            v = sub[sub["grupo"] == g]["exactitud"].to_numpy()
            x = j + (k - 1) * 0.25
            ax.scatter(np.full(len(v), x) + np.random.default_rng(j).uniform(-0.05, 0.05, len(v)), v, s=8,
                       color=colores[nombre], alpha=0.5)
            ax.scatter(x, v.mean(), s=80, color=colores[nombre], edgecolor="k", label=nombre if j == 0 else None, zorder=3)
    n = t.drop_duplicates("grupo").set_index("grupo")["n"]
    ax.set_xticks(range(len(grupos)), [f"{g.replace('sf_', 'conductor ').replace('ses_', 'sesión ')}\n(n={n[g]})" for g in grupos],
                  fontsize=7.5)
    ax.set_ylabel("Exactitud en prueba")
    ax.set_title("Exactitud por grupo de prueba (puntos pequeños: semillas; grandes: media)")
    ax.axhline(0.2, color="k", ls=":", lw=1)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "figM2-17_exactitud_por_grupo.png", dpi=150)
    plt.close(fig)
    return resumen


# ---------------------------------------------------------------- arquitectura
def resumen_capas(modelo: nn.Module, familia: str) -> list[dict]:
    """Capas de alto nivel con forma de salida, activación y parámetros (pasada con una imagen de 224 x 224)."""
    if familia == "transfer":
        bloques = [(f"features[{i}]", b) for i, b in enumerate(modelo.features)] + [("avgpool", modelo.avgpool)] + \
                  [(f"classifier[{i}]", b) for i, b in enumerate(modelo.classifier)]
    else:
        convs = list(modelo.features.children())
        bloques = [(f"bloque {i // 7 + 1}", nn.Sequential(*convs[i : i + 7])) for i in range(0, len(convs), 7)] + \
                  [("pool", modelo.pool)] + [(f"classifier[{i}]", b) for i, b in enumerate(modelo.classifier)]
    filas, x = [], torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        for nombre, b in bloques:
            if nombre == "classifier[0]" and familia == "transfer":
                x = torch.flatten(x, 1)
            x = b(x)
            tipo = type(b).__name__
            if familia == "transfer" and nombre.startswith("features"):
                tipo = "Conv-BN-SiLU" if int(nombre[9:-1]) in (0, 8) else f"MBConv x{len(b)}"
            if familia == "cnn" and nombre.startswith("bloque"):
                tipo = "[Conv3x3-BN-ReLU]x2 + MaxPool"
            filas.append(dict(capa=nombre, tipo=tipo, salida="x".join(map(str, x.shape[1:])),
                              parametros=sum(p.numel() for p in b.parameters())))
    return filas


def fig_arquitectura(modelos: dict, fig_dir):
    fig, axes = plt.subplots(1, len(modelos), figsize=(7 * len(modelos), 7.5), squeeze=False)
    for ax, (fam, (m, rcfg)) in zip(axes[0], modelos.items()):
        filas = resumen_capas(m.cpu(), fam)
        ax.axis("off")
        n = len(filas)
        ax.set_xlim(0, 1)
        ax.set_ylim(-2.6, n + 1)
        ax.text(0.5, n + 0.6, f"{FAMILIAS[fam]}\nentrada 3x224x224 (RGB normalizada)", ha="center", fontsize=10, weight="bold")
        for k, f in enumerate(filas):
            yk = n - 1 - k
            congelado = fam == "transfer" and f["capa"].startswith("features[") and \
                int(f["capa"][9:-1]) < 9 - rcfg["transfer"]["bloques_descongelados"]
            color = "#D9D9D9" if congelado else ("#C6DBEF" if fam == "transfer" else "#FDD0A2")
            ax.add_patch(plt.Rectangle((0.05, yk - 0.4), 0.9, 0.8, color=color, ec="k", lw=0.6))
            ax.text(0.08, yk, f"{f['capa']}: {f['tipo']}", va="center", fontsize=8)
            ax.text(0.70, yk, f"{f['salida']}", va="center", fontsize=8, family="monospace")
            ax.text(0.93, yk, f"{f['parametros']:,}".replace(",", "."), va="center", ha="right", fontsize=8)
        total = sum(p.numel() for p in m.parameters())
        nota = f"Parámetros totales: {total:,}".replace(",", ".")
        if fam == "transfer":
            nota += f"\nGris: bloques congelados (preentrenados en ImageNet); azul: ajustados.\n" \
                    f"Bloques descongelados: {rcfg['transfer']['bloques_descongelados']}; dropout {rcfg['dropout']}"
        else:
            nota += f"\nTodos entrenables; dropout {rcfg['dropout']}; activación ReLU"
        ax.text(0.5, -0.75, nota + "\nSalida: 5 logits -> softmax", ha="center", va="top", fontsize=8.5)
        ax.text(0.70, n + 0.05, "salida", fontsize=7.5, weight="bold")
        ax.text(0.93, n + 0.05, "parámetros", fontsize=7.5, weight="bold", ha="right")
    fig.tight_layout()
    fig.savefig(fig_dir / "figM2-18_arquitecturas.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- principal
def ejecutar(cfg: dict) -> dict:
    dev = get_device(cfg.get("dispositivo", "auto"))
    res_dir, fig_dir = resolve(cfg["salidas"]["resultados"]), resolve(cfg["salidas"]["figuras"])
    df = cargar_splits(cfg)
    dva = df[df["particion"] == "validacion"].reset_index(drop=True)
    dpr = df[df["particion"] == "prueba"].reset_index(drop=True)
    tf = transformacion_evaluacion(cfg["preproceso"]["tamano"])
    salida = {}

    modelos = {}
    for fam in FAMILIAS:
        if (resolve(cfg["salidas"]["modelos"]) / f"{fam}_final.pt").exists():
            modelos[fam] = cargar_modelo(cfg, fam, dev)
    fam = cfg["evaluacion"]["familia"]
    modelo, _ = modelos[fam]

    # Probabilidades del modelo final en validación y prueba
    p_va, _ = predecir(modelo, cargador(cfg, dva, tf, 64), dev)
    p_pr, _ = predecir(modelo, cargador(cfg, dpr, tf, 64), dev)
    y_va, y_pr = dva["etiqueta"].to_numpy(), dpr["etiqueta"].to_numpy()
    pred = p_pr.argmax(1)
    print(f"Modelo final ({fam}, semilla 0): exactitud en prueba {np.mean(pred == y_pr):.4f}")

    # 1. Grad-CAM: ejemplos
    seed = cfg["seed"]
    for aciertos, archivo, titulo in [
        (True, "figM2-13_gradcam_aciertos.png", "10 imágenes de prueba bien clasificadas"),
        (False, "figM2-14_gradcam_errores.png", "10 imágenes de prueba mal clasificadas"),
    ]:
        sel = _seleccion(dpr, pred, 10, aciertos, seed)
        cams, probs = _cams(cfg, modelo, fam, sel, dev)
        fig_ejemplos(cfg, sel, cams, probs, f"{FAMILIAS[fam]}, semilla 0: {titulo}", fig_dir / archivo)
        sel.assign(prediccion=[CLASES[k] for k in probs.argmax(1)], confianza=probs.max(1).round(4))[
            ["ruta", "ruta_proc", "clase", "prediccion", "confianza", "procedencia", "grupo"]
        ].to_csv(res_dir / archivo.replace(".png", ".csv").replace("figM2-13_", "").replace("figM2-14_", ""), index=False)

    # Grad-CAM promedio por clase (aciertos de State Farm)
    dsf = dpr[dpr["procedencia"] == "state_farm"].reset_index(drop=True)
    cams_sf, probs_sf = _cams(cfg, modelo, fam, dsf, dev)
    fig_cam_promedio(cfg, cams_sf, dsf, probs_sf.argmax(1), fig_dir / "figM2-15_gradcam_promedio.png")

    # 2. Alertas
    salida["alertas"] = politica_alertas(cfg, y_va, p_va, y_pr, p_pr, res_dir, fig_dir)

    # 3. Por grupo
    salida["por_grupo"] = metricas_por_grupo(dpr, res_dir, fig_dir).reset_index().to_dict(orient="records")

    # 4. Arquitecturas
    fig_arquitectura(modelos, fig_dir)
    salida["arquitecturas"] = {f: resumen_capas(m.cpu(), f) for f, (m, _) in modelos.items()}

    with open(res_dir / "evaluacion.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print(json.dumps(salida["alertas"]["elegidos"], indent=2, ensure_ascii=False))
    print(pd.DataFrame(salida["por_grupo"]).to_string(index=False))
    return salida
