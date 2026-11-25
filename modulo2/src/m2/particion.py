"""Paso 3: partición sin fuga de información, por conductor y por grupos de similitud.

Contexto (registro, secciones 8 y 9): el 91 % de las imágenes proviene del conjunto
State Farm (26 conductores, todos presentes en todas las clases). Para esas imágenes
el identificador de conductor sale de `driver_imgs_list.csv` (Kaggle).

1. Grupos (union-find):
   - todas las imágenes de un mismo conductor de State Farm forman un grupo;
   - además, se unen las imágenes conectadas por un par con coseno centrado
     >= umbral_coseno (casi duplicados o la misma sesión). En las imágenes sin
     identificador de conductor esto aproxima «misma persona o sesión».
2. Partición por grupos, por separado dentro de cada procedencia (State Farm y
   otras), para que las tres particiones tengan imágenes de ambas. Se prueban
   n_intentos órdenes aleatorios de los grupos (semilla fija) y se elige el que
   más se acerca a las proporciones objetivo y a la distribución de clases.
3. Verificación independiente: ningún grupo ni MD5 compartido entre particiones,
   y ninguna imagen con un par >= umbral en otra partición (similitudes recalculadas).

Salidas:
  data/processed/splits.csv            ruta, clase, etiqueta, origen, procedencia, grupo, particion
  data/processed/verificacion_fuga.json
  figuras/figM2-6_particion_por_clase.png
  figuras/figM2-7_verificacion_fuga.png
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .duplicados import similitudes_bloque
from .inventario import CLASES, NOMBRES
from .utils import resolve

PARTICIONES = ["entrenamiento", "validacion", "prueba"]


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def conductores_state_farm(inv: pd.DataFrame, cfg: dict) -> pd.Series:
    ruta = resolve(cfg["datos"]["state_farm_lista"])
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Descarga driver_imgs_list.csv de la competencia 'State Farm Distracted "
            "Driver Detection' en Kaggle y ajusta datos.state_farm_lista (registro, sección 9)."
        )
    lista = pd.read_csv(ruta).set_index("img")["subject"]
    nombre = inv["ruta"].str.rsplit("/", n=1).str[1]
    return nombre.map(lista)


def _momento_captura(nombre: str):
    """Fecha y hora de captura contenidas en el nombre del archivo, si las hay."""
    import re

    m = re.match(r"IMG_(\d{8})_(\d{6})", nombre)
    if m:
        return pd.Timestamp(m[1] + m[2])
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})(\d{2})-(\d{2})-(\d{2})\.png$", nombre)
    if m:
        return pd.Timestamp(f"{m[1]}-{m[2]}-{m[3]} {m[4]}:{m[5]}:{m[6]}")
    m = re.match(r"photo_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})", nombre)
    if m:
        return pd.Timestamp(f"{m[1]} {m[2]}:{m[3]}:{m[4]}")
    return pd.NaT


def sesiones_captura(inv: pd.DataFrame, subject: pd.Series, gap_min: float) -> pd.Series:
    """Sesión de captura de las imágenes que no son de State Farm.

    En estas imágenes la similitud de embeddings no separa bien a las personas
    (el mismo bus o la misma pose dan coseno alto entre personas distintas, y la
    misma persona puede quedar por debajo del umbral; registro, sección 10).
    Se agrupan por sesión: fotos del mismo origen tomadas con menos de `gap_min`
    minutos entre una y la siguiente. Los orígenes sin fecha en el nombre
    (foto_IMG_N, captura_pantalla) forman una sesión cada uno.
    """
    ses = pd.Series(pd.NA, index=inv.index, dtype="object")
    otra = inv[subject.isna()].copy()
    otra["t"] = otra["ruta"].str.rsplit("/", n=1).str[1].map(_momento_captura)
    for org, g in otra.groupby("origen"):
        if g["t"].isna().all():
            ses[g.index] = f"{org}"
            continue
        g = g.sort_values("t")
        corte = (g["t"].diff() > pd.Timedelta(minutes=gap_min)).cumsum()
        ses[g.index] = [f"{org}_s{k}" for k in corte]
    return ses


def agrupar(inv: pd.DataFrame, subject: pd.Series, sesion: pd.Series, pares: pd.DataFrame, umbral: float) -> np.ndarray:
    n = len(inv)
    uf = UnionFind(n)
    for etiqueta in (subject, sesion):
        validos = etiqueta.notna()
        for _, idx in pd.Series(np.arange(n))[validos].groupby(etiqueta[validos]):
            idx = idx.to_numpy()
            for k in idx[1:]:
                uf.union(int(idx[0]), int(k))
    for i, j in pares.loc[pares["coseno"] >= umbral, ["i", "j"]].itertuples(index=False):
        uf.union(int(i), int(j))
    raices = np.array([uf.find(i) for i in range(n)])
    _, grupo = np.unique(raices, return_inverse=True)
    return grupo


def _puntaje(asig, tam_grupo, clases_grupo, objetivo, dist_global):
    """Desvío de las proporciones de imágenes y de la distribución de clases."""
    total = tam_grupo.sum()
    s = 0.0
    for p, obj in objetivo.items():
        m = asig == p
        n = tam_grupo[m].sum()
        if n == 0:
            return np.inf
        s += abs(n / total - obj)
        s += 0.5 * np.abs(clases_grupo[m].sum(0) / n - dist_global).sum()
    return s


def partir_estrato(grupos: np.ndarray, y: np.ndarray, pcfg: dict, rng) -> dict:
    """Asigna cada grupo de un estrato a una partición (devuelve {grupo: partición})."""
    ids = np.unique(grupos)
    tam = np.array([(grupos == g).sum() for g in ids])
    clases = np.array([np.bincount(y[grupos == g], minlength=len(CLASES)) for g in ids])
    dist_global = np.bincount(y, minlength=len(CLASES)) / len(y)
    objetivo = {
        "prueba": pcfg["proporcion_prueba"],
        "validacion": pcfg["proporcion_validacion"],
        "entrenamiento": 1 - pcfg["proporcion_prueba"] - pcfg["proporcion_validacion"],
    }
    mejor, mejor_s = None, np.inf
    for _ in range(pcfg["n_intentos"]):
        orden = rng.permutation(len(ids))
        acum = np.cumsum(tam[orden]) / tam.sum()
        asig_ord = np.where(
            acum - tam[orden] / tam.sum() < objetivo["prueba"], "prueba",
            np.where(acum - tam[orden] / tam.sum() < objetivo["prueba"] + objetivo["validacion"], "validacion", "entrenamiento"),
        )
        asig = np.empty(len(ids), dtype=object)
        asig[orden] = asig_ord
        s = _puntaje(asig, tam, clases, objetivo, dist_global)
        if s < mejor_s:
            mejor, mejor_s = asig, s
    return dict(zip(ids, mejor))


def verificar(car: dict, df: pd.DataFrame, md5: np.ndarray, umbral: float) -> tuple[dict, dict]:
    part = df["particion"].to_numpy()
    idx = {p: np.where(part == p)[0] for p in PARTICIONES}
    combos = [("validacion", "entrenamiento"), ("prueba", "entrenamiento"), ("prueba", "validacion")]
    rep, dist = {}, {}
    for a, b in combos:
        max_cos = []
        for ini in range(0, len(idx[a]), 512):
            cos, _ = similitudes_bloque(car, idx[a][ini : ini + 512], idx[b])
            max_cos.append(cos.max(1))
        max_cos = np.concatenate(max_cos)
        dist[f"{a}->{b}"] = max_cos
        rep[f"{a}->{b}"] = {
            "coseno_centrado_maximo": round(float(max_cos.max()), 4),
            "coseno_centrado_p99": round(float(np.quantile(max_cos, 0.99)), 4),
            "imagenes_con_par_sobre_umbral": int((max_cos >= umbral).sum()),
            "grupos_compartidos": len(set(df["grupo"].to_numpy()[idx[a]]) & set(df["grupo"].to_numpy()[idx[b]])),
            "md5_compartidos": len(set(md5[idx[a]]) & set(md5[idx[b]])),
        }
    return rep, dist


def figuras(cfg, df, dist, nn_train):
    out = resolve(cfg["salidas"]["figuras"])
    out.mkdir(parents=True, exist_ok=True)

    # Fig. M2-6: imágenes por clase y partición (barras apiladas por procedencia)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(CLASES))
    colores = {"entrenamiento": "#4C72B0", "validacion": "#DD8452", "prueba": "#55A868"}
    for k, p in enumerate(PARTICIONES):
        sub = df[df["particion"] == p]
        sf = sub[sub["procedencia"] == "state_farm"]["clase"].value_counts().reindex(CLASES, fill_value=0).values
        ot = sub[sub["procedencia"] != "state_farm"]["clase"].value_counts().reindex(CLASES, fill_value=0).values
        pos = x + (k - 1) * 0.27
        ax.bar(pos, sf, width=0.27, color=colores[p], label=f"{p.capitalize()} – State Farm")
        b = ax.bar(pos, ot, width=0.27, bottom=sf, color=colores[p], alpha=0.45, hatch="//",
                   label=f"{p.capitalize()} – otras procedencias")
        ax.bar_label(b, labels=[str(v) for v in sf + ot], fontsize=7)
    ax.set_xticks(x, [NOMBRES[c] for c in CLASES], rotation=15)
    ax.set_ylabel("Número de imágenes")
    ax.set_title("Imágenes por clase, partición y procedencia")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(out / "figM2-6_particion_por_clase.png", dpi=150)
    plt.close(fig)

    # Fig. M2-7: similitud máxima con otra partición frente al vecino dentro de entrenamiento
    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.linspace(-0.2, 1.0, 100)
    ax.hist(nn_train, bins=bins, alpha=0.45, density=True, label="Entrenamiento → entrenamiento (vecino más cercano)")
    for k, v in dist.items():
        ax.hist(v, bins=bins, histtype="step", lw=1.8, density=True, label=k.replace("->", " → "))
    ax.axvline(cfg["particion"]["umbral_coseno"], color="red", ls="--", label="umbral de agrupación")
    ax.set_xlabel("Coseno centrado máximo con la otra partición")
    ax.set_ylabel("Densidad")
    ax.set_title("Verificación de fuga entre particiones")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figM2-7_verificacion_fuga.png", dpi=150)
    plt.close(fig)


def ejecutar(cfg: dict) -> dict:
    proc = resolve(cfg["datos"]["processed_dir"])
    pcfg = cfg["particion"]
    umbral = pcfg["umbral_coseno"]
    if umbral < cfg["duplicados"]["registro_coseno"]:
        raise ValueError("umbral_coseno es menor que duplicados.registro_coseno: baja el de registro y corre 'duplicados'.")
    inv = pd.read_csv(proc / "inventario.csv")
    pares = pd.read_csv(proc / "duplicados" / "pares_candidatos.csv.gz")
    car = dict(np.load(proc / "duplicados" / "caracteristicas.npz"))
    n = len(inv)
    if len(car["emb"]) != n:
        raise ValueError("caracteristicas.npz no corresponde a inventario.csv: vuelve a correr 'duplicados'.")

    subject = conductores_state_farm(inv, cfg)
    procedencia = np.where(subject.notna(), "state_farm", "otra")
    sesion = sesiones_captura(inv, subject, pcfg["sesion_gap_min"])
    grupo = agrupar(inv, subject, sesion, pares, umbral)

    # Protección: un grupo que abarca demasiado de su procedencia indica un umbral demasiado laxo
    for pr in ["state_farm", "otra"]:
        m = procedencia == pr
        if m.any():
            mayor = np.bincount(grupo[m]).max()
            if mayor > pcfg["max_fraccion_grupo"] * m.sum():
                raise ValueError(
                    f"Un grupo reúne {mayor} de {m.sum()} imágenes de procedencia '{pr}': el umbral es demasiado laxo "
                    "(o el backbone no tiene pesos preentrenados). Revisa figM2-5 y sube umbral_coseno."
                )

    y = inv["etiqueta"].to_numpy()
    rng = np.random.default_rng(cfg["seed"])
    part = np.empty(n, dtype=object)
    for pr in ["state_farm", "otra"]:
        m = np.where(procedencia == pr)[0]
        if len(m) == 0:
            continue
        asig = partir_estrato(grupo[m], y[m], pcfg, rng)
        part[m] = [asig[g] for g in grupo[m]]

    # Identificador de grupo legible: conductor de State Farm o grupo de similitud
    nombre_grupo = {}
    for g in np.unique(grupo):
        s = subject[grupo == g].dropna().unique()
        se = sesion[grupo == g].dropna().unique()
        if len(s):
            nombre_grupo[g] = "sf_" + "+".join(sorted(s))
        elif len(se):
            nombre_grupo[g] = "ses_" + "+".join(sorted(se))
        else:
            nombre_grupo[g] = f"sim_{g:05d}"
    df = inv[["ruta", "clase", "etiqueta", "origen"]].copy()
    df["procedencia"] = procedencia
    df["grupo"] = [nombre_grupo[g] for g in grupo]
    df["particion"] = part
    df.to_csv(proc / "splits.csv", index=False)

    rep, dist = verificar(car, df, inv["md5"].to_numpy(), umbral)
    tr = np.where(part == "entrenamiento")[0]
    nn_train = []
    for ini in range(0, len(tr), 512):
        filas = tr[ini : ini + 512]
        cos, _ = similitudes_bloque(car, filas, tr)
        cos[np.arange(len(filas)), ini + np.arange(len(filas))] = -9
        nn_train.append(cos.max(1))
    nn_train = np.concatenate(nn_train)

    conductores = {
        p: sorted(subject[part == p].dropna().unique().tolist()) for p in PARTICIONES
    }
    tam = pd.Series(grupo).value_counts()
    resumen = {
        "umbral_coseno_centrado": umbral,
        "grupos": int(tam.size),
        "grupos_por_procedencia": pd.Series(df.groupby("procedencia")["grupo"].nunique()).to_dict(),
        "tamano_grupo_maximo": int(tam.max()),
        "conductores_state_farm_por_particion": {p: len(v) for p, v in conductores.items()},
        "conductores_state_farm": conductores,
        "imagenes_por_particion": df["particion"].value_counts().reindex(PARTICIONES).astype(int).to_dict(),
        "imagenes_por_particion_y_procedencia": pd.crosstab(df["particion"], df["procedencia"]).reindex(PARTICIONES).to_dict(orient="index"),
        "proporcion_clase_por_particion": pd.crosstab(df["particion"], df["clase"], normalize="index")
        .reindex(index=PARTICIONES, columns=CLASES).round(4).to_dict(orient="index"),
        "verificacion": rep,
    }
    with open(proc / "verificacion_fuga.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    figuras(cfg, df, dist, nn_train)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))

    total = sum(v["imagenes_con_par_sobre_umbral"] + v["grupos_compartidos"] + v["md5_compartidos"] for v in rep.values())
    if total:
        raise AssertionError(f"Hay {total} casos de fuga entre particiones. Revisa verificacion_fuga.json")
    print("OK: ningún conductor, grupo, MD5 ni casi duplicado compartido entre particiones.")
    return resumen
