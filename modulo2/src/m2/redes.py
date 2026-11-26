"""Pasos 6-8: redes neuronales (CNN desde cero y transfer learning con EfficientNet-B0).

  py run.py entrenar  [--set red.familia=cnn --set red.lr=0.003 ...]
      Una corrida con la configuración `red` del YAML (más los cambios de la CLI).
      Sirve para la modificación en vivo de la sustentación.
  py run.py busqueda  --set red.familia=transfer
      Búsqueda aleatoria con presupuesto fijo (busqueda.presupuesto configuraciones),
      semilla de búsqueda 42, criterio F1 macro en validación. Guarda la mejor
      configuración en resultados/mejor_config_<familia>.yaml.
  py run.py semillas  --set red.familia=transfer
      Entrena la mejor configuración con las semillas 0..29. Se puede interrumpir y
      reanudar: las semillas ya hechas no se repiten. El modelo de la semilla 0 se
      guarda como modelo final (elegido de antemano, no por su resultado).
"""
from __future__ import annotations

import copy
import json
import time

import numpy as np
import pandas as pd
import torch
import yaml

from .datos_torch import cargar_splits
from .entrenamiento import entrenar_una
from .inventario import CLASES
from .utils import entorno, resolve


def _set(d: dict, clave: str, valor):
    # En el espacio de búsqueda las claves anidadas se escriben con "/" (p. ej. cnn/bloques)
    # para no chocar con los "." que usa --set para navegar el YAML.
    partes = clave.replace("/", ".").split(".")
    for p in partes[:-1]:
        d = d.setdefault(p, {})
    d[partes[-1]] = valor


def _dirs(cfg):
    out = {k: resolve(cfg["salidas"][k]) for k in ("resultados", "modelos", "figuras")}
    out["curvas"] = out["resultados"] / "curvas"
    for d in out.values():
        d.mkdir(parents=True, exist_ok=True)
    return out


def _fila(res: dict) -> dict:
    return {
        "f1_macro_validacion": res["validacion"]["f1_macro"],
        "exactitud_validacion": res["validacion"]["exactitud"],
        "f1_macro_prueba": res["prueba"]["f1_macro"],
        "exactitud_prueba": res["prueba"]["exactitud"],
        **{f"f1_prueba_{k}": v["f1_macro"] for k, v in res["prueba_por_procedencia"].items()},
        **{f"exactitud_prueba_{k}": v["exactitud"] for k, v in res["prueba_por_procedencia"].items()},
        "mejor_epoca": res["mejor_epoca"],
        "epocas_entrenadas": res["epocas_entrenadas"],
        "tiempo_s": res["tiempo_s"],
        "parametros_entrenables": res["parametros"]["entrenables"],
        "parametros_totales": res["parametros"]["totales"],
    }


def config_final(cfg: dict, familia: str) -> dict:
    """Configuración ganadora de la búsqueda si existe; si no, la del YAML."""
    ruta = resolve(cfg["salidas"]["resultados"]) / f"mejor_config_{familia}.yaml"
    if ruta.exists():
        with open(ruta, encoding="utf-8") as f:
            return yaml.safe_load(f)
    print(f"AVISO: no existe {ruta.name}; se usa la configuración 'red' del YAML.")
    rcfg = copy.deepcopy(cfg["red"])
    rcfg["familia"] = familia
    return rcfg


def entrenar(cfg: dict) -> dict:
    d = _dirs(cfg)
    df = cargar_splits(cfg)
    rcfg = copy.deepcopy(cfg["red"])
    print(f"Configuración: {json.dumps(rcfg, ensure_ascii=False)}")
    res = entrenar_una(cfg, rcfg, df, cfg["seed"])
    marca = time.strftime("%Y%m%d_%H%M%S")
    res["curvas"].to_csv(d["curvas"] / f"entrenar_{rcfg['familia']}_{marca}.csv", index=False)
    salida = {"config": rcfg, "semilla": cfg["seed"], **_fila(res), "prueba_por_clase": res["prueba"]["por_clase"],
              "matriz_confusion_prueba": res["prueba"]["matriz_confusion"], "entorno": entorno()}
    with open(d["resultados"] / f"entrenar_{rcfg['familia']}_{marca}.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in salida.items()
                      if k not in ("config", "prueba_por_clase", "entorno", "matriz_confusion_prueba")}, indent=2))
    return salida


def busqueda(cfg: dict) -> pd.DataFrame:
    d = _dirs(cfg)
    df = cargar_splits(cfg)
    familia = cfg["red"]["familia"]
    bcfg = cfg["busqueda"]
    espacio = bcfg["espacio"][familia]
    rng = np.random.default_rng(bcfg["semilla"])
    ruta_csv = d["resultados"] / f"busqueda_{familia}.csv"
    hechas = pd.read_csv(ruta_csv) if ruta_csv.exists() else pd.DataFrame()

    filas = hechas.to_dict("records")
    t0 = time.time()
    for k in range(bcfg["presupuesto"]):
        # Se sortean todas las configuraciones aunque se reanude, para que la k-ésima sea siempre la misma
        rcfg = copy.deepcopy(cfg["red"])
        rcfg["epocas_max"] = bcfg["epocas_max"]
        rcfg["paciencia"] = bcfg["paciencia"]
        elegidos = {}
        for clave, opciones in espacio.items():
            valor = opciones[rng.integers(len(opciones))]
            if clave.startswith("_"):
                # Opción combinada: un diccionario de varias claves que se eligen juntas
                for k2, v2 in valor.items():
                    _set(rcfg, k2, copy.deepcopy(v2))
            else:
                _set(rcfg, clave, copy.deepcopy(valor))
            elegidos[clave] = valor
        if len(hechas) and k in set(hechas["config"]):
            continue
        print(f"\n[{familia}] configuración {k + 1}/{bcfg['presupuesto']}: {json.dumps(elegidos, ensure_ascii=False)}")
        res = entrenar_una(cfg, rcfg, df, bcfg["semilla"])
        res["curvas"].to_csv(d["curvas"] / f"busqueda_{familia}_{k:02d}.csv", index=False)
        filas.append({"config": k, **{f"hp_{c}": json.dumps(v) for c, v in elegidos.items()}, **_fila(res),
                      "rcfg": json.dumps(rcfg, ensure_ascii=False)})
        pd.DataFrame(filas).to_csv(ruta_csv, index=False)
        del res
        torch.cuda.empty_cache()
    tabla = pd.DataFrame(filas).sort_values("config")
    tabla.to_csv(ruta_csv, index=False)
    mejor = tabla.loc[tabla["f1_macro_validacion"].idxmax()]
    rcfg = json.loads(mejor["rcfg"])
    rcfg["epocas_max"], rcfg["paciencia"] = cfg["red"]["epocas_max"], cfg["red"]["paciencia"]
    with open(d["resultados"] / f"mejor_config_{familia}.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(rcfg, f, allow_unicode=True, sort_keys=False)
    print(f"\nBúsqueda {familia}: {len(tabla)} configuraciones, {(time.time() - t0) / 60:.1f} min en esta ejecución, "
          f"{tabla['tiempo_s'].sum() / 60:.1f} min de entrenamiento en total.")
    print(f"Mejor: configuración {int(mejor['config'])}, F1 macro val {mejor['f1_macro_validacion']:.4f}")
    return tabla


def semillas(cfg: dict) -> pd.DataFrame:
    d = _dirs(cfg)
    df = cargar_splits(cfg)
    familia = cfg["red"]["familia"]
    rcfg = config_final(cfg, familia)
    scfg = cfg["semillas"]
    ruta_csv = d["resultados"] / f"semillas_{familia}.csv"
    ruta_prob = d["resultados"] / f"prob_prueba_{familia}.npz"
    filas = pd.read_csv(ruta_csv).to_dict("records") if ruta_csv.exists() else []
    probs = dict(np.load(ruta_prob)) if ruta_prob.exists() else {}
    hechas = {int(f["semilla"]) for f in filas}
    print(f"Configuración final ({familia}): {json.dumps(rcfg, ensure_ascii=False)}")
    for s in range(scfg["inicio"], scfg["inicio"] + scfg["n"]):
        if s in hechas:
            continue
        print(f"\n[{familia}] semilla {s}")
        res = entrenar_una(cfg, rcfg, df, s, verbose=scfg.get("verbose", False))
        filas.append({"semilla": s, **_fila(res)})
        probs[f"semilla_{s}"] = res["prob_prueba"].astype(np.float32)
        print(f"  F1 val {res['validacion']['f1_macro']:.4f} | F1 prueba {res['prueba']['f1_macro']:.4f} | "
              f"época {res['mejor_epoca']}/{res['epocas_entrenadas']} | {res['tiempo_s']:.0f} s")
        if s == scfg["inicio"]:
            res["curvas"].to_csv(d["curvas"] / f"final_{familia}_semilla{s}.csv", index=False)
            torch.save({"estado": res["modelo"].state_dict(), "rcfg": rcfg, "clases": CLASES, "semilla": s},
                       d["modelos"] / f"{familia}_final.pt")
            with open(d["resultados"] / f"final_{familia}_metricas.json", "w", encoding="utf-8") as f:
                json.dump({"semilla": s, "config": rcfg, "parametros": res["parametros"], "validacion": res["validacion"],
                           "prueba": res["prueba"], "prueba_por_procedencia": res["prueba_por_procedencia"]},
                          f, indent=2, ensure_ascii=False)
        pd.DataFrame(filas).sort_values("semilla").to_csv(ruta_csv, index=False)
        np.savez_compressed(ruta_prob, **probs)
        del res
        torch.cuda.empty_cache()
    tabla = pd.DataFrame(filas).sort_values("semilla")
    cols = [c for c in tabla.columns if c.startswith(("f1_", "exactitud_"))]
    print("\n" + tabla[cols].describe().loc[["mean", "std", "min", "max"]].round(4).to_string())
    return tabla
