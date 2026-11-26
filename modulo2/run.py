"""Punto de entrada único del Módulo 2.

Uso (en Windows, `py` en lugar de `python`):
  py run.py inventario
  py run.py duplicados
  py run.py particion [--set particion.umbral_coseno=0.80]
  py run.py preproceso
  py run.py datos            # inventario + duplicados + particion + preproceso
  py run.py baseline         # modelo de referencia (logística sobre red congelada)
  py run.py entrenar [--set red.familia=cnn --set red.lr=0.003]   # una corrida (modificación en vivo)
  py run.py busqueda --set red.familia=transfer                   # búsqueda con presupuesto fijo
  py run.py semillas --set red.familia=transfer                   # 30 semillas de la mejor configuración
  py run.py figuras                                               # figuras de comparación (sin reentrenar)
  py run.py evaluacion                                            # Grad-CAM, alertas, por grupo, arquitecturas

Opciones comunes:
  --config configs/m2.yaml       archivo de configuración
  --set clave.subclave=valor     cambia cualquier parámetro sin editar el YAML (repetible)
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from m2.utils import entorno, load_config, resolve, set_seed  # noqa: E402

PASOS = ["inventario", "duplicados", "particion", "preproceso", "baseline", "entrenar", "busqueda", "semillas", "figuras", "evaluacion"]
COMPUESTOS = {"datos": ["inventario", "duplicados", "particion", "preproceso"]}
# Pasos que viven dentro de otro módulo: paso -> (módulo, función)
FUNCIONES = {"entrenar": ("redes", "entrenar"), "busqueda": ("redes", "busqueda"), "semillas": ("redes", "semillas"),
             "figuras": ("figuras_redes", "ejecutar")}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Módulo 2: conducción distractiva")
    ap.add_argument("paso", choices=PASOS + list(COMPUESTOS))
    ap.add_argument("--config", default="configs/m2.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    print(f"Entorno: {json.dumps(entorno())}")

    for paso in COMPUESTOS.get(args.paso, [args.paso]):
        t0 = time.time()
        print(f"\n=== {paso} ===")
        modulo, funcion = FUNCIONES.get(paso, (paso, "ejecutar"))
        getattr(importlib.import_module(f"m2.{modulo}"), funcion)(cfg)
        print(f"[{paso}] {time.time() - t0:.1f} s")

    # Deja la configuración efectiva junto a los resultados, para reproducibilidad
    proc = resolve(cfg["datos"]["processed_dir"])
    proc.mkdir(parents=True, exist_ok=True)
    with open(proc / f"config_efectiva_{args.paso}.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "entorno": entorno(), "overrides": args.set}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":  # necesario en Windows para DataLoader con num_workers > 0
    main()
