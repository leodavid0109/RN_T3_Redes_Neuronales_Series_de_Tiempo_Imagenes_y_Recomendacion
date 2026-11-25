"""Punto de entrada único del Módulo 2.

Uso:
  python run.py inventario
  python run.py duplicados
  python run.py particion [--set particion.umbral_coseno=0.93]
  python run.py datos            # los tres pasos anteriores en orden

Opciones comunes:
  --config configs/m2.yaml       archivo de configuración
  --set clave.subclave=valor     cambia cualquier parámetro sin editar el YAML (repetible)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from m2.utils import entorno, load_config, resolve, set_seed  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="Módulo 2: conducción distractiva")
    ap.add_argument("paso", choices=["inventario", "duplicados", "particion", "datos"])
    ap.add_argument("--config", default="configs/m2.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    print(f"Entorno: {json.dumps(entorno())}")

    pasos = ["inventario", "duplicados", "particion"] if args.paso == "datos" else [args.paso]
    for paso in pasos:
        t0 = time.time()
        print(f"\n=== {paso} ===")
        if paso == "inventario":
            from m2 import inventario

            inventario.ejecutar(cfg)
        elif paso == "duplicados":
            from m2 import duplicados

            duplicados.ejecutar(cfg)
        elif paso == "particion":
            from m2 import particion

            particion.ejecutar(cfg)
        print(f"[{paso}] {time.time() - t0:.1f} s")

    # Deja la configuración efectiva junto a los resultados, para reproducibilidad
    proc = resolve(cfg["datos"]["processed_dir"])
    proc.mkdir(parents=True, exist_ok=True)
    with open(proc / f"config_efectiva_{args.paso}.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "entorno": entorno(), "overrides": args.set}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":  # necesario en Windows para DataLoader con num_workers > 0
    main()
