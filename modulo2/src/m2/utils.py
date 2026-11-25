"""Utilidades comunes del Módulo 2: configuración, semillas y rutas."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml

# Carpeta raíz del módulo (modulo2/)
ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    """Rutas relativas se interpretan desde modulo2/; las absolutas se respetan."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Carga el YAML y aplica cambios de la forma clave.subclave=valor (desde la CLI)."""
    with open(resolve(path), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override inválido '{ov}'. Usa clave.subclave=valor")
        key, val = ov.split("=", 1)
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(val)
    return cfg


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fija todas las fuentes de aleatoriedad (Python, NumPy, PyTorch CPU/GPU)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Necesario para que cuBLAS sea determinista; debe fijarse antes de usar CUDA.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(pref: str = "auto"):
    import torch

    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def entorno() -> dict:
    """Versiones y hardware, para dejar constancia en los resultados."""
    info = {"numpy": np.__version__}
    try:
        import torch
        import torchvision

        info.update(
            torch=torch.__version__,
            torchvision=torchvision.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        )
    except ImportError:
        pass
    return info
