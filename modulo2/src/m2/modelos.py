"""Arquitecturas del Módulo 2: CNN desde cero y EfficientNet-B0 para ajuste fino."""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from .inventario import CLASES


class CNNDesdeCero(nn.Module):
    """Bloques [Conv3x3–BN–ReLU] x 2 + MaxPool 2x2; filtros que se duplican en cada bloque.

    Entrada 3 x 224 x 224. Tras b bloques, el mapa queda de (224 / 2^b) x (224 / 2^b).
    Cabeza: Global Average Pooling -> Dropout -> Lineal(5).
    """

    def __init__(self, bloques: int = 4, filtros_base: int = 32, dropout: float = 0.3, n_clases: int = len(CLASES)):
        super().__init__()
        capas, c_in = [], 3
        for i in range(bloques):
            c_out = filtros_base * 2**i
            capas += [
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*capas)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, n_clases))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def efficientnet_transfer(dropout: float = 0.3, n_clases: int = len(CLASES), preentrenado: bool = True) -> nn.Module:
    """EfficientNet-B0 preentrenada en ImageNet con una cabeza nueva de 5 clases.

    `features` tiene 9 bloques (0 = convolución inicial, 1-7 = bloques MBConv, 8 = conv 1x1 final).
    """
    pesos = models.EfficientNet_B0_Weights.DEFAULT if preentrenado else None
    m = models.efficientnet_b0(weights=pesos)
    m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, n_clases))
    return m


def construir(rcfg: dict) -> nn.Module:
    if rcfg["familia"] == "cnn":
        c = rcfg["cnn"]
        return CNNDesdeCero(c["bloques"], c["filtros_base"], rcfg["dropout"])
    if rcfg["familia"] == "transfer":
        return efficientnet_transfer(rcfg["dropout"], preentrenado=rcfg["transfer"].get("preentrenado", True))
    raise ValueError(f"familia desconocida: {rcfg['familia']}")


def grupos_parametros(modelo: nn.Module, rcfg: dict) -> tuple[list[dict], list[nn.Module], list[nn.Module]]:
    """Grupos del optimizador, bloques siempre congelados y bloques que se descongelan.

    CNN: un solo grupo, todo entrenable.
    Transfer: grupo «cabeza» (lr) + grupo «base» con los últimos `bloques_descongelados`
    bloques de `features` (lr * factor_lr_base). Durante las primeras `epocas_cabeza`
    épocas la base no recibe gradiente (requires_grad=False) y su BatchNorm va en modo eval.
    """
    lr = rcfg["lr"]
    if rcfg["familia"] == "cnn":
        return [{"params": list(modelo.parameters()), "lr": lr, "nombre": "todo"}], [], []
    t = rcfg["transfer"]
    k = t["bloques_descongelados"]
    bloques = list(modelo.features.children())
    base_entrenable = bloques[len(bloques) - k :] if k > 0 else []
    congelados_siempre = bloques[: len(bloques) - k]
    for b in bloques:
        for p in b.parameters():
            p.requires_grad_(False)
    grupos = [{"params": list(modelo.classifier.parameters()), "lr": lr, "nombre": "cabeza"}]
    if base_entrenable:
        grupos.append(
            {"params": [p for b in base_entrenable for p in b.parameters()], "lr": lr * t["factor_lr_base"], "nombre": "base"}
        )
    return grupos, congelados_siempre, base_entrenable


def parametros_entrenables(modelo: nn.Module, rcfg: dict) -> dict:
    total = sum(p.numel() for p in modelo.parameters())
    if rcfg["familia"] == "cnn":
        return {"entrenables": total, "totales": total}
    k = rcfg["transfer"]["bloques_descongelados"]
    bloques = list(modelo.features.children())
    base = sum(p.numel() for b in bloques[len(bloques) - k :] for p in b.parameters()) if k > 0 else 0
    cabeza = sum(p.numel() for p in modelo.classifier.parameters())
    return {"entrenables": base + cabeza, "totales": total, "cabeza": cabeza, "base_descongelada": base}
