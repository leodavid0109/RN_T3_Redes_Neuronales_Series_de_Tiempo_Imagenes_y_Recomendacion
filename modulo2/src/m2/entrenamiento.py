"""Entrenamiento de una red con parada temprana (común a la CNN y al transfer learning)."""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

from . import metricas
from .datos_torch import cargador, transformacion_entrenamiento, transformacion_evaluacion
from .modelos import construir, grupos_parametros, parametros_entrenables
from .utils import get_device, set_seed


def _optimizador(grupos, rcfg):
    if rcfg["optimizador"] == "adamw":
        return torch.optim.AdamW(grupos, weight_decay=rcfg["weight_decay"])
    if rcfg["optimizador"] == "sgd":
        return torch.optim.SGD(grupos, momentum=rcfg["momento"], nesterov=True, weight_decay=rcfg["weight_decay"])
    raise ValueError(f"optimizador desconocido: {rcfg['optimizador']}")


@torch.no_grad()
def predecir(modelo, dl, dev, amp_dtype=None) -> tuple[np.ndarray, float]:
    """Probabilidades softmax y pérdida media de entropía cruzada."""
    modelo.eval()
    probs, perdida, n = [], 0.0, 0
    for x, y in dl:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            z = modelo(x)
        z = z.float()
        perdida += nn.functional.cross_entropy(z, y, reduction="sum").item()
        n += len(y)
        probs.append(torch.softmax(z, 1).cpu().numpy())
    return np.concatenate(probs), perdida / max(n, 1)


def entrenar_una(cfg: dict, rcfg: dict, df: pd.DataFrame, semilla: int, verbose: bool = True) -> dict:
    """Entrena una red con la configuración `rcfg` y la semilla dada.

    La semilla controla la inicialización de la cabeza o de toda la red, el orden de
    los lotes, el dropout y el aumento de datos. La partición es siempre la misma.
    Devuelve el modelo con los pesos de la mejor época (F1 macro en validación),
    sus curvas y sus probabilidades en validación y prueba.
    """
    set_seed(semilla)
    dev = get_device(cfg.get("dispositivo", "auto"))
    amp = None
    if rcfg.get("precision_mixta", True) and dev.type == "cuda":
        amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    escalador = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)

    tam = cfg["preproceso"]["tamano"]
    gen = torch.Generator().manual_seed(semilla)
    dtr = df[df["particion"] == "entrenamiento"]
    dva = df[df["particion"] == "validacion"]
    dpr = df[df["particion"] == "prueba"]
    dl_tr = cargador(cfg, dtr, transformacion_entrenamiento(rcfg["aumento"], tam), rcfg["batch_size"], barajar=True, generador=gen)
    tf_ev = transformacion_evaluacion(tam)
    dl_va = cargador(cfg, dva, tf_ev, rcfg["batch_size"] * 2)
    dl_pr = cargador(cfg, dpr, tf_ev, rcfg["batch_size"] * 2)

    modelo = construir(rcfg).to(dev)
    grupos, congelados, descongelables = grupos_parametros(modelo, rcfg)
    opt = _optimizador(grupos, rcfg)
    epocas_cabeza = rcfg["transfer"]["epocas_cabeza"] if rcfg["familia"] == "transfer" else 0
    total_pasos = rcfg["epocas_max"] * len(dl_tr)
    # Calentamiento lineal de 1 época y luego decaimiento coseno (por paso)
    calent = len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / calent) * 0.5 * (1 + math.cos(math.pi * min(s, total_pasos) / total_pasos))
    )
    criterio = nn.CrossEntropyLoss()

    y_va = dva["etiqueta"].to_numpy()
    curvas, mejor, mejor_f1, mejor_ep, sin_mejora = [], None, -1.0, -1, 0
    t0 = time.time()
    for ep in range(rcfg["epocas_max"]):
        if ep == epocas_cabeza and descongelables:
            for b in descongelables:
                for p in b.parameters():
                    p.requires_grad_(True)
        modelo.train()
        for b in congelados + (descongelables if ep < epocas_cabeza else []):
            b.eval()  # BatchNorm de los bloques congelados conserva las estadísticas de ImageNet
        perd_tr, aciertos, n = 0.0, 0, 0
        te = time.time()
        for x, y in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
                z = modelo(x)
                loss = criterio(z.float(), y)
            escalador.scale(loss).backward()
            escalador.step(opt)
            escalador.update()
            sched.step()
            perd_tr += loss.item() * len(y)
            aciertos += (z.argmax(1) == y).sum().item()
            n += len(y)
        p_va, perd_va = predecir(modelo, dl_va, dev, amp)
        f1_va = f1_score(y_va, p_va.argmax(1), average="macro")
        curvas.append(
            dict(epoca=ep + 1, perdida_entrenamiento=perd_tr / n, exactitud_entrenamiento=aciertos / n,
                 perdida_validacion=perd_va, f1_macro_validacion=f1_va, segundos=time.time() - te)
        )
        if verbose:
            print(f"    época {ep + 1:2d}  pérdida tr {perd_tr / n:.4f}  val {perd_va:.4f}  F1 val {f1_va:.4f}  ({time.time() - te:.0f} s)")
        if f1_va > mejor_f1:
            mejor_f1, mejor_ep, sin_mejora = f1_va, ep + 1, 0
            mejor = copy.deepcopy(modelo.state_dict())
        else:
            sin_mejora += 1
            if sin_mejora >= rcfg["paciencia"]:
                break
    tiempo = time.time() - t0
    modelo.load_state_dict(mejor)
    p_va, _ = predecir(modelo, dl_va, dev, amp)
    p_pr, _ = predecir(modelo, dl_pr, dev, amp)
    y_pr = dpr["etiqueta"].to_numpy()
    return {
        "modelo": modelo,
        "curvas": pd.DataFrame(curvas),
        "mejor_epoca": mejor_ep,
        "epocas_entrenadas": len(curvas),
        "tiempo_s": round(tiempo, 1),
        "parametros": parametros_entrenables(modelo, rcfg),
        "prob_validacion": p_va,
        "prob_prueba": p_pr,
        "validacion": metricas.calcular(y_va, p_va),
        "prueba": metricas.calcular(y_pr, p_pr),
        "prueba_por_procedencia": metricas.por_grupo(dpr, p_pr, "procedencia"),
    }
