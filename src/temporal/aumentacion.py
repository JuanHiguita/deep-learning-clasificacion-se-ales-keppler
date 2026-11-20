"""Aumentación de datos, solo para el entrenamiento (spec 004, RF-1). Valores en config/protocolo.json."""

from __future__ import annotations

import numpy as np

CFG = {
    "ruido_sigma": 0.05,
    "desplazamiento_max_puntos": 2,
    "prob_conservar_transito": 0.8,
    "min_transitos": 4,
}


def _desplazar(x: np.ndarray, pasos: np.ndarray, eje: int = 1) -> np.ndarray:
    return np.stack([np.roll(fila, int(p), axis=eje - 1) for fila, p in zip(x, pasos, strict=True)])


def aumentar_vistas(entradas: dict, rng: np.random.Generator, cfg: dict = CFG) -> dict:
    """Ruido gaussiano en las dos vistas y desplazamiento de ±k puntos en la vista local."""
    s = dict(entradas)
    g, loc = entradas["vista_global"], entradas["vista_local"]
    k = cfg["desplazamiento_max_puntos"]
    s["vista_global"] = (g + rng.normal(0, cfg["ruido_sigma"], g.shape)).astype(np.float32)
    loc = _desplazar(loc, rng.integers(-k, k + 1, len(loc)))
    s["vista_local"] = (loc + rng.normal(0, cfg["ruido_sigma"], loc.shape)).astype(np.float32)
    return s


def aumentar_transitos(entradas: dict, rng: np.random.Generator, cfg: dict = CFG) -> dict:
    """Ruido, desplazamiento por ejemplo y eliminación aleatoria de tránsitos (se conserva el orden)."""
    s = dict(entradas)
    tr, mk = entradas["transitos"], entradas["mascara_transitos"]
    k = cfg["desplazamiento_max_puntos"]
    tr = _desplazar(tr, rng.integers(-k, k + 1, len(tr)), eje=2)
    tr = tr + rng.normal(0, cfg["ruido_sigma"], tr.shape)
    conservar = (rng.random(mk.shape) < cfg["prob_conservar_transito"]) & (mk > 0)
    pocos = conservar.sum(axis=1) < np.minimum(cfg["min_transitos"], (mk > 0).sum(axis=1))
    conservar[pocos] = mk[pocos] > 0
    # compactar al inicio, para que los vacíos sigan quedando al final (lo que espera la GRU)
    orden = np.argsort(~conservar, axis=1, kind="stable")
    tr = np.take_along_axis(tr, orden[:, :, None], axis=1)
    nueva = np.take_along_axis(conservar, orden, axis=1).astype(np.float32)
    s["transitos"] = (tr * nueva[:, :, None]).astype(np.float32)
    s["mascara_transitos"] = nueva
    return s
