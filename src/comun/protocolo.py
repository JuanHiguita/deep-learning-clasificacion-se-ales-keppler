"""Protocolo experimental: carga de la configuración y separación por partición.

Regla central (constitución, II y III): para la partición de prueba k, la validación
interna es la partición (k + 1) mod n y el entrenamiento son las restantes. Ninguna
estrella (kepid) puede aparecer en dos de los tres conjuntos.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RUTA_POR_DEFECTO = Path("config") / "protocolo.json"


def cargar_protocolo(ruta: str | Path = RUTA_POR_DEFECTO) -> dict:
    """Lee el protocolo fijo del proyecto. Falla si faltan claves obligatorias."""
    with open(ruta, encoding="utf-8") as f:
        protocolo = json.load(f)
    for clave in ("particiones", "semillas", "entrenamiento", "formas", "metricas"):
        if clave not in protocolo:
            raise KeyError(f"El protocolo no tiene la clave obligatoria '{clave}'")
    return protocolo


@dataclass(frozen=True)
class Separacion:
    """Máscaras booleanas de una partición de prueba."""

    k: int
    entrenamiento: np.ndarray
    validacion: np.ndarray
    prueba: np.ndarray


def separar(
    particion: np.ndarray,
    k: int,
    n: int = 5,
    prerregistrado: np.ndarray | None = None,
) -> Separacion:
    """Devuelve las máscaras de entrenamiento, validación interna y prueba.

    particion: partición de prueba de cada ejemplo (enteros 0..n-1).
    prerregistrado: si se da, esos ejemplos quedan fuera de los tres conjuntos.
    """
    particion = np.asarray(particion)
    if not 0 <= k < n:
        raise ValueError(f"k debe estar entre 0 y {n - 1}")
    valores = set(np.unique(particion).tolist())
    if not valores <= set(range(n)):
        raise ValueError(f"Particiones fuera de rango: {sorted(valores)}")
    k_val = (k + 1) % n
    usable = np.ones(len(particion), dtype=bool)
    if prerregistrado is not None:
        usable &= ~np.asarray(prerregistrado, dtype=bool)
    prueba = (particion == k) & usable
    validacion = (particion == k_val) & usable
    entrenamiento = (particion != k) & (particion != k_val) & usable
    return Separacion(k, entrenamiento, validacion, prueba)


def comprobar_sin_solape(kepid: np.ndarray, sep: Separacion) -> None:
    """Lanza un error si una estrella aparece en más de un conjunto."""
    kepid = np.asarray(kepid)
    conjuntos = {
        "entrenamiento": set(kepid[sep.entrenamiento].tolist()),
        "validacion": set(kepid[sep.validacion].tolist()),
        "prueba": set(kepid[sep.prueba].tolist()),
    }
    nombres = list(conjuntos)
    for i, a in enumerate(nombres):
        for b in nombres[i + 1:]:
            comunes = conjuntos[a] & conjuntos[b]
            if comunes:
                muestra = sorted(comunes)[:5]
                raise ValueError(
                    f"Fuga: {len(comunes)} estrellas en {a} y {b} (partición {sep.k}); ej. {muestra}"
                )
