"""Lectura de las curvas de luz de Kepler (archivos *_llc.fits de MAST). Único punto que abre FITS."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from comun.utilidades import validar_kepid


def leer_llc(ruta: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Tiempo (BKJD) y flujo PDCSAP de un trimestre, solo cadencias con SAP_QUALITY = 0 y valores finitos.

    Exigir calidad 0 es más estricto que la máscara 'default' de lightkurve: descarta cualquier
    cadencia marcada por la misión.
    """
    with fits.open(ruta, memmap=False) as h:
        datos = h[1].data
        t = np.asarray(datos["TIME"], dtype=float)
        f = np.asarray(datos["PDCSAP_FLUX"], dtype=float)
        calidad = np.asarray(datos["SAP_QUALITY"])
    ok = (calidad == 0) & np.isfinite(t) & np.isfinite(f) & (f > 0)
    return t[ok], f[ok]


def archivos_de_estrella(carpeta_llc: str | Path, kepid: int) -> list[Path]:
    """Archivos de cadencia larga de una estrella, en orden cronológico."""
    kic = f"{validar_kepid(kepid):09d}"
    return sorted((Path(carpeta_llc) / kic).glob(f"kplr{kic}-*_llc.fits"))


def trimestres_de_estrella(carpeta_llc: str | Path, kepid: int) -> list[tuple[np.ndarray, np.ndarray]]:
    trimestres = []
    for ruta in archivos_de_estrella(carpeta_llc, kepid):
        t, f = leer_llc(ruta)
        if len(t):
            trimestres.append((t, f))
    return trimestres
