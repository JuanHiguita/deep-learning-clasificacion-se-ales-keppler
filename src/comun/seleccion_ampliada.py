"""Selección ampliada de señales para la iteración 2 (spec 014).

Toma de la tabla DR25 TODAS las señales con etiqueta definitiva (CONFIRMED o FALSE POSITIVE) y
periodo dentro de un rango, en vez de las 260 de la demo. No usa ninguna columna prohibida para
elegir: solo la disposición final y el periodo.

Uso:
  python -I src/comun/seleccion_ampliada.py --koi data/koi_dr25.csv \
      --salida iteracion_2/datos/seleccion.csv [--periodo-min 1 --periodo-max 30]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

COLUMNAS = [
    "kepid",
    "kepoi_name",
    "koi_disposition",
    "koi_period",
    "koi_time0bk",
    "koi_duration",
    "koi_depth",
    "koi_max_mult_ev",
    "koi_kepmag",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "koi_dikco_msky",
    "koi_dicco_msky",
]
ETIQUETAS_DEFINITIVAS = ("CONFIRMED", "FALSE POSITIVE")


def seleccionar(koi: pd.DataFrame, periodo_min: float, periodo_max: float) -> pd.DataFrame:
    """Señales con etiqueta definitiva, periodo en rango y efemérides completas."""
    con_etiqueta = koi["koi_disposition"].isin(ETIQUETAS_DEFINITIVAS)
    en_rango = koi["koi_period"].between(periodo_min, periodo_max)
    con_efemerides = koi[["koi_period", "koi_time0bk", "koi_duration"]].notna().all(axis=1)
    seleccion = koi.loc[con_etiqueta & en_rango & con_efemerides, COLUMNAS]
    return seleccion.sort_values("kepoi_name").reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--koi", type=Path, required=True, help="tabla q1_q17_dr25_koi")
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--periodo-min", type=float, default=1.0)
    ap.add_argument("--periodo-max", type=float, default=30.0)
    a = ap.parse_args(argv)
    seleccion = seleccionar(pd.read_csv(a.koi), a.periodo_min, a.periodo_max)
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    seleccion.to_csv(a.salida, index=False, lineterminator="\n")
    conteo = seleccion["koi_disposition"].value_counts().to_dict()
    print(f"{len(seleccion)} señales de {seleccion['kepid'].nunique()} estrellas: {conteo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
