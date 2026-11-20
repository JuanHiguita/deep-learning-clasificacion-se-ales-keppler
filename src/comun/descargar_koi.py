"""Descarga una tabla KOI del NASA Exoplanet Archive por TAP (spec 001, RF-1 a RF-3).

Uso:
  python -I src/comun/descargar_koi.py --tabla q1_q17_dr25_koi --salida data/koi_dr25.csv
  python -I src/comun/descargar_koi.py --tabla cumulative --salida data/koi_cumulative.csv
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.parse
from pathlib import Path

import pandas as pd

from comun.utilidades import escribir_manifiesto, leer_url, sha256_archivo

TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
COLUMNAS = {
    "q1_q17_dr25_koi": (
        "kepid", "kepoi_name", "koi_disposition", "koi_pdisposition", "koi_score",
        "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
        "koi_period", "koi_time0bk", "koi_duration", "koi_depth", "koi_model_snr",
        "koi_max_mult_ev", "koi_num_transits", "koi_kepmag", "koi_dikco_msky", "koi_dicco_msky",
    ),
    # De la tabla acumulada solo hace falta la disposición actual (para el prerregistro).
    "cumulative": ("kepid", "kepoi_name", "koi_disposition"),
}


def construir_consulta(tabla: str) -> str:
    """ADQL con tabla y columnas de una lista cerrada: no admite texto libre."""
    if tabla not in COLUMNAS:
        raise ValueError(f"Tabla no permitida: {tabla!r}")
    return f"select {','.join(COLUMNAS[tabla])} from {tabla}"


def construir_url(tabla: str) -> str:
    return TAP + "?" + urllib.parse.urlencode({"query": construir_consulta(tabla), "format": "csv"})


def comprobar_columnas(df: pd.DataFrame, tabla: str) -> None:
    faltan = [c for c in COLUMNAS[tabla] if c not in df.columns]
    if faltan:
        raise KeyError(f"La tabla {tabla} no devolvió las columnas: {faltan}")
    if len(df) < 1000:
        raise ValueError(f"La tabla {tabla} devolvió solo {len(df)} filas; se esperaban miles")


def descargar(tabla: str, salida: Path) -> pd.DataFrame:
    url = construir_url(tabla)
    contenido = leer_url(url)
    df = pd.read_csv(io.BytesIO(contenido))
    comprobar_columnas(df, tabla)
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_bytes(contenido)
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tabla", choices=sorted(COLUMNAS), default="q1_q17_dr25_koi")
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--manifiesto", type=Path, default=None, help="por defecto: <salida>.manifiesto.json")
    a = ap.parse_args(argv)
    df = descargar(a.tabla, a.salida)
    manifiesto = a.manifiesto or a.salida.with_suffix(".manifiesto.json")
    escribir_manifiesto(manifiesto, vars(a), extra={
        "tabla": a.tabla, "consulta": construir_consulta(a.tabla), "url": construir_url(a.tabla),
        "filas": int(len(df)), "sha256": sha256_archivo(a.salida),
    })
    print(f"{a.tabla}: {len(df)} filas → {a.salida}")
    print(df["koi_disposition"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
