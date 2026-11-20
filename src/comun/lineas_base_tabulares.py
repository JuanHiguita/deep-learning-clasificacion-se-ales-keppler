"""Líneas base tabulares sin columnas con fuga: S3-B, S3-C y koi_score como comparador (spec 001, RF-10).

Se evalúa solo en ejemplos 'transito' no prerregistrados, con las particiones de ejemplos.csv.
  (i)  'seleccion': el modelo se entrena solo con la selección del curso.
  (ii) 'tabla_completa': se entrena con todos los KOIs etiquetados de DR25, sin las estrellas de
       prueba ni de validación interna y sin prerregistrados.
koi_score NO es una entrada: se reporta su AUC directo como comparador externo.

Uso:
  python -I src/comun/lineas_base_tabulares.py --ejemplos compartido/ejemplos.csv \
      --koi data/koi_dr25.csv --acumulada data/koi_cumulative.csv --salida resultados/dl/lineas_base_tabulares
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from comun import estadistica, metricas
from comun.construir_ejemplos import CLASE, marcar_prerregistrados
from comun.protocolo import cargar_protocolo, separar
from comun.utilidades import escribir_manifiesto

S3B = ["koi_period", "koi_duration", "koi_depth", "koi_model_snr", "koi_num_transits", "koi_kepmag"]
S3C = S3B + ["koi_dikco_msky", "koi_dicco_msky"]


def _comprobar_sin_fuga(columnas: list[str], protocolo: dict) -> None:
    prohibidas = set(protocolo["columnas_prohibidas_como_entrada"])
    malas = [c for c in columnas if c in prohibidas or c.startswith("koi_fpflag")]
    if malas:
        raise ValueError(f"FUGA: columnas prohibidas como entrada: {malas}")


def predecir_oof(
    sel: pd.DataFrame, columnas: list[str], protocolo: dict, extra: pd.DataFrame | None = None
) -> np.ndarray:
    """P(planeta) fuera de muestra para cada fila de `sel` (KOIs del curso, una fila por KOI)."""
    _comprobar_sin_fuga(columnas, protocolo)
    n = protocolo["particiones"]["n"]
    p = np.full(len(sel), np.nan)
    y = (sel["clase"].values == 0).astype(int)
    for k in range(n):
        sep = separar(sel["particion"].values, k, n)
        X_ent, y_ent = sel.loc[sep.entrenamiento, columnas], y[sep.entrenamiento]
        if extra is not None:
            fuera = set(sel.loc[sep.prueba | sep.validacion, "kepid"])
            ex = extra[~extra["kepid"].isin(fuera)]
            X_ent = pd.concat([X_ent, ex[columnas]], ignore_index=True)
            y_ent = np.concatenate([y_ent, (ex["clase"].values == 0).astype(int)])
        # Con pocas filas, el mínimo por hoja por defecto (20) impide cualquier división.
        hoja = max(2, min(20, len(y_ent) // 20))
        modelo = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, min_samples_leaf=hoja, random_state=0
        )
        modelo.fit(X_ent, y_ent)  # los NaN se dejan como NaN: el modelo los maneja
        p[sep.prueba] = modelo.predict_proba(sel.loc[sep.prueba, columnas])[:, 1]
    return p


def evaluar(puntaje: np.ndarray, sel: pd.DataFrame, protocolo: dict) -> dict:
    ok = np.isfinite(puntaje)
    s, pos, grupos = puntaje[ok], (sel["clase"].values == 0)[ok], sel["kepid"].values[ok]
    cfg = protocolo["metricas"]
    r = estadistica.bootstrap_por_estrella(
        grupos,
        lambda i: metricas.auc(s[i], pos[i]),
        cfg["bootstrap_remuestreos"],
        cfg["bootstrap_semilla"],
        tuple(cfg["ic_percentiles"]),
    )
    return {
        "auc": round(r.estimacion, 4),
        "ic_inf": round(r.ic_inf, 4),
        "ic_sup": round(r.ic_sup, 4),
        "n": int(ok.sum()),
        "n_sin_valor": int((~ok).sum()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ejemplos", type=Path, required=True)
    ap.add_argument("--koi", type=Path, required=True)
    ap.add_argument("--acumulada", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--protocolo", type=Path, default=Path("config/protocolo.json"))
    a = ap.parse_args(argv)

    protocolo = cargar_protocolo(a.protocolo)
    ejemplos = pd.read_csv(a.ejemplos)
    dr25 = pd.read_csv(a.koi)
    base = ejemplos[(ejemplos["tipo"] == "transito") & ~ejemplos["prerregistrado"].astype(bool)]
    sel = (
        base[["kepoi_name", "kepid", "clase", "particion"]]
        .merge(dr25.drop(columns=["kepid"]), on="kepoi_name", how="left", validate="one_to_one")
        .reset_index(drop=True)
    )

    extra = dr25[dr25["koi_disposition"].isin(CLASE)].copy()
    extra["clase"] = extra["koi_disposition"].map(CLASE)
    extra = extra[~marcar_prerregistrados(extra, pd.read_csv(a.acumulada)).values]
    extra = extra[~extra["kepoi_name"].isin(set(ejemplos["kepoi_name"]))]

    R = {
        "n_kois_evaluados": int(len(sel)),
        "n_kois_extra_tabla_completa": int(len(extra)),
        "modelos": {},
    }
    for nombre, columnas in (("S3-B", S3B), ("S3-C", S3C)):
        R["modelos"][f"{nombre} (seleccion)"] = evaluar(
            predecir_oof(sel, columnas, protocolo), sel, protocolo
        )
        R["modelos"][f"{nombre} (tabla_completa)"] = evaluar(
            predecir_oof(sel, columnas, protocolo, extra), sel, protocolo
        )
    R["modelos"]["koi_score (comparador externo, contiene la decisión del Robovetter)"] = evaluar(
        sel["koi_score"].values.astype(float), sel, protocolo
    )

    a.salida.mkdir(parents=True, exist_ok=True)
    with open(a.salida / "metricas.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(R, f, ensure_ascii=False, indent=1)
    escribir_manifiesto(
        a.salida / "manifiesto.json",
        vars(a),
        entradas=[a.ejemplos, a.koi, a.acumulada, a.protocolo],
    )
    print(json.dumps(R, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
