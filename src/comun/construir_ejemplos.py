"""Construye la tabla maestra compartido/ejemplos.csv (spec 001, RF-4 a RF-9).

Dos filas por KOI: '<kepoi_name>|transito' (clase 0 planeta o 1 falso positivo, etiqueta de DR25)
y '<kepoi_name>|sin_senal' (clase 2, misma estrella en una época desplazada).

Uso:
  python -I src/comun/construir_ejemplos.py --seleccion data/seleccion_demo.csv \
      --koi data/koi_dr25.csv --acumulada data/koi_cumulative.csv --salida compartido/ejemplos.csv
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from comun.protocolo import cargar_protocolo
from comun.utilidades import escribir_manifiesto, sha256_archivo, validar_kepid

CLASE = {"CONFIRMED": 0, "FALSE POSITIVE": 1}
COLUMNAS_SALIDA = [
    "id_ejemplo", "kepid", "kepoi_name", "tipo", "clase", "tipo_fp", "periodo_d", "mes", "particion",
    "prerregistrado", "fp_nt", "fp_ss", "fp_co", "fp_ec", "n_koi_en_estrella", "t0_bkjd", "duracion_h",
]
MAX_FRACCION_PRERREGISTRADA = 0.10


def tipo_fp(fila) -> str:
    """Tipo de falso positivo para reportar por estrato. Nunca es una entrada del modelo."""
    if fila["koi_disposition"] != "FALSE POSITIVE":
        return ""
    banderas = [n for n in ("nt", "ss", "co", "ec") if int(fila[f"koi_fpflag_{n}"]) == 1]
    if len(banderas) != 1:
        return "otro"
    return {"nt": "no_transito", "ss": "eclipse_secundario", "co": "centroide", "ec": "otro"}[banderas[0]]


def marcar_prerregistrados(dr25: pd.DataFrame, acumulada: pd.DataFrame) -> pd.Series:
    """Verdadero si la disposición del KOI difiere entre DR25 y la tabla acumulada (spec 001, RF-9)."""
    actual = acumulada.set_index("kepoi_name")["koi_disposition"]
    hoy = dr25["kepoi_name"].map(actual)
    return (hoy.notna() & (hoy != dr25["koi_disposition"])).rename("prerregistrado")


def asignar_particiones(kois: pd.DataFrame, n: int, semilla: int) -> np.ndarray:
    """Particiones agrupadas por estrella y estratificadas por clase, sin depender de ninguna librería.

    Dentro de cada clase, las estrellas se ordenan por SHA-256 de "<semilla>:<kepid>" y se reparten
    por turnos entre las n particiones. El resultado es el mismo en cualquier máquina y versión
    (StratifiedGroupKFold de scikit-learn da repartos distintos, y a veces desbalanceados, según la
    versión). Una estrella con varios KOIs va completa a una partición, con la clase de su primer KOI.
    """
    estrellas = kois.sort_values("kepoi_name").groupby("kepid")["clase"].first()
    asignada, turno = {}, 0
    for clase in sorted(estrellas.unique()):
        de_la_clase = [int(k) for k in estrellas.index[estrellas == clase]]
        de_la_clase.sort(key=lambda k: hashlib.sha256(f"{semilla}:{k}".encode()).hexdigest())
        for kepid in de_la_clase:
            asignada[kepid] = turno % n
            turno += 1
    return kois["kepid"].map(asignada).to_numpy(dtype=int)


def construir(seleccion: pd.DataFrame, dr25: pd.DataFrame, acumulada: pd.DataFrame, protocolo: dict
              ) -> tuple[pd.DataFrame, dict]:
    """Devuelve la tabla de ejemplos y un resumen para el manifiesto."""
    dr25 = dr25.copy()
    dr25["kepid"] = dr25["kepid"].map(validar_kepid)
    dr25["prerregistrado"] = marcar_prerregistrados(dr25, acumulada).values
    n_koi = dr25.groupby("kepid")["kepoi_name"].transform("count")
    dr25["n_koi_en_estrella"] = n_koi

    sel = seleccion[["kepoi_name", "koi_disposition"]].rename(columns={"koi_disposition": "disp_seleccion"})
    kois = sel.merge(dr25, on="kepoi_name", how="left", indicator=True)
    sin_dr25 = kois.loc[kois["_merge"] == "left_only", "kepoi_name"].tolist()
    kois = kois[kois["_merge"] == "both"].drop(columns="_merge")
    no_etiquetados = kois.loc[~kois["koi_disposition"].isin(CLASE), "kepoi_name"].tolist()
    kois = kois[kois["koi_disposition"].isin(CLASE)].copy()
    cambian = kois.loc[kois["koi_disposition"] != kois["disp_seleccion"], "kepoi_name"].tolist()

    for col in ("koi_period", "koi_time0bk", "koi_duration"):
        if kois[col].isna().any():
            raise ValueError(f"Hay KOIs seleccionados sin {col}")

    kois["clase"] = kois["koi_disposition"].map(CLASE).astype(int)
    kois["tipo_fp"] = kois.apply(tipo_fp, axis=1)
    kois = kois.sort_values("kepoi_name").reset_index(drop=True)
    cfg = protocolo["particiones"]
    kois["particion"] = asignar_particiones(kois, cfg["n"], cfg["semilla"])

    fraccion = float(kois["prerregistrado"].mean())
    if fraccion > MAX_FRACCION_PRERREGISTRADA:
        raise RuntimeError(
            f"El prerregistro marca el {fraccion:.1%} de la selección (> {MAX_FRACCION_PRERREGISTRADA:.0%}). "
            "Detenerse y consultar a Juan (spec 001, RF-9)."
        )

    filas = []
    for k in kois.itertuples():
        comun = dict(
            kepid=int(k.kepid), kepoi_name=k.kepoi_name, periodo_d=k.koi_period, mes=k.koi_max_mult_ev,
            particion=int(k.particion), prerregistrado=bool(k.prerregistrado),
            fp_nt=int(k.koi_fpflag_nt), fp_ss=int(k.koi_fpflag_ss), fp_co=int(k.koi_fpflag_co),
            fp_ec=int(k.koi_fpflag_ec), n_koi_en_estrella=int(k.n_koi_en_estrella),
            t0_bkjd=k.koi_time0bk, duracion_h=k.koi_duration,
        )
        filas.append(dict(id_ejemplo=f"{k.kepoi_name}|transito", tipo="transito", clase=int(k.clase),
                          tipo_fp=k.tipo_fp, **comun))
        filas.append(dict(id_ejemplo=f"{k.kepoi_name}|sin_senal", tipo="sin_senal", clase=2, tipo_fp="", **comun))
    ejemplos = pd.DataFrame(filas)[COLUMNAS_SALIDA].sort_values("id_ejemplo").reset_index(drop=True)

    usables = ejemplos[(ejemplos["tipo"] == "transito") & ~ejemplos["prerregistrado"]]
    resumen = {
        "n_kois": int(len(kois)), "n_ejemplos": int(len(ejemplos)), "n_estrellas": int(kois["kepid"].nunique()),
        "por_clase": {str(c): int(n) for c, n in ejemplos["clase"].value_counts().sort_index().items()},
        "por_particion_y_clase": {
            str(p): {str(c): int(n) for c, n in g["clase"].value_counts().sort_index().items()}
            for p, g in ejemplos.groupby("particion")
        },
        "sin_fila_en_dr25": sin_dr25, "sin_etiqueta_en_dr25": no_etiquetados,
        "cambian_de_etiqueta_respecto_a_la_seleccion": cambian,
        "n_prerregistrados_kois": int(kois["prerregistrado"].sum()),
        "prerregistrados": kois.loc[kois["prerregistrado"], "kepoi_name"].tolist(),
        "terciles_mes": [float(x) for x in np.nanquantile(usables["mes"], [1 / 3, 2 / 3])],
    }
    return ejemplos, resumen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seleccion", type=Path, required=True)
    ap.add_argument("--koi", type=Path, required=True, help="tabla q1_q17_dr25_koi")
    ap.add_argument("--acumulada", type=Path, required=True, help="tabla cumulative (para el prerregistro)")
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--manifiesto", type=Path, default=None)
    ap.add_argument("--protocolo", type=Path, default=Path("config/protocolo.json"))
    a = ap.parse_args(argv)

    protocolo = cargar_protocolo(a.protocolo)
    ejemplos, resumen = construir(pd.read_csv(a.seleccion), pd.read_csv(a.koi), pd.read_csv(a.acumulada), protocolo)
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    ejemplos.to_csv(a.salida, index=False, lineterminator="\n", float_format="%.10g")
    manifiesto = a.manifiesto or a.salida.with_name("manifiesto_base.json")
    escribir_manifiesto(manifiesto, vars(a), entradas=[a.seleccion, a.koi, a.acumulada, a.protocolo],
                        extra={"sha256_ejemplos": sha256_archivo(a.salida), "resumen": resumen})
    for clave in ("n_kois", "n_ejemplos", "por_clase", "n_prerregistrados_kois", "terciles_mes"):
        print(clave, "=", resumen[clave])
    print("cambian de etiqueta:", resumen["cambian_de_etiqueta_respecto_a_la_seleccion"])
    print("excluidos:", resumen["sin_fila_en_dr25"] + resumen["sin_etiqueta_en_dr25"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
