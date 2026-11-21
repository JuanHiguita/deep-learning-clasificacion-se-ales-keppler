"""Construye data/dataset_temporal.npz: entradas de las ramas 1, 2 y 4 (spec 002, RF-20 a RF-22).

Uso:
  python -I src/temporal/construir_dataset_temporal.py --llc data/llc --ejemplos compartido/ejemplos.csv \
      --koi data/koi_dr25.csv --salida data/dataset_temporal.npz
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from comun.protocolo import cargar_protocolo
from comun.utilidades import escribir_manifiesto, sha256_archivo
from temporal import curvas, lectura
from temporal.curvas import Efemerides

CLAVES = (
    "vista_global",
    "vista_local",
    "transitos",
    "mascara_transitos",
    "paridad_transitos",
    "n_transitos_validos",
    "consistencia",
    "escalares",
    "faltantes",
    "disponible_vistas",
    "disponible_transitos",
    "aviso_sin_senal",
)


def efemerides_de_estrella(koi_estrella: pd.DataFrame) -> dict[str, Efemerides]:
    """Efemérides de todos los KOIs de la estrella que tengan periodo, época y duración."""
    efs = {}
    for k in koi_estrella.itertuples():
        if (
            np.isfinite([k.koi_period, k.koi_time0bk, k.koi_duration]).all()
            and k.koi_period > 0
            and k.koi_duration > 0
        ):
            efs[k.kepoi_name] = Efemerides(
                float(k.koi_period), float(k.koi_time0bk), float(k.koi_duration) / 24.0
            )
    return efs


def construir(
    ejemplos: pd.DataFrame, koi: pd.DataFrame, carpeta_llc: Path, protocolo: dict
) -> tuple[dict, pd.DataFrame]:
    cfg = protocolo["extractor"]
    usar = ejemplos[~ejemplos["prerregistrado"].astype(bool)]
    koi_por_estrella = {kepid: g for kepid, g in koi.groupby("kepid")}
    koi_por_nombre = koi.set_index("kepoi_name")
    columnas = {c: [] for c in CLAVES}
    ids, registro = [], []
    estrellas = sorted(usar["kepid"].unique())
    for n, kepid in enumerate(estrellas, 1):
        efs = efemerides_de_estrella(koi_por_estrella[kepid])
        trimestres = lectura.trimestres_de_estrella(carpeta_llc, kepid)
        if len(trimestres) >= cfg["min_trimestres"]:
            t, f, s = curvas.preparar_curva(trimestres, list(efs.values()), cfg)
        else:
            t = f = s = np.empty(0)
        for fila in usar[usar["kepid"] == kepid].sort_values("id_ejemplo").itertuples():
            real = efs[fila.kepoi_name]
            tol = cfg["tolerancia_mismo_periodo"]
            # Un KOI con el mismo periodo suele ser el eclipse secundario catalogado aparte: no se oculta.
            otras = [
                e
                for nombre, e in efs.items()
                if nombre != fila.kepoi_name and abs(e.periodo / real.periodo - 1) > tol
            ]
            if fila.tipo == "sin_senal":
                ef = Efemerides(
                    real.periodo, real.t0 + cfg["desfase_sin_senal"] * real.periodo, real.duracion
                )
                ej = curvas.construir_ejemplo(t, f, s, ef, otras + [real], cfg)
            else:
                k = koi_por_nombre.loc[fila.kepoi_name]
                ej = curvas.construir_ejemplo(
                    t,
                    f,
                    s,
                    real,
                    otras,
                    cfg,
                    centroide=(float(k["koi_dikco_msky"]), float(k["koi_dicco_msky"])),
                )
            ids.append(fila.id_ejemplo)
            for c in CLAVES:
                columnas[c].append(ej[c])
            registro.append(
                {
                    "id_ejemplo": fila.id_ejemplo,
                    "kepid": int(kepid),
                    "trimestres": len(trimestres),
                    "cadencias": int(len(t)),
                    "transitos_validos": int(ej["n_transitos_validos"]),
                    **{f"descartes_{m}": v for m, v in ej["descartes"].items()},
                    "disponible_vistas": int(ej["disponible_vistas"]),
                    "disponible_transitos": int(ej["disponible_transitos"]),
                    "aviso_sin_senal": int(ej["aviso_sin_senal"]),
                }
            )
        if n % 20 == 0 or n == len(estrellas):
            print(f"[{n}/{len(estrellas)}] estrellas procesadas", flush=True)
    datos = {"id_ejemplo": np.array(ids, dtype="U")}
    for c in CLAVES:
        arr = np.stack(columnas[c]) if np.ndim(columnas[c][0]) else np.array(columnas[c])
        datos[c] = arr.astype(
            np.int8
            if c == "paridad_transitos"
            else np.float32
            if arr.dtype.kind == "f"
            else np.int32
        )
    datos["nombres_escalares"] = np.array(protocolo["escalares_permitidos"], dtype="U")
    return datos, pd.DataFrame(registro)


def _construir_un_bloque(argumentos) -> tuple[dict, pd.DataFrame]:
    """Función de nivel de módulo para que los procesos hijos puedan recibirla (Windows)."""
    ejemplos, koi, carpeta_llc, protocolo = argumentos
    return construir(ejemplos, koi, carpeta_llc, protocolo)


def construir_en_paralelo(
    ejemplos: pd.DataFrame, koi: pd.DataFrame, carpeta_llc: Path, protocolo: dict, procesos: int
) -> tuple[dict, pd.DataFrame]:
    """Igual que `construir`, repartiendo las estrellas en bloques que corren a la vez.

    Los bloques son tramos consecutivos de estrellas ordenadas por `kepid` y se unen en ese
    mismo orden: el resultado es idéntico al de un solo proceso.
    """
    if procesos <= 1:
        return construir(ejemplos, koi, carpeta_llc, protocolo)
    estrellas = np.array(sorted(ejemplos["kepid"].unique()))
    bloques = [b for b in np.array_split(estrellas, min(8 * procesos, len(estrellas))) if len(b)]
    tareas = [
        (ejemplos[ejemplos["kepid"].isin(b)], koi[koi["kepid"].isin(b)], carpeta_llc, protocolo)
        for b in bloques
    ]
    partes = []
    with ProcessPoolExecutor(max_workers=procesos) as grupo:
        for i, parte in enumerate(grupo.map(_construir_un_bloque, tareas), 1):
            partes.append(parte)
            print(f"== bloque {i} de {len(tareas)} terminado ==", flush=True)
    datos = {}
    for clave in partes[0][0]:
        if clave == "nombres_escalares":  # igual en todos los bloques
            datos[clave] = partes[0][0][clave]
        else:
            datos[clave] = np.concatenate([parte[0][clave] for parte in partes])
    return datos, pd.concat([parte[1] for parte in partes], ignore_index=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--llc", type=Path, required=True)
    ap.add_argument("--ejemplos", type=Path, required=True)
    ap.add_argument("--koi", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--protocolo", type=Path, default=Path("config/protocolo.json"))
    ap.add_argument("--procesos", type=int, default=1, help="procesos en paralelo")
    a = ap.parse_args(argv)

    protocolo = cargar_protocolo(a.protocolo)
    datos, registro = construir_en_paralelo(
        pd.read_csv(a.ejemplos), pd.read_csv(a.koi), a.llc, protocolo, a.procesos
    )
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.salida, **datos)
    ruta_registro = a.salida.with_name(a.salida.stem + "_registro.csv")
    registro.to_csv(ruta_registro, index=False, lineterminator="\n")
    n = len(registro)
    resumen = {
        "n_ejemplos": n,
        "disponible_vistas": int(registro["disponible_vistas"].sum()),
        "disponible_transitos": int(registro["disponible_transitos"].sum()),
        "sin_senal_con_snr_mayor_que_5": int(
            registro["aviso_sin_senal"][registro["id_ejemplo"].str.endswith("sin_senal")].sum()
        ),
        "estrellas_con_menos_trimestres_que_el_minimo": int(
            (
                registro.groupby("kepid")["trimestres"].first()
                < protocolo["extractor"]["min_trimestres"]
            ).sum()
        ),
    }
    escribir_manifiesto(
        a.salida.with_suffix(".manifiesto.json"),
        vars(a),
        entradas=[a.ejemplos, a.koi, a.protocolo],
        extra={"sha256": sha256_archivo(a.salida), "resumen": resumen},
    )
    print(resumen)
    fraccion = min(resumen["disponible_vistas"], resumen["disponible_transitos"]) / max(n, 1)
    if fraccion < 0.95:
        print(
            f"ATENCIÓN: solo el {fraccion:.1%} de los ejemplos está disponible (< 95 %). Detenerse y revisar."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
