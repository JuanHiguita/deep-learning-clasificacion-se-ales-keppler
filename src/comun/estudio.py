"""Estudio de decisiones de diseño: una variante cambia UNA cosa respecto al protocolo (spec 010).

Cada variante se entrena con el mismo motor, las mismas particiones y la semilla 0, y se compara
con la referencia (el experimento publicado) mediante la diferencia pareada de AUC en prueba.

Advertencia que acompaña siempre a esta tabla: son muchas comparaciones con una sola semilla.
Con ~15 variantes, es esperable que alguna "se distinga de cero" solo por azar.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from comun import estadistica, metricas
from comun.busqueda import _cargar_predicciones, _intervalo, nll_de_validacion_por_particion

REFERENCIA = "referencia"


def resumir(
    carpeta: Path, ejemplos: pd.DataFrame, protocolo: dict, descripciones: dict
) -> pd.DataFrame:
    """Tabla de variantes terminadas: validación, prueba y diferencia pareada con la referencia.

    descripciones: variante → (factor, qué cambia). Define el orden de las filas.
    """
    cfg, n_particiones = protocolo["metricas"], protocolo["particiones"]["n"]
    referencia = _cargar_predicciones(carpeta / REFERENCIA)
    if referencia is None:
        raise RuntimeError(f"Falta la referencia en {carpeta}")
    ids = [str(i) for i in referencia["id_ejemplo"]]
    tabla = ejemplos.set_index("id_ejemplo").loc[ids].reset_index()
    y, grupos = tabla["clase"].to_numpy().astype(int), tabla["kepid"].to_numpy()
    p_referencia = referencia["p_prueba"].astype(float)

    filas = []
    for variante, (factor, cambio) in descripciones.items():
        pred = _cargar_predicciones(carpeta / variante)
        if pred is None:
            continue  # estudio a medias: se resume lo terminado
        p = pred["p_prueba"].astype(float)
        resumen = json.loads((carpeta / variante / "metricas.json").read_text(encoding="utf-8"))
        fila = {
            "variante": variante,
            "factor": factor,
            "cambio": cambio,
            "parametros": resumen.get("parametros"),
            "epocas_medias": round(float(np.mean(resumen["por_semilla"]["0"]["epocas"])), 1),
            "nll_validacion": round(
                float(np.mean(nll_de_validacion_por_particion(pred, y, n_particiones))), 4
            ),
            "auc_prueba": round(metricas.auc_planeta_fp(p, y), 4),
            "nll_prueba": round(metricas.nll(p, y), 4),
            "ece_prueba": round(metricas.ece_top(p, y, cfg["ece_intervalos"]), 4),
        }
        if variante == REFERENCIA:
            fila.update(diferencia_auc=0.0, ic_inf=0.0, ic_sup=0.0, distinguible=False)
        else:

            def diferencia(i, p=p):
                return metricas.auc_planeta_fp(p[i], y[i]) - metricas.auc_planeta_fp(
                    p_referencia[i], y[i]
                )

            intervalo = _intervalo(grupos, diferencia, cfg)
            fila.update(
                diferencia_auc=intervalo["estimacion"],
                ic_inf=intervalo["ic_inf"],
                ic_sup=intervalo["ic_sup"],
                distinguible=not (intervalo["ic_inf"] <= 0 <= intervalo["ic_sup"]),
            )
        filas.append(fila)
    resultado = pd.DataFrame(filas)
    resultado.to_csv(carpeta / "variantes.csv", index=False, lineterminator="\n")
    return resultado


def confirmar(
    carpeta: Path,
    carpeta_referencia: Path,
    variantes: list[str],
    semillas: list[int],
    ejemplos: pd.DataFrame,
    protocolo: dict,
) -> pd.DataFrame:
    """Confirmación con más semillas de las variantes que parecieron mejorar con una sola.

    Compara la media de AUC de la variante con la del protocolo, semilla a semilla sobre las
    mismas estrellas. Se reporta con todas las semillas y también SIN la semilla 0, que fue la
    que motivó elegir la variante (así se evita confirmar con el mismo dato que la seleccionó).
    """
    cfg = protocolo["metricas"]

    def predicciones(de: Path, cuales: list[int]) -> tuple[np.ndarray, list[np.ndarray]]:
        ids, lista = None, []
        for semilla in cuales:
            with np.load(de / f"pred_semilla{semilla}.npz", allow_pickle=False) as archivo:
                ids = archivo["id_ejemplo"].astype(str)
                lista.append(archivo["p_prueba"].astype(float))
        return ids, lista

    filas = []
    for variante in variantes:
        for etiqueta, cuales in (
            ("todas las semillas", semillas),
            ("sin la semilla 0", [s for s in semillas if s != 0]),
        ):
            ids, de_la_variante = predicciones(carpeta / variante, cuales)
            _, del_protocolo = predicciones(carpeta_referencia, cuales)
            tabla = ejemplos.set_index("id_ejemplo").loc[ids]
            y, grupos = tabla["clase"].to_numpy().astype(int), tabla["kepid"].to_numpy()
            diferencia = estadistica.diferencia_media_semillas(
                metricas.auc_planeta_fp, y, de_la_variante, del_protocolo
            )
            intervalo = _intervalo(grupos, diferencia, cfg)
            filas.append(
                {
                    "variante": variante,
                    "semillas": etiqueta,
                    "auc_variante": round(
                        float(np.mean([metricas.auc_planeta_fp(p, y) for p in de_la_variante])), 4
                    ),
                    "auc_protocolo": round(
                        float(np.mean([metricas.auc_planeta_fp(p, y) for p in del_protocolo])), 4
                    ),
                    "diferencia_auc": intervalo["estimacion"],
                    "ic_inf": intervalo["ic_inf"],
                    "ic_sup": intervalo["ic_sup"],
                    "distinguible": not (intervalo["ic_inf"] <= 0 <= intervalo["ic_sup"]),
                }
            )
    resultado = pd.DataFrame(filas)
    resultado.to_csv(carpeta / "confirmacion.csv", index=False, lineterminator="\n")
    return resultado


def consolidar(diseno: Path, e4: Path, tablas: Path) -> pd.DataFrame:
    """Reúne el estudio en `tablas/`: diseno.csv, diseno_confirmacion.csv y e4_menos_transitos.csv."""
    tablas.mkdir(parents=True, exist_ok=True)
    partes, confirmaciones = [], []
    for carpeta in sorted(c for c in Path(diseno).iterdir() if (c / "variantes.csv").is_file()):
        partes.append(pd.read_csv(carpeta / "variantes.csv").assign(arquitectura=carpeta.name))
        if (carpeta / "confirmacion.csv").is_file():
            confirmaciones.append(
                pd.read_csv(carpeta / "confirmacion.csv").assign(arquitectura=carpeta.name)
            )
    todo = pd.concat(partes, ignore_index=True)
    todo.to_csv(tablas / "diseno.csv", index=False, lineterminator="\n")
    if confirmaciones:
        pd.concat(confirmaciones, ignore_index=True).to_csv(
            tablas / "diseno_confirmacion.csv", index=False, lineterminator="\n"
        )
    if (Path(e4) / "e4_menos_transitos.csv").is_file():
        shutil.copyfile(Path(e4) / "e4_menos_transitos.csv", tablas / "e4_menos_transitos.csv")
    return todo


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Consolida las tablas del estudio de diseño (spec 010)"
    )
    ap.add_argument("--diseno", type=Path, required=True)
    ap.add_argument("--e4", type=Path, required=True)
    ap.add_argument("--tablas", type=Path, required=True)
    a = ap.parse_args(argv)
    todo = consolidar(a.diseno, a.e4, a.tablas)
    mejoran = todo[todo.distinguible & (todo.diferencia_auc > 0)]
    empeoran = todo[todo.distinguible & (todo.diferencia_auc < 0)]
    print(
        f"{len(todo)} filas · mejoran de forma distinguible: {len(mejoran)} · empeoran: {len(empeoran)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
