"""Búsqueda aleatoria de hiperparámetros con selección anidada (spec 009).

Reglas que este módulo hace cumplir:

1. El espacio de búsqueda se declara ANTES de entrenar y el muestreo es reproducible (semilla fija).
2. La configuración 0 es siempre la del protocolo fijo: la búsqueda nunca puede quedar peor que
   ella en validación, y sirve de referencia.
3. La selección es ANIDADA: el modelo de la partición k elige su configuración mirando solo su
   validación interna (la partición k + 1). La prueba (partición k) nunca interviene.

Formato de un espacio: `{nombre: [opciones]}` o `{nombre: ("log", mínimo, máximo)}`.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from comun import estadistica, metricas
from comun.protocolo import cargar_protocolo

SEMILLA_DE_BUSQUEDA = 20261120
# Hiperparámetros de entrenamiento: van al protocolo. El resto va al constructor del modelo.
DE_ENTRENAMIENTO = ("tasa_aprendizaje", "weight_decay", "lote", "perdida")


def muestrear(espacio: dict, fija: dict, n: int, indice_arquitectura: int) -> list[dict]:
    """`n` configuraciones: la primera es `fija` (el protocolo) y el resto se sortea del espacio."""
    if set(fija) != set(espacio):
        raise ValueError("La configuración fija debe tener las mismas claves que el espacio")
    rng = np.random.default_rng([SEMILLA_DE_BUSQUEDA, indice_arquitectura])
    configuraciones = [dict(fija)]
    while len(configuraciones) < n:
        candidata = {}
        for nombre, opciones in espacio.items():
            if isinstance(opciones, tuple) and opciones[0] == "log":
                _, minimo, maximo = opciones
                valor = float(np.exp(rng.uniform(np.log(minimo), np.log(maximo))))
                candidata[nombre] = float(f"{valor:.3g}")
            else:
                candidata[nombre] = opciones[int(rng.integers(len(opciones)))]
        if candidata not in configuraciones:
            configuraciones.append(candidata)
    return configuraciones


def protocolo_con(protocolo: dict, configuracion: dict, extra: dict | None = None) -> dict:
    """Copia del protocolo con los hiperparámetros de entrenamiento de `configuracion`."""
    copia = json.loads(json.dumps(protocolo))
    for nombre in DE_ENTRENAMIENTO:
        if nombre in configuracion:
            copia["entrenamiento"][nombre] = configuracion[nombre]
    copia["entrenamiento"].update(extra or {})
    return copia


def de_arquitectura(configuracion: dict) -> dict:
    """Los hiperparámetros que recibe el constructor del modelo."""
    return {k: v for k, v in configuracion.items() if k not in DE_ENTRENAMIENTO}


def nombre_de(indice: int) -> str:
    return f"cfg{indice:02d}"


# ---------------------------------------------------------------------------------------------
# Resumen y selección anidada
# ---------------------------------------------------------------------------------------------


def _cargar_predicciones(carpeta: Path) -> dict | None:
    ruta = carpeta / "pred_semilla0.npz"
    if not ruta.is_file():
        return None
    with np.load(ruta, allow_pickle=False) as archivo:
        return {clave: archivo[clave] for clave in archivo.files}


def nll_de_validacion_por_particion(pred: dict, y: np.ndarray, n_particiones: int) -> list[float]:
    """NLL de cada modelo (uno por partición) en SU validación interna."""
    valores = []
    for k in range(n_particiones):
        suya = pred["particion_modelo_validacion"] == k
        valores.append(metricas.nll(pred["p_validacion"][suya].astype(float), y[suya]))
    return valores


def seleccion_anidada(nll_validacion: np.ndarray) -> np.ndarray:
    """Para cada partición, la configuración con menor NLL en su validación interna.

    nll_validacion: (configuraciones, particiones). Ante empate gana la de índice menor, de modo
    que la configuración fija (índice 0) se conserva si nadie la mejora.
    """
    return np.argmin(nll_validacion, axis=0)


def _intervalo(grupos, estadistico, cfg) -> dict:
    r = estadistica.bootstrap_por_estrella(
        grupos, estadistico, cfg["bootstrap_remuestreos"], cfg["bootstrap_semilla"]
    )
    return {
        "estimacion": round(r.estimacion, 4),
        "ic_inf": round(r.ic_inf, 4),
        "ic_sup": round(r.ic_sup, 4),
    }


def resumir(carpeta: Path, ejemplos: pd.DataFrame, protocolo: dict) -> dict:
    """Resume la búsqueda de una arquitectura y escribe `resumen.json` y `configuraciones.csv`.

    Compara, sobre la prueba y con la semilla 0:
      · protocolo fijo (configuración 0)
      · con búsqueda (cada partición usa la configuración que eligió su validación interna)
    """
    configuraciones = json.loads((carpeta / "configuraciones.json").read_text(encoding="utf-8"))
    n_particiones = protocolo["particiones"]["n"]
    cfg = protocolo["metricas"]

    filas, predicciones, tabla, y = [], [], None, None
    for indice, configuracion in enumerate(configuraciones):
        pred = _cargar_predicciones(carpeta / nombre_de(indice))
        if pred is None:
            continue  # búsqueda a medias: se resume lo terminado
        if tabla is None:
            ids = [str(i) for i in pred["id_ejemplo"]]
            tabla = ejemplos.set_index("id_ejemplo").loc[ids].reset_index()
            y = tabla["clase"].to_numpy().astype(int)
        por_particion = nll_de_validacion_por_particion(pred, y, n_particiones)
        p_prueba = pred["p_prueba"].astype(float)
        metricas_cfg = json.loads(
            (carpeta / nombre_de(indice) / "metricas.json").read_text(encoding="utf-8")
        )
        filas.append(
            {
                "configuracion": nombre_de(indice),
                **configuracion,
                "parametros": metricas_cfg.get("parametros"),
                "epocas_medias": round(
                    float(np.mean(metricas_cfg["por_semilla"]["0"]["epocas"])), 1
                ),
                **{f"nll_val_k{k}": round(v, 4) for k, v in enumerate(por_particion)},
                "nll_val_media": round(float(np.mean(por_particion)), 4),
                "auc_prueba": round(metricas.auc_planeta_fp(p_prueba, y), 4),
                "nll_prueba": round(metricas.nll(p_prueba, y), 4),
            }
        )
        predicciones.append(pred)
    if not filas or filas[0]["configuracion"] != nombre_de(0):
        raise RuntimeError(f"Falta la configuración fija en {carpeta}")

    tabla_cfg = pd.DataFrame(filas)
    nll_validacion = tabla_cfg[[f"nll_val_k{k}" for k in range(n_particiones)]].to_numpy()
    elegida = seleccion_anidada(nll_validacion)

    # Predicción "con búsqueda": cada partición de prueba la predice su configuración elegida
    p_fija = predicciones[0]["p_prueba"].astype(float)
    p_busqueda = p_fija.copy()
    particion = predicciones[0]["particion"]
    for k in range(n_particiones):
        p_busqueda[particion == k] = predicciones[elegida[k]]["p_prueba"][particion == k]

    grupos = tabla["kepid"].to_numpy()

    def auc(p):
        return lambda i: metricas.auc_planeta_fp(p[i], y[i])

    def nll(p):
        return lambda i: metricas.nll(p[i], y[i])

    resumen = {
        "n_configuraciones": len(filas),
        "elegida_por_particion": [filas[i]["configuracion"] for i in elegida],
        "mejor_en_validacion_media": filas[int(np.argmin(tabla_cfg.nll_val_media))][
            "configuracion"
        ],
        "nll_validacion": {
            "fijo": round(float(nll_validacion[0].mean()), 4),
            "con_busqueda": round(
                float(nll_validacion[elegida, np.arange(n_particiones)].mean()), 4
            ),
        },
        "auc_prueba": {
            "fijo": _intervalo(grupos, auc(p_fija), cfg),
            "con_busqueda": _intervalo(grupos, auc(p_busqueda), cfg),
            "diferencia": _intervalo(grupos, lambda i: auc(p_busqueda)(i) - auc(p_fija)(i), cfg),
        },
        "nll_prueba": {
            "fijo": _intervalo(grupos, nll(p_fija), cfg),
            "con_busqueda": _intervalo(grupos, nll(p_busqueda), cfg),
            "diferencia": _intervalo(grupos, lambda i: nll(p_busqueda)(i) - nll(p_fija)(i), cfg),
        },
        "nota": (
            "Una semilla. La NLL de validación 'con búsqueda' es optimista por construcción "
            "(se eligió mirándola); la comparación válida es la de prueba."
        ),
    }
    tabla_cfg.to_csv(carpeta / "configuraciones.csv", index=False, lineterminator="\n")
    (carpeta / "resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        carpeta / "pred_con_busqueda.npz",
        id_ejemplo=predicciones[0]["id_ejemplo"],
        particion=particion,
        p_prueba=p_busqueda.astype(np.float32),
    )
    return resumen


# ---------------------------------------------------------------------------------------------
# Tablas de toda la búsqueda
# ---------------------------------------------------------------------------------------------


def consolidar(base: Path, ejemplos: pd.DataFrame, protocolo: dict, tablas: Path) -> pd.DataFrame:
    """Reúne los resúmenes de todas las arquitecturas buscadas en dos tablas.

    busqueda_resumen.csv   por arquitectura: protocolo fijo, con búsqueda y su diferencia pareada
    busqueda_pares.csv     diferencias de AUC entre arquitecturas, las dos ya con búsqueda
    """
    cfg = protocolo["metricas"]
    filas, con_busqueda = [], {}
    for carpeta in sorted(c for c in Path(base).iterdir() if (c / "resumen.json").is_file()):
        resumen = json.loads((carpeta / "resumen.json").read_text(encoding="utf-8"))
        for metrica in ("auc_prueba", "nll_prueba"):
            for version, valores in resumen[metrica].items():
                filas.append(
                    {
                        "arquitectura": carpeta.name,
                        "metrica": metrica,
                        "version": version,
                        **valores,
                        "n_configuraciones": resumen["n_configuraciones"],
                    }
                )
        with np.load(carpeta / "pred_con_busqueda.npz", allow_pickle=False) as archivo:
            con_busqueda[carpeta.name] = (archivo["id_ejemplo"].astype(str), archivo["p_prueba"])
    resumen_global = pd.DataFrame(filas)
    tablas.mkdir(parents=True, exist_ok=True)
    resumen_global.to_csv(tablas / "busqueda_resumen.csv", index=False, lineterminator="\n")

    pares = []
    indexados = ejemplos.set_index("id_ejemplo")
    for a, b in combinations(con_busqueda, 2):
        ids_a, p_a = con_busqueda[a]
        ids_b, p_b = con_busqueda[b]
        if not np.array_equal(ids_a, ids_b):
            continue  # solo se comparan modelos evaluados sobre los mismos ejemplos
        y = indexados.loc[ids_a, "clase"].to_numpy().astype(int)
        grupos = indexados.loc[ids_a, "kepid"].to_numpy()
        p_a, p_b = p_a.astype(float), p_b.astype(float)

        def diferencia(i, p_a=p_a, p_b=p_b, y=y):
            return metricas.auc_planeta_fp(p_a[i], y[i]) - metricas.auc_planeta_fp(p_b[i], y[i])

        intervalo = _intervalo(grupos, diferencia, cfg)
        distinguible = not (intervalo["ic_inf"] <= 0 <= intervalo["ic_sup"])
        pares.append({"par": f"{a} − {b}", **intervalo, "distinguible": distinguible})
    pd.DataFrame(pares).to_csv(tablas / "busqueda_pares.csv", index=False, lineterminator="\n")
    return resumen_global


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Consolida las tablas de la búsqueda (spec 009)")
    ap.add_argument("--busqueda", type=Path, required=True)
    ap.add_argument("--ejemplos", type=Path, required=True)
    ap.add_argument("--tablas", type=Path, required=True)
    ap.add_argument("--protocolo", type=Path, default=Path("config/protocolo.json"))
    a = ap.parse_args(argv)
    protocolo = cargar_protocolo(a.protocolo)
    ejemplos = pd.read_csv(a.ejemplos)
    for carpeta in sorted(
        c for c in a.busqueda.iterdir() if (c / "configuraciones.json").is_file()
    ):
        resumir(carpeta, ejemplos, protocolo)
    tabla = consolidar(a.busqueda, ejemplos, protocolo, a.tablas)
    print(tabla[tabla.metrica == "auc_prueba"].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
