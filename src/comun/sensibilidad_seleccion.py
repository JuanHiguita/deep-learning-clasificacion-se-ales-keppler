"""Análisis de sensibilidad EXPLORATORIO: ¿cambia la búsqueda si se elige por AUC en vez de NLL?

El criterio declarado antes de entrenar es la NLL de la validación interna, y sigue siendo el
principal. Este análisis se añadió después de ver los resultados de la rama 4 (la NLL penaliza al
suavizado de etiquetas, que baja la confianza a propósito), así que NO reemplaza al principal:
solo dice si la conclusión depende del criterio. No entrena nada: reutiliza las predicciones.

En los dos criterios, la partición k elige mirando solo SU validación interna; la prueba no decide.

Uso:
  python -I src/comun/sensibilidad_seleccion.py --busqueda iteracion_2/evidencias/resultados/busqueda \
      --ejemplos iteracion_2/datos/ejemplos.csv --protocolo iteracion_2/config/protocolo.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from comun import busqueda, estadistica, metricas
from comun.protocolo import cargar_protocolo


def auc_de_validacion_por_particion(pred: dict, y: np.ndarray, n_particiones: int) -> list[float]:
    """AUC planeta vs. falso positivo de cada modelo (uno por partición) en SU validación interna."""
    valores = []
    for k in range(n_particiones):
        suya = pred["particion_modelo_validacion"] == k
        valores.append(metricas.auc_planeta_fp(pred["p_validacion"][suya].astype(float), y[suya]))
    return valores


def analizar(carpeta: Path, ejemplos: pd.DataFrame, protocolo: dict) -> dict:
    """Compara, sobre la prueba, la selección por NLL (principal) con la selección por AUC."""
    carpeta = Path(carpeta)
    configuraciones = json.loads((carpeta / "configuraciones.json").read_text(encoding="utf-8"))
    n_particiones = protocolo["particiones"]["n"]
    cfg = protocolo["metricas"]

    nombres, predicciones, por_nll, por_auc, tabla, y = [], [], [], [], None, None
    for indice in range(len(configuraciones)):
        pred = busqueda._cargar_predicciones(carpeta / busqueda.nombre_de(indice))
        if pred is None:
            continue
        if tabla is None:
            ids = [str(i) for i in pred["id_ejemplo"]]
            tabla = ejemplos.set_index("id_ejemplo").loc[ids].reset_index()
            y = tabla["clase"].to_numpy().astype(int)
        nombres.append(busqueda.nombre_de(indice))
        predicciones.append(pred)
        por_nll.append(busqueda.nll_de_validacion_por_particion(pred, y, n_particiones))
        por_auc.append(auc_de_validacion_por_particion(pred, y, n_particiones))
    if not nombres or nombres[0] != busqueda.nombre_de(0):
        raise RuntimeError(f"Falta la configuración fija en {carpeta}")

    elegida = {
        "nll": busqueda.seleccion_anidada(np.array(por_nll)),
        "auc": np.argmax(np.array(por_auc), axis=0),  # ante empate, la de índice menor
    }
    particion = predicciones[0]["particion"]
    p = {"fijo": predicciones[0]["p_prueba"].astype(float)}
    for criterio, indices in elegida.items():
        p[criterio] = p["fijo"].copy()
        for k in range(n_particiones):
            p[criterio][particion == k] = predicciones[indices[k]]["p_prueba"][particion == k]

    grupos = tabla["kepid"].to_numpy()

    def intervalo(estadistico) -> dict:
        r = estadistica.bootstrap_por_estrella(
            grupos, estadistico, cfg["bootstrap_remuestreos"], cfg["bootstrap_semilla"]
        )
        return {
            "estimacion": round(r.estimacion, 4),
            "ic_inf": round(r.ic_inf, 4),
            "ic_sup": round(r.ic_sup, 4),
        }

    def auc(a):
        return lambda i: metricas.auc_planeta_fp(p[a][i], y[i])

    def nll(a):
        return lambda i: metricas.nll(p[a][i], y[i])

    def exactitud(a):
        return lambda i: metricas.exactitud(p[a][i], y[i])

    resultado = {
        "caracter": "exploratorio: añadido después de ver resultados; el criterio principal es NLL",
        "n_configuraciones": len(nombres),
        "elegida_por_particion": {c: [nombres[i] for i in v] for c, v in elegida.items()},
        "particiones_con_la_misma_eleccion": int((elegida["nll"] == elegida["auc"]).sum()),
        "auc_prueba": {
            "fijo": intervalo(auc("fijo")),
            "seleccion_por_nll": intervalo(auc("nll")),
            "seleccion_por_auc": intervalo(auc("auc")),
            "auc_menos_nll": intervalo(lambda i: auc("auc")(i) - auc("nll")(i)),
        },
        # Fracción de ejemplos cuya clase más probable (de las tres) es la correcta.
        "exactitud_prueba": {
            "fijo": intervalo(exactitud("fijo")),
            "seleccion_por_nll": intervalo(exactitud("nll")),
            "seleccion_por_auc": intervalo(exactitud("auc")),
            "nll_menos_fijo": intervalo(lambda i: exactitud("nll")(i) - exactitud("fijo")(i)),
            "auc_menos_nll": intervalo(lambda i: exactitud("auc")(i) - exactitud("nll")(i)),
        },
        "nll_prueba": {
            "fijo": intervalo(nll("fijo")),
            "seleccion_por_nll": intervalo(nll("nll")),
            "seleccion_por_auc": intervalo(nll("auc")),
            "auc_menos_nll": intervalo(lambda i: nll("auc")(i) - nll("nll")(i)),
        },
    }
    (carpeta / "sensibilidad_seleccion.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return resultado


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--busqueda", type=Path, required=True, help="carpeta con una subcarpeta por red"
    )
    ap.add_argument("--ejemplos", type=Path, required=True)
    ap.add_argument("--protocolo", type=Path, required=True)
    ap.add_argument("--arquitecturas", nargs="+", default=None)
    a = ap.parse_args(argv)
    protocolo = cargar_protocolo(a.protocolo)
    ejemplos = pd.read_csv(a.ejemplos)
    carpetas = sorted(c for c in a.busqueda.iterdir() if (c / "resumen.json").is_file())
    for carpeta in carpetas:
        if a.arquitecturas and carpeta.name not in a.arquitecturas:
            continue
        r = analizar(carpeta, ejemplos, protocolo)
        print(
            carpeta.name, json.dumps(r["auc_prueba"]["auc_menos_nll"]), r["elegida_por_particion"]
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
