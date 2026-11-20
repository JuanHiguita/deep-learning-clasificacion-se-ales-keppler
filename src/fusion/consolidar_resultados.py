"""Consolida todos los resultados en tablas: único origen de las cifras del informe (spec 007).

Lee lo que dejaron los experimentos en `resultados/dl/` y escribe en `resultados/dl/tablas/`:

    lineas_base.csv            modelos de referencia (tabulares y demo vuelta a medir)
    p2_fusion.csv              fusión, ablación y sus diferencias pareadas (pregunta 2)
    p3_calibracion.csv         temperature scaling y prior explícito (pregunta 3)
    p3_incertidumbre.csv       determinista, MC Dropout y ensamble (pregunta 3)
    resumen_hipotesis.csv      veredicto mecánico de cada hipótesis
    p1_permutacion_resumen.csv prueba de permutación del orden de los tránsitos (pregunta 1)

El veredicto es mecánico: una hipótesis "se sostiene" solo si el intervalo del 95 % de la
diferencia pareada queda entero del lado esperado.

Uso:
  python -I src/fusion/consolidar_resultados.py --resultados resultados/dl \
      --ejemplos compartido/ejemplos.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from comun import estadistica, metricas
from comun.protocolo import cargar_protocolo

REFERENCIA = "e6a_cuatro_ramas"
NOMBRES = {
    REFERENCIA: "Fusión de 4 ramas (E6a)",
    "e6a_ajuste_cruzado": "Fusión con ajuste cruzado",
    "e6c_fusion_tardia": "Fusión tardía: promedio de probabilidades (E6c)",
    "e7_sin_vistas": "Sin rama 1 (vistas)",
    "e7_sin_transitos": "Sin rama 2 (tránsitos)",
    "e7_sin_imagen": "Sin rama 3 (imagen v0)",
    "e7_sin_escalares": "Sin rama 4 (escalares)",
    "e7_solo_vistas": "Solo rama 1 (vistas)",
    "e7_solo_transitos": "Solo rama 2 (tránsitos)",
    "e7_solo_imagen": "Solo rama 3 (imagen v0, demo)",
    "e7_solo_escalares": "Solo rama 4 (escalares)",
}
MODELOS_DE_LA_DEMO = {
    "m0_demo_curva": "Demo vuelta a medir: solo curva (M0)",
    "m0_demo_completo": "Demo vuelta a medir: imagen + curva (M0)",
    "imagen_v0": "Imagen v0 de la demo, sola",
}
NO_SE_SOSTIENE = "no se sostiene (el intervalo incluye el 0 o va en contra)"


# ---------------------------------------------------------------------------------------------
# Lectura y utilidades
# ---------------------------------------------------------------------------------------------


def _leer_json(ruta: Path):
    return json.loads(ruta.read_text(encoding="utf-8"))


def _guardar(tabla: pd.DataFrame, ruta: Path) -> None:
    tabla.to_csv(ruta, index=False, lineterminator="\n")


def _predicciones(carpeta: Path, semillas) -> tuple[np.ndarray, list[np.ndarray]]:
    """(ids, [p_prueba de cada semilla disponible]) de un experimento."""
    ids, por_semilla = None, []
    for semilla in semillas:
        ruta = carpeta / f"pred_semilla{semilla}.npz"
        if ruta.is_file():
            with np.load(ruta, allow_pickle=False) as d:
                ids = d["id_ejemplo"].astype(str)
                por_semilla.append(d["p_prueba"].astype(float))
    return ids, por_semilla


class _Bootstrap:
    """Intervalos por bootstrap de estrellas con la configuración del protocolo."""

    def __init__(self, cfg: dict):
        self.remuestreos = cfg["bootstrap_remuestreos"]
        self.semilla = cfg["bootstrap_semilla"]

    def __call__(self, grupos, estadistico) -> tuple[float, float, float]:
        r = estadistica.bootstrap_por_estrella(grupos, estadistico, self.remuestreos, self.semilla)
        return round(r.estimacion, 4), round(r.ic_inf, 4), round(r.ic_sup, 4)


def _media_de_semillas(metrica, y, por_semilla):
    """Estadístico: la métrica promediada sobre las semillas, en los índices remuestreados."""
    return lambda i: float(np.mean([metrica(p[i], y[i]) for p in por_semilla]))


# ---------------------------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------------------------


def tabla_lineas_base(resultados: Path, ejemplos: pd.DataFrame, protocolo, ic) -> pd.DataFrame:
    """Modelos tabulares de referencia y los modelos de la demo vueltos a medir."""
    tabulares = _leer_json(resultados / "lineas_base_tabulares" / "metricas.json")["modelos"]
    filas = [
        dict(modelo=nombre, auc=v["auc"], ic_inf=v["ic_inf"], ic_sup=v["ic_sup"], n=v["n"])
        for nombre, v in tabulares.items()
    ]
    for carpeta, etiqueta in MODELOS_DE_LA_DEMO.items():
        ids, por_semilla = _predicciones(resultados / carpeta, protocolo["semillas"])
        y = ejemplos.loc[ids, "clase"].to_numpy()
        grupos = ejemplos.loc[ids, "kepid"].to_numpy()
        auc, inf, sup = ic(grupos, _media_de_semillas(metricas.auc_planeta_fp, y, por_semilla))
        filas.append(dict(modelo=etiqueta, auc=auc, ic_inf=inf, ic_sup=sup, n=int((y < 2).sum())))
    return pd.DataFrame(filas)


def tabla_fusion(resultados: Path, ejemplos: pd.DataFrame, protocolo, ic) -> pd.DataFrame:
    """Pregunta 2: cada configuración de fusión y su diferencia pareada con la fusión completa."""
    cfg = protocolo["metricas"]
    semillas = protocolo["semillas_cabeza_fusion"]
    funciones = {
        "auc_planeta_fp": metricas.auc_planeta_fp,
        "nll": metricas.nll,
        "exactitud": metricas.exactitud,
        "ece_top": lambda p, yy: metricas.ece_top(p, yy, cfg["ece_intervalos"]),
    }
    predicciones = {}
    for clave in NOMBRES:
        ids, predicciones[clave] = _predicciones(resultados / "fusion" / clave, semillas)
    y = ejemplos.loc[ids, "clase"].to_numpy()
    grupos = ejemplos.loc[ids, "kepid"].to_numpy()
    filas = []

    # a) cada modelo, con su intervalo
    for clave, por_semilla in predicciones.items():
        for metrica, funcion in funciones.items():
            est, inf, sup = ic(grupos, _media_de_semillas(funcion, y, por_semilla))
            filas.append(
                dict(
                    tipo="modelo",
                    modelo=NOMBRES[clave],
                    clave=clave,
                    metrica=metrica,
                    estimacion=est,
                    ic_inf=inf,
                    ic_sup=sup,
                    semillas=len(por_semilla),
                )
            )

    def fila_diferencia(etiqueta, clave, metrica, est, inf, sup) -> dict:
        return dict(
            tipo="diferencia",
            modelo=etiqueta,
            clave=clave,
            metrica=metrica,
            estimacion=est,
            ic_inf=inf,
            ic_sup=sup,
            distinguible=not (inf <= 0 <= sup),
        )

    # b) fusión completa − cada alternativa (mismos ejemplos, mismos remuestreos)
    for clave in NOMBRES:
        if clave == REFERENCIA:
            continue
        for metrica in ("auc_planeta_fp", "nll"):
            diferencia = estadistica.diferencia_media_semillas(
                funciones[metrica], y, predicciones[REFERENCIA], predicciones[clave]
            )
            etiqueta = f"Fusión (E6a) − {NOMBRES[clave]}"
            filas.append(fila_diferencia(etiqueta, clave, metrica, *ic(grupos, diferencia)))

    # c) fusión frente a la demo completa, solo en los ejemplos que tienen imagen
    ids_demo, p_demo = _predicciones(resultados / "m0_demo_completo", protocolo["semillas"])
    posicion = {ident: j for j, ident in enumerate(ids)}
    con_imagen = np.array([posicion[ident] for ident in ids_demo])
    diferencia = estadistica.diferencia_media_semillas(
        metricas.auc_planeta_fp,
        y[con_imagen],
        [p[con_imagen] for p in predicciones[REFERENCIA]],
        p_demo,
    )
    filas.append(
        fila_diferencia(
            "Fusión (E6a) − demo completa (M0), ejemplos con imagen",
            "m0_demo_completo",
            "auc_planeta_fp",
            *ic(grupos[con_imagen], diferencia),
        )
    )
    return pd.DataFrame(filas)


def tabla_calibracion(calibracion: dict) -> pd.DataFrame:
    """Pregunta 3: antes y después de temperature scaling (E8) y del prior explícito (E9)."""
    e8 = calibracion["E8_temperatura"]
    filas = []
    for metrica in e8["antes"]:
        diferencia = e8["diferencia_despues_menos_antes"][metrica]
        filas.append(
            dict(
                experimento="E8",
                cantidad=metrica,
                antes=e8["antes"][metrica]["estimacion"],
                despues=e8["despues"][metrica]["estimacion"],
                diferencia=diferencia["estimacion"],
                ic_inf=diferencia["ic_inf"],
                ic_sup=diferencia["ic_sup"],
            )
        )
    for v in calibracion["E9_prior"]["por_tasa"].values():
        diferencia = v["diferencia_corregido_menos_sin"]
        filas.append(
            dict(
                experimento="E9",
                cantidad=f"ECE ponderado, tasa base {v['tasa']}",
                antes=v["ece_ponderado_sin_corregir"],
                despues=v["ece_ponderado_corregido"],
                diferencia=diferencia["estimacion"],
                ic_inf=diferencia["ic_inf"],
                ic_sup=diferencia["ic_sup"],
            )
        )
    return pd.DataFrame(filas)


def tabla_incertidumbre(incertidumbre: dict) -> pd.DataFrame:
    """Pregunta 3: una fila por método (determinista, MC Dropout, ensamble)."""
    filas = []
    for metodo, v in incertidumbre.items():
        if not (isinstance(v, dict) and "nll" in v):
            continue
        sin_ic = {clave: valor for clave, valor in v.items() if clave != "nll_ic"}
        filas.append(
            {"metodo": metodo, **sin_ic, "nll_ic_inf": v["nll_ic"][0], "nll_ic_sup": v["nll_ic"][1]}
        )
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------------------------
# Veredicto de las hipótesis
# ---------------------------------------------------------------------------------------------


def _veredicto(hipotesis, pregunta, cantidad, est, inf, sup, se_espera_positivo=True) -> dict:
    """Se sostiene solo si TODO el intervalo queda del lado esperado."""
    se_sostiene = (inf > 0) if se_espera_positivo else (sup < 0)
    return dict(
        hipotesis=hipotesis,
        pregunta=pregunta,
        cantidad=cantidad,
        estimacion=est,
        ic_inf=inf,
        ic_sup=sup,
        veredicto="se sostiene" if se_sostiene else NO_SE_SOSTIENE,
    )


def _diferencia(tabla: pd.DataFrame, columna: str, valor: str, metrica="auc_planeta_fp"):
    """(estimación, ic_inf, ic_sup) de una fila de diferencias pareadas."""
    es_la_fila = (tabla.tipo == "diferencia") & (tabla[columna] == valor)
    fila = tabla[es_la_fila & (tabla.metrica == metrica)].iloc[0]
    return fila.estimacion, fila.ic_inf, fila.ic_sup


def _terna(intervalo: dict) -> tuple[float, float, float]:
    return intervalo["estimacion"], intervalo["ic_inf"], intervalo["ic_sup"]


def tabla_hipotesis(p1: pd.DataFrame, p2: pd.DataFrame, calibracion: dict, incertidumbre: dict):
    """Una fila por hipótesis, con la diferencia que la decide y su veredicto."""

    def de_p1(par: str):
        return _diferencia(p1, "modelo", par)

    def de_p2(clave: str):
        return _diferencia(p2, "clave", clave)

    filas = [
        _veredicto(
            "P1a",
            "¿La lectura tránsito por tránsito supera a la curva plegada?",
            "AUC: rama 2 con atención − rama 1 (AstroNet)",
            *de_p1("m2_atencion − m1_astronet"),
        ),
        _veredicto(
            "P1b",
            "¿La atención aporta frente al promedio?",
            "AUC: atención − promedio",
            *de_p1("m2_atencion − m2_promedio"),
        ),
        _veredicto(
            "P1c",
            "¿La GRU supera a la atención?",
            "AUC: GRU − atención",
            *de_p1("m2_gru − m2_atencion"),
        ),
        _veredicto(
            "P2",
            "¿La fusión supera a la mejor rama sola?",
            "AUC: fusión − solo rama 4 (escalares)",
            *de_p2("e7_solo_escalares"),
        ),
    ]
    for rama in ("vistas", "transitos", "imagen", "escalares"):
        filas.append(
            _veredicto(
                "P2",
                f"¿Aporta la rama? ({NOMBRES['e7_sin_' + rama]})",
                f"AUC: fusión − sin {rama}",
                *de_p2(f"e7_sin_{rama}"),
            )
        )

    e8 = calibracion["E8_temperatura"]["diferencia_despues_menos_antes"]["nll"]
    e9 = calibracion["E9_prior"]["por_tasa"]["rejilla_0.05"]["diferencia_corregido_menos_sin"]
    e10 = incertidumbre["diferencia_nll_ensamble_menos_mc"]
    filas += [
        _veredicto(
            "P2",
            "¿La fusión supera a la demo vuelta a medir?",
            "AUC: fusión − demo completa",
            *de_p2("m0_demo_completo"),
        ),
        # En P3 se espera que la cantidad BAJE: el intervalo debe quedar bajo 0.
        _veredicto(
            "P3a",
            "¿Mejora el NLL con temperature scaling?",
            "NLL: después − antes",
            *_terna(e8),
            se_espera_positivo=False,
        ),
        _veredicto(
            "P3b",
            "¿El prior explícito mejora la calibración (tasa base 0,05)?",
            "ECE ponderado: corregido − sin corregir",
            *_terna(e9),
            se_espera_positivo=False,
        ),
        _veredicto(
            "P3c",
            "¿El ensamble supera a MC Dropout?",
            "NLL: ensamble − MC Dropout",
            *_terna(e10),
            se_espera_positivo=False,
        ),
    ]
    return pd.DataFrame(filas)


def _resumen_de_permutacion(permutacion: dict) -> str:
    lineas = ["modelo,cambio_medio_abs_de_q,cambio_de_auc"]
    for modelo, v in permutacion.items():
        lineas.append(f"{modelo},{v['cambio_medio_abs_de_q']},{v['cambio_de_auc']}")
    return "\n".join(lineas) + "\n"


# ---------------------------------------------------------------------------------------------
# Programa
# ---------------------------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--resultados", type=Path, required=True)
    ap.add_argument("--ejemplos", type=Path, required=True)
    ap.add_argument("--protocolo", type=Path, default=Path("config/protocolo.json"))
    a = ap.parse_args(argv)

    protocolo = cargar_protocolo(a.protocolo)
    ic = _Bootstrap(protocolo["metricas"])
    resultados, tablas = a.resultados, a.resultados / "tablas"
    tablas.mkdir(parents=True, exist_ok=True)
    ejemplos = pd.read_csv(a.ejemplos).set_index("id_ejemplo")
    calibracion = _leer_json(resultados / "calibracion" / "metricas.json")
    incertidumbre = _leer_json(resultados / "incertidumbre" / "metricas.json")

    _guardar(tabla_lineas_base(resultados, ejemplos, protocolo, ic), tablas / "lineas_base.csv")
    p2 = tabla_fusion(resultados, ejemplos, protocolo, ic)
    _guardar(p2, tablas / "p2_fusion.csv")
    _guardar(tabla_calibracion(calibracion), tablas / "p3_calibracion.csv")
    _guardar(tabla_incertidumbre(incertidumbre), tablas / "p3_incertidumbre.csv")

    # p1_arquitecturas.csv y p1_permutacion.json los escribe temporal/analisis_ramas.py
    p1 = pd.read_csv(tablas / "p1_arquitecturas.csv")
    hipotesis = tabla_hipotesis(p1, p2, calibracion, incertidumbre)
    _guardar(hipotesis, tablas / "resumen_hipotesis.csv")
    permutacion = _leer_json(tablas / "p1_permutacion.json")
    (tablas / "p1_permutacion_resumen.csv").write_text(
        _resumen_de_permutacion(permutacion), encoding="utf-8"
    )

    columnas = ["tipo", "modelo", "estimacion", "ic_inf", "ic_sup"]
    print(p2[p2.metrica == "auc_planeta_fp"][columnas].to_string(index=False))
    columnas = ["hipotesis", "cantidad", "estimacion", "ic_inf", "ic_sup", "veredicto"]
    print(hipotesis[columnas].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
