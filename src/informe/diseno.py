"""Figuras del notebook `03_estudio_de_diseno.ipynb` (specs 010 y 011)."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from informe.datos import Resultados
from informe.graficos import COLOR, bosque, color_si_distingue

NOMBRE = {
    "m1_astronet": "Rama 1 · CNN tipo AstroNet",
    "m2_atencion": "Rama 2 · atención",
    "m4_escalares": "Rama 4 · escalares",
    "m2_promedio": "Rama 2 · promedio",
    "m2_gru": "Rama 2 · GRU",
}
CON_ESTUDIO_COMPLETO = ("m1_astronet", "m2_atencion", "m4_escalares")
AGREGADOR = {"promedio": "promedio", "gru": "GRU", "atencion": "atención"}
COLOR_DE_AGREGADOR = {"promedio": COLOR["base"], "gru": COLOR["demo"], "atencion": COLOR["curso"]}


def variantes(res: Resultados, arquitectura: str) -> pd.DataFrame:
    """Tabla de variantes de una arquitectura (la escribe `temporal/estudio_diseno.py`)."""
    return pd.read_csv(res.carpeta / "diseno" / arquitectura / "variantes.csv")


def disponibles(res: Resultados) -> list[str]:
    return [
        a for a in CON_ESTUDIO_COMPLETO if (res.carpeta / "diseno" / a / "variantes.csv").is_file()
    ]


def efecto_de_cada_decision(res: Resultados, arquitectura: str) -> pd.DataFrame:
    """Cada variante frente al protocolo: diferencia pareada de AUC en prueba con IC del 95 %.

    A la izquierda de 0 la variante empeora; a la derecha mejora. En verde, las que se
    distinguen de cero. Las variantes cuyo entrenamiento no convergió se listan aparte.
    """
    tabla = variantes(res, arquitectura)
    tabla = tabla[tabla.variante != "referencia"].reset_index(drop=True)
    etiquetas = [f"{fila.factor}: {fila.cambio}" for fila in tabla.itertuples()]

    fig, ax = plt.subplots(figsize=(11, 0.36 * len(tabla) + 1.6))
    bosque(
        ax,
        etiquetas,
        tabla.diferencia_auc,
        tabla.ic_inf,
        tabla.ic_sup,
        color_si_distingue(tabla.distinguible),
        referencia=0,
    )
    ax.set_xlabel("AUC de la variante − AUC del protocolo (pareada, semilla 0), IC 95 %")
    ax.set_title(f"{NOMBRE[arquitectura]}: efecto de cambiar UNA decisión de diseño")
    plt.tight_layout()
    plt.show()

    referencia = variantes(res, arquitectura).iloc[0]
    print(
        f"Protocolo: AUC {referencia.auc_prueba:.3f} · NLL de validación "
        f"{referencia.nll_validacion:.3f} · {int(referencia.parametros)} parámetros"
    )
    columnas = [
        "factor",
        "cambio",
        "parametros",
        "epocas_medias",
        "nll_validacion",
        "auc_prueba",
        "diferencia_auc",
        "ic_inf",
        "ic_sup",
        "distinguible",
    ]
    return tabla[columnas]


def resumen_por_factor(res: Resultados) -> pd.DataFrame:
    """Para cada modelo y factor: la mejor variante en VALIDACIÓN y qué pasó con ella en prueba."""
    filas = []
    for arquitectura in disponibles(res):
        tabla = variantes(res, arquitectura)
        referencia = tabla.iloc[0]
        for factor, grupo in tabla.iloc[1:].groupby("factor", sort=False):
            mejor = grupo.loc[grupo.nll_validacion.idxmin()]
            filas.append(
                {
                    "modelo": NOMBRE[arquitectura],
                    "factor": factor,
                    "mejor variante en validación": mejor.cambio,
                    "¿mejora la validación?": bool(
                        mejor.nll_validacion < referencia.nll_validacion
                    ),
                    "diferencia de AUC en prueba": mejor.diferencia_auc,
                    "IC inf": mejor.ic_inf,
                    "IC sup": mejor.ic_sup,
                    "se distingue de 0": bool(mejor.distinguible),
                }
            )
    return pd.DataFrame(filas)


def sin_consistencia(res: Resultados) -> pd.DataFrame:
    """E2b: qué pierde cada agregador al quitarle los 4 estadísticos de consistencia."""
    filas = []
    for agregador in AGREGADOR:
        tabla = variantes(res, f"m2_{agregador}").set_index("variante")
        fila = tabla.loc["sin_consistencia"]
        filas.append(
            {
                "agregador": AGREGADOR[agregador],
                "AUC con consistencia": tabla.loc["referencia", "auc_prueba"],
                "AUC sin consistencia": fila.auc_prueba,
                "diferencia": fila.diferencia_auc,
                "IC inf": fila.ic_inf,
                "IC sup": fila.ic_sup,
                "se distingue de 0": bool(fila.distinguible),
            }
        )
    resultado = pd.DataFrame(filas)
    fig, ax = plt.subplots(figsize=(8, 2.4))
    bosque(
        ax,
        list(resultado.agregador),
        resultado.diferencia,
        resultado["IC inf"],
        resultado["IC sup"],
        color_si_distingue(resultado["se distingue de 0"]),
        referencia=0,
    )
    ax.set_xlabel("AUC sin consistencia − AUC con consistencia (pareada), IC 95 %")
    ax.set_title("E2b: lo que pierde cada agregador sin los estadísticos entre tránsitos")
    plt.tight_layout()
    plt.show()
    return resultado


def menos_transitos(res: Resultados) -> pd.DataFrame:
    """E4: AUC de cada agregador según cuántos tránsitos se le dejan ver en la prueba."""
    tabla = pd.read_csv(res.carpeta / "e4" / "e4_menos_transitos.csv")
    modelos, diferencias = tabla[tabla.tipo == "modelo"], tabla[tabla.tipo == "diferencia"]
    fig, (ax_auc, ax_dif) = plt.subplots(1, 2, figsize=(13, 3.6))

    for agregador, nombre in AGREGADOR.items():
        fila = modelos[modelos.modelo == agregador].sort_values("transitos")
        color = COLOR_DE_AGREGADOR[agregador]
        ax_auc.plot(fila.transitos, fila.estimacion, "o-", color=color, label=nombre)
        ax_auc.fill_between(fila.transitos, fila.ic_inf, fila.ic_sup, color=color, alpha=0.12)
    ax_auc.set_xscale("log", base=2)
    ax_auc.set_xticks(sorted(modelos.transitos.unique()), sorted(modelos.transitos.unique()))
    ax_auc.set_xlabel("tránsitos que ve el modelo en la prueba")
    ax_auc.set_ylabel("AUC planeta vs FP")
    ax_auc.set_title("Menos tránsitos, menos AUC (franja = IC 95 %)")
    ax_auc.legend()

    contra_promedio = diferencias[diferencias.modelo == "atencion − promedio"].sort_values(
        "transitos", ascending=False
    )
    bosque(
        ax_dif,
        [f"{n} tránsitos" for n in contra_promedio.transitos],
        contra_promedio.estimacion,
        contra_promedio.ic_inf,
        contra_promedio.ic_sup,
        color_si_distingue(contra_promedio.distinguible.astype(bool)),
        referencia=0,
    )
    ax_dif.set_xlabel("AUC de atención − AUC de promedio (pareada), IC 95 %")
    ax_dif.set_title("¿Cuándo le gana la atención al promedio?")
    plt.tight_layout()
    plt.show()
    return diferencias.drop(columns="tipo").reset_index(drop=True)


def prueba_de_extremo_a_extremo(res: Resultados) -> pd.DataFrame:
    """Resultado guardado de `scripts/prueba_e2e.py`: una fila por estrella probada."""
    informe = json.loads((res.carpeta / "e2e" / "informe.json").read_text(encoding="utf-8"))
    filas = []
    for estrella in informe["estrellas"]:
        filas.append(
            {
                "kepid": estrella["kepid"],
                "partición": estrella["particion"],
                "entradas": estrella["diferencia_entradas"],
                "vectores": max(estrella["diferencia_vectores"].values()),
                "probabilidad": estrella["diferencia_probabilidad"],
                "coincide": estrella["correcto"],
            }
        )
    veredicto = "correcta" if informe["correcto"] else "CON DIFERENCIAS"
    print(
        f"Prueba de extremo a extremo: {veredicto} · {len(filas)} estrellas al azar "
        f"(semilla {informe['semilla']}) · tolerancia {informe['tolerancia']}"
    )
    return pd.DataFrame(filas)


def respuesta_del_sistema(
    res: Resultados, id_ejemplo: str, tasa_base: float = 0.05
) -> pd.DataFrame:
    """Ejecuta el sistema completo sobre una señal y muestra su respuesta conjunta."""
    from fusion.sistema import RAMAS, Sistema  # importa Keras: solo cuando se usa

    i = res.posicion(id_ejemplo)
    particion = int(res.tabla.particion.iloc[i])
    entradas = {c: v[i : i + 1] for c, v in res.dataset.items() if len(v) == len(res.ids)}
    demo = dict(np.load(res.raiz / "data" / "dataset_demo.npz", allow_pickle=False))
    en_demo = np.flatnonzero(demo["id_ejemplo"].astype(str) == id_ejemplo)
    if len(en_demo):
        entradas["img"] = demo["img"][en_demo]
        entradas["disponible_imagen"] = np.ones(1, np.float32)

    sistema = Sistema(res.raiz / "modelos")
    respuesta = sistema.responder(entradas, particion=particion, tasa_base=tasa_base)
    clases = ("planeta", "falso positivo", "sin señal")
    real = clases[int(res.y[i])]
    con_dato = [r for r, tiene in zip(RAMAS, respuesta.ramas_con_dato[0], strict=True) if tiene]
    print(f"{id_ejemplo} · clase real: {real} · se usó el sistema de la partición {particion}")
    print("Ramas con dato:", ", ".join(con_dato))
    return pd.DataFrame(
        {
            "cantidad": [
                *[f"P({c})" for c in clases],
                "P(planeta | hay señal)",
                f"P(planeta | hay señal) con tasa base {tasa_base:g}",
                "entropía (duda)",
            ],
            "valor": np.round(
                [
                    *respuesta.p[0],
                    respuesta.q_planeta[0],
                    respuesta.q_con_tasa_base[0],
                    respuesta.entropia[0],
                ],
                4,
            ),
        }
    )


def confirmacion(res: Resultados) -> pd.DataFrame:
    """Variantes de la rama 1 repetidas con 3 semillas: ¿se sostiene la mejora?"""
    tabla = res.csv("diseno_confirmacion")
    descripcion = variantes(res, "m1_astronet").set_index("variante").cambio
    etiquetas = [f"{descripcion[f.variante]} · {f.semillas}" for f in tabla.itertuples()]
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(tabla) + 1.5))
    bosque(
        ax,
        etiquetas,
        tabla.diferencia_auc,
        tabla.ic_inf,
        tabla.ic_sup,
        color_si_distingue(tabla.distinguible),
        referencia=0,
    )
    ax.set_xlabel("AUC de la variante − AUC del protocolo (media de semillas, pareada), IC 95 %")
    ax.set_title("Rama 1: confirmación con 3 semillas")
    plt.tight_layout()
    plt.show()
    return tabla.drop(columns="arquitectura").assign(variante=tabla.variante.map(descripcion))
