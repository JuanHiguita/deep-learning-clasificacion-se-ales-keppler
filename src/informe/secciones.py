"""Una función por figura del notebook, en el orden de sus secciones.

Todas reciben el objeto `Resultados` y leen sus cifras de `resultados/dl/`; ninguna entrena.
Las que tienen parámetros extra (métrica, semilla, número de intervalos…) son las que el
notebook conecta a controles interactivos.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comun import metricas
from comun.metricas import entropia, riesgo_cobertura
from fusion.calibracion import aplicar_temperatura
from informe.datos import CLASES, NOMBRE_DE_METRICA, NOMBRE_DE_RAMA, Resultados
from informe.graficos import (
    COLOR,
    COLOR_DE_CLASE,
    bosque,
    color_si_distingue,
    confiabilidad,
)

RAMAS_TEMPORALES = ("m1_astronet", "m2_promedio", "m2_gru", "m2_atencion")
BANDERAS_DEL_CATALOGO = {
    "fp_ss": "eclipse\nsecundario",
    "fp_co": "centroide",
    "fp_nt": "no parece\ntránsito",
    "fp_ec": "contaminación\npor efemérides",
}
METODOS_DE_INCERTIDUMBRE = (
    ("determinista", "p_determinista", COLOR["no"]),
    ("MC Dropout", "p_mc_dropout", COLOR["demo"]),
    ("ensamble de 5", "p_ensamble", COLOR["curso"]),
)


# ---------------------------------------------------------------------------------------------
# 1 · Los datos
# ---------------------------------------------------------------------------------------------


def resumen_de_datos(res: Resultados) -> None:
    """Cuántos ejemplos hay por clase, cómo quedan las particiones y cuántos tránsitos se ven."""
    fig, (ax_clases, ax_particiones, ax_transitos) = plt.subplots(1, 3, figsize=(13, 3.2))

    conteo = np.bincount(res.y)
    ax_clases.bar(CLASES, conteo, color=COLOR_DE_CLASE)
    for i, n in enumerate(conteo):
        ax_clases.text(i, n + 3, str(n), ha="center")
    ax_clases.set_title("Ejemplos por clase")

    senales = res.tabla[res.tabla.tipo == "transito"]
    por_particion = senales.groupby(["particion", "clase"]).size().unstack()
    por_particion.plot.bar(ax=ax_particiones, color=COLOR_DE_CLASE[:2], legend=False, rot=0)
    ax_particiones.set_title("Planetas y falsos positivos por partición")
    ax_particiones.set_xlabel("partición de prueba")

    n_transitos = res.dataset["n_transitos_validos"]
    for clase in (0, 1):
        ax_transitos.hist(
            np.log10(n_transitos[res.y == clase] + 1),
            bins=25,
            alpha=0.6,
            color=COLOR_DE_CLASE[clase],
            label=CLASES[clase],
        )
    ax_transitos.set_xlabel("log10(tránsitos válidos + 1)")
    ax_transitos.set_title("Tránsitos observados por señal")
    ax_transitos.legend()
    plt.tight_layout()
    plt.show()

    n_vistas = int(res.dataset["disponible_vistas"].sum())
    n_ventanas = int(res.dataset["disponible_transitos"].sum())
    print(
        f"{len(res.ids)} ejemplos de {res.tabla.kepid.nunique()} estrellas · disponibles para la "
        f"rama 1 (vistas): {n_vistas} · para la rama 2 (tránsitos): {n_ventanas}"
    )


# ---------------------------------------------------------------------------------------------
# 2 · El listón
# ---------------------------------------------------------------------------------------------


def _color_de_linea_base(modelo: str) -> str:
    if "demo" in modelo.lower():
        return COLOR["demo"]
    return "#000000" if "koi_score" in modelo else COLOR["base"]


def liston(res: Resultados) -> pd.DataFrame:
    """AUC de los modelos de referencia. Devuelve la tabla para mostrarla debajo."""
    tabla = res.csv("lineas_base")
    etiquetas = [m.replace(", contiene la decisión del Robovetter", "") for m in tabla.modelo]
    colores = [_color_de_linea_base(m) for m in tabla.modelo]

    fig, ax = plt.subplots(figsize=(9, 3.6))
    bosque(ax, etiquetas, tabla.auc, tabla.ic_inf, tabla.ic_sup, colores)
    ax.set_xlabel("AUC planeta vs falso positivo")
    ax.set_xlim(0.72, 1.03)
    ax.set_title("Listón: tabulares (gris), demo vuelta a medir (naranja) y Robovetter (negro)")
    plt.tight_layout()
    plt.show()
    return tabla


# ---------------------------------------------------------------------------------------------
# 3 · P1: arquitecturas
# ---------------------------------------------------------------------------------------------


def _etiqueta_corta(par_de_modelos: str) -> str:
    """'m2_atencion − m1_astronet' → 'atención − AstroNet'."""
    return (
        par_de_modelos.replace("m2_", "")
        .replace("m1_astronet", "AstroNet")
        .replace("atencion", "atención")
    )


def _panel_de_diferencias(ax, tabla: pd.DataFrame, metrica: str, etiquetas) -> None:
    """Diferencias pareadas con intervalo; avisa si la tabla no las tiene para esa métrica."""
    diferencias = tabla[(tabla.tipo == "diferencia") & (tabla.metrica == metrica)]
    if diferencias.empty:
        ax.text(
            0.5,
            0.5,
            "Las diferencias pareadas solo se calcularon\n"
            "para AUC y NLL (así se fijó el protocolo)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return
    bosque(
        ax,
        etiquetas(diferencias),
        diferencias.estimacion,
        diferencias.ic_inf,
        diferencias.ic_sup,
        color_si_distingue(diferencias.distinguible),
        referencia=0,
    )
    ax.set_xlabel("diferencia pareada (mismas estrellas), IC 95 %")
    ax.set_title("¿Se distingue de cero? (verde = sí)")


def arquitecturas(res: Resultados, metrica: str = "auc_planeta_fp") -> pd.DataFrame:
    """Cada arquitectura por separado (izquierda) y sus diferencias pareadas (derecha)."""
    tabla = res.csv("p1_arquitecturas")
    modelos = tabla[(tabla.tipo == "modelo") & (tabla.metrica == metrica)]
    modelos = modelos.set_index("modelo").loc[list(NOMBRE_DE_RAMA)]
    colores = ["#555555", COLOR["base"], COLOR["curso"], COLOR["curso"], COLOR["curso"]]

    fig, (ax_modelos, ax_diferencias) = plt.subplots(1, 2, figsize=(14, 3.6))
    etiquetas = [NOMBRE_DE_RAMA[m] for m in modelos.index]
    bosque(ax_modelos, etiquetas, modelos.estimacion, modelos.ic_inf, modelos.ic_sup, colores)

    # una rayita por semilla, para ver cuánto varía el resultado al reentrenar
    filas = np.arange(len(modelos))[::-1]
    for experimento, fila in zip(modelos.index, filas, strict=True):
        por_semilla = res.json(experimento, "metricas.json")["por_semilla"]
        valores = [s[metrica] for s in por_semilla.values() if metrica in s]
        ax_modelos.plot(valores, [fila - 0.22] * len(valores), "|", color="k", ms=7)
    ax_modelos.set_xlabel(f"{NOMBRE_DE_METRICA[metrica]} · punto = media de 3 semillas")
    ax_modelos.set_title("Cada arquitectura por separado")

    _panel_de_diferencias(
        ax_diferencias, tabla, metrica, lambda d: [_etiqueta_corta(m) for m in d.modelo]
    )
    plt.tight_layout()
    plt.show()
    columnas = ["estimacion", "ic_inf", "ic_sup", "parametros", "desviacion_semillas"]
    return modelos[columnas].rename(index=NOMBRE_DE_RAMA).round(4)


def permutacion(res: Resultados) -> pd.DataFrame:
    """¿Cambia la predicción al barajar el orden de los tránsitos?"""
    tabla = res.csv("p1_permutacion_resumen")
    nombres = [NOMBRE_DE_RAMA[m].replace("Rama 2 · tránsitos + ", "") for m in tabla.modelo]

    fig, (ax_q, ax_auc) = plt.subplots(1, 2, figsize=(11, 2.8))
    ax_q.bar(nombres, tabla.cambio_medio_abs_de_q, color=COLOR["curso"])
    ax_q.set_title("Cambio medio de P(planeta) al barajar el orden")
    ax_auc.bar(nombres, tabla.cambio_de_auc, color=COLOR["curso"])
    ax_auc.axhline(0, color="k", lw=0.8)
    ax_auc.set_title("Cambio de AUC al barajar el orden")
    plt.tight_layout()
    plt.show()
    return tabla


def curvas_de_perdida(res: Resultados, semilla: int = 0) -> None:
    """Pérdida de entrenamiento y de validación interna de cada rama temporal."""
    fig, ejes = plt.subplots(1, len(RAMAS_TEMPORALES), figsize=(15, 2.9), sharey=True)
    for ax, experimento in zip(ejes, RAMAS_TEMPORALES, strict=True):
        for particion in res.json(experimento, f"historial_semilla{semilla}.json"):
            ax.plot(particion["perdida_entrenamiento"], color=COLOR["planeta"], alpha=0.45, lw=0.9)
            ax.plot(particion["perdida_validacion"], color=COLOR["fp"], alpha=0.6, lw=0.9)
            ax.axvline(particion["epoca_elegida"] - 1, color="k", alpha=0.15, lw=0.8)
        ax.set_title(NOMBRE_DE_RAMA[experimento].split("· ")[1], fontsize=9)
        ax.set_xlabel("época")
    ejes[0].set_ylabel("entropía cruzada")
    ejes[0].plot([], [], color=COLOR["planeta"], label="entrenamiento")
    ejes[0].plot([], [], color=COLOR["fp"], label="validación interna")
    ejes[0].legend(fontsize=7)
    plt.suptitle(
        f"Semilla {semilla}, las 5 particiones. "
        "La línea vertical marca la época elegida por parada temprana.",
        y=1.03,
        fontsize=10,
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------------------------
# 4 · Atención
# ---------------------------------------------------------------------------------------------


def pesos_de_atencion(res: Resultados) -> None:
    """Doce ejemplos al azar (semilla fija): sus tránsitos y el peso que les dio la atención."""
    casos = res.npz("m2_atencion", "pesos_atencion_casos.npz")
    por_fila = 6
    fig, ejes = plt.subplots(
        4, por_fila, figsize=(16, 7.5), gridspec_kw={"height_ratios": [2.2, 1, 2.2, 1]}
    )
    for j, id_ejemplo in enumerate(casos["id_ejemplo"]):
        ax_transitos = ejes[(j // por_fila) * 2, j % por_fila]
        ax_pesos = ejes[(j // por_fila) * 2 + 1, j % por_fila]
        n = int(casos["mascara_transitos"][j].sum())
        real, predicha = int(casos["clase"][j]), int(casos["p"][j].argmax())
        acierto = real == predicha

        ax_transitos.imshow(
            casos["transitos"][j, :n], aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2
        )
        ax_transitos.set_xticks([])
        ax_transitos.set_yticks([])
        ax_transitos.set_title(
            f"{str(id_ejemplo).split('|')[0]} · {CLASES[real]}\n"
            f"predicho: {CLASES[predicha]} {'✓' if acierto else '✗'}",
            fontsize=8,
            color="darkgreen" if acierto else "darkred",
        )

        es_par = casos["paridad_transitos"][j, :n] == 0
        ax_pesos.bar(
            np.arange(n),
            casos["pesos"][j, :n],
            color=np.where(es_par, COLOR["par"], COLOR["impar"]),
        )
        ax_pesos.axhline(1 / max(n, 1), color="k", ls="--", lw=0.7)  # peso si todos valieran igual
        ax_pesos.set_xticks([])
        ax_pesos.tick_params(labelsize=7)
    ejes[1, 0].set_ylabel("peso", fontsize=8)
    ejes[3, 0].set_ylabel("peso", fontsize=8)
    plt.suptitle(
        "Pesos de atención por tránsito "
        "(línea punteada = peso uniforme; verde = par, naranja = impar)",
        y=1.0,
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------------------------
# 5 · P2: fusión
# ---------------------------------------------------------------------------------------------


def _color_de_configuracion(clave: str) -> str:
    if clave.startswith("e6a_cuatro"):
        return COLOR["curso"]
    if clave.startswith("e6"):
        return "#b08fd0"
    return COLOR["si"] if "sin" in clave else COLOR["base"]


def fusion(res: Resultados, metrica: str = "auc_planeta_fp") -> None:
    """Fusión, ablación y ramas solas (izquierda); cuánto pierde cada alternativa (derecha)."""
    tabla = res.csv("p2_fusion")
    modelos = tabla[(tabla.tipo == "modelo") & (tabla.metrica == metrica)]

    fig, (ax_modelos, ax_diferencias) = plt.subplots(1, 2, figsize=(15, 4.3))
    colores = [_color_de_configuracion(c) for c in modelos.clave]
    bosque(
        ax_modelos,
        list(modelos.modelo),
        modelos.estimacion,
        modelos.ic_inf,
        modelos.ic_sup,
        colores,
    )
    ax_modelos.set_xlabel(f"{NOMBRE_DE_METRICA[metrica]} · media de 5 semillas de cabeza")
    ax_modelos.set_title("Fusión, ablación y ramas solas")

    if metrica == "auc_planeta_fp":  # el listón tabular solo se midió en AUC
        liston_tabular = res.csv("lineas_base").set_index("modelo")
        for nombre, trazo in (("S3-C (seleccion)", ":"), ("S3-C (tabla_completa)", "--")):
            auc = liston_tabular.loc[nombre, "auc"]
            etiqueta = nombre.replace("seleccion", "selección").replace(
                "tabla_completa", "tabla completa"
            )
            ax_modelos.axvline(auc, color="k", ls=trazo, lw=0.9)
            ax_modelos.text(
                auc, len(modelos) - 0.4, etiqueta, rotation=90, fontsize=7, va="top", ha="right"
            )
        ax_modelos.set_xlim(0.84, 1.005)

    _panel_de_diferencias(
        ax_diferencias,
        tabla,
        metrica,
        lambda d: [m.replace("Fusión (E6a) − ", "− ") for m in d.modelo],
    )
    if ax_diferencias.axison:
        ax_diferencias.set_xlabel("fusión menos el otro modelo (pareada), IC 95 %")
    plt.tight_layout()
    plt.show()


def fusion_en_detalle(res: Resultados) -> None:
    """Matriz de confusión, falsos positivos por tipo y AUC por intensidad de la señal."""
    resumen = res.json("fusion", "e6a_cuatro_ramas", "metricas.json")
    semilla_0 = resumen["semilla_0"]
    fig, (ax_matriz, ax_banderas, ax_mes) = plt.subplots(1, 3, figsize=(14, 3.6))

    matriz = np.array(semilla_0["matriz_confusion"])
    ax_matriz.imshow(matriz, cmap="Purples")
    ax_matriz.grid(False)
    for (fila, columna), n in np.ndenumerate(matriz):
        color = "white" if n > matriz.max() / 2 else "black"
        ax_matriz.text(columna, fila, n, ha="center", va="center", color=color, fontsize=12)
    ax_matriz.set_xticks(range(3), CLASES, fontsize=8)
    ax_matriz.set_yticks(range(3), CLASES, fontsize=8)
    ax_matriz.set_xlabel("predicho")
    ax_matriz.set_ylabel("real")
    exactitud = semilla_0["global"]["exactitud"]["estimacion"]
    ax_matriz.set_title(f"Fusión · matriz de confusión (exactitud {exactitud:.3f})")

    por_bandera = semilla_0["acierto_fp_por_bandera"]
    fraccion = [v["aciertos"] / v["n"] for v in por_bandera.values()]
    ax_banderas.bar([BANDERAS_DEL_CATALOGO[b] for b in por_bandera], fraccion, color=COLOR["fp"])
    for i, v in enumerate(por_bandera.values()):
        ax_banderas.text(
            i, fraccion[i] + 0.01, f"{v['aciertos']}/{v['n']}", ha="center", fontsize=9
        )
    ax_banderas.set_ylim(0, 1.1)
    ax_banderas.set_title("Falsos positivos detectados, por bandera del catálogo")
    ax_banderas.tick_params(axis="x", labelsize=8)

    por_tercil = semilla_0["auc_por_tercil_mes"]
    etiquetas = [f"{nombre}\n(n={v['n']})" for nombre, v in por_tercil.items()]
    ax_mes.bar(etiquetas, [v["estimacion"] for v in por_tercil.values()], color=COLOR["curso"])
    ax_mes.set_ylim(0.9, 1.0)
    ax_mes.set_title("AUC por intensidad de la señal (tercil de MES)")
    plt.tight_layout()
    plt.show()

    apilado = resumen["apilado"]
    print(
        f"Diagnóstico del apilado: NLL de la cabeza en entrenamiento {apilado['nll_entrenamiento']}"
        f" · en validación interna {apilado['nll_validacion']} · razón {apilado['razon']}"
        f" → alerta: {apilado['alerta']}"
    )


# ---------------------------------------------------------------------------------------------
# 6 · P3: calibración e incertidumbre
# ---------------------------------------------------------------------------------------------


def calibracion(res: Resultados, n_intervalos: int = 10) -> None:
    """Confiabilidad antes y después de temperature scaling, y la diferencia con intervalo."""
    e8 = res.json("calibracion", "metricas.json")["E8_temperatura"]
    pred = res.npz("calibracion", "pred_calibrada.npz")
    fig, (ax_antes, ax_despues, ax_diferencia) = plt.subplots(1, 3, figsize=(14, 3.8))

    for ax, momento, clave in (
        (ax_antes, "antes", "p_antes"),
        (ax_despues, "despues", "p_despues"),
    ):
        ece, nll = e8[momento]["ece_top"]["estimacion"], e8[momento]["nll"]["estimacion"]
        titulo = f"{'Antes' if momento == 'antes' else 'Después'} · ECE {ece:.3f} · NLL {nll:.3f}"
        confiabilidad(ax, pred[clave], res.y, titulo, n_intervalos)

    diferencia = e8["diferencia_despues_menos_antes"]
    claves = ("nll", "brier", "ece_top")
    bosque(
        ax_diferencia,
        ["NLL", "Brier", "ECE"],
        [diferencia[c]["estimacion"] for c in claves],
        [diferencia[c]["ic_inf"] for c in claves],
        [diferencia[c]["ic_sup"] for c in claves],
        referencia=0,
    )
    ax_diferencia.set_title("Después − antes (negativo = mejora)")
    ax_diferencia.set_xlabel("diferencia con IC 95 %")
    plt.suptitle(
        f"Temperaturas por partición: {e8['temperaturas']} "
        "(mayor que 1 = el modelo estaba demasiado confiado)",
        y=1.02,
        fontsize=10,
    )
    plt.tight_layout()
    plt.show()


def probar_temperatura(res: Resultados, temperatura: float = 1.0) -> None:
    """Experimento didáctico: una sola temperatura, elegida a mano, sobre toda la prueba.

    Sirve para VER qué hace la temperatura. No es el resultado del informe: allí la temperatura
    de cada partición se ajusta en su validación interna, nunca mirando la prueba.
    """
    p_original = res.npz("calibracion", "pred_calibrada.npz")["p_antes"].astype(float)
    p = aplicar_temperatura(p_original, temperatura)
    ece, nll = metricas.ece_top(p, res.y, 15), metricas.nll(p, res.y)

    fig, (ax_diagrama, ax_confianza) = plt.subplots(1, 2, figsize=(10, 3.6))
    confiabilidad(ax_diagrama, p, res.y, f"T = {temperatura:.2f} · ECE {ece:.3f} · NLL {nll:.3f}")
    ax_confianza.hist(
        p_original.max(axis=1), bins=25, alpha=0.5, color=COLOR["base"], label="T = 1"
    )
    ax_confianza.hist(
        p.max(axis=1), bins=25, alpha=0.6, color=COLOR["curso"], label=f"T = {temperatura:.2f}"
    )
    ax_confianza.set_xlabel("confianza de la clase elegida")
    ax_confianza.set_title("T > 1 baja la confianza; T < 1 la sube")
    ax_confianza.legend()
    plt.tight_layout()
    plt.show()


def prior_explicito(res: Resultados) -> None:
    """Efecto de declarar una tasa base de planetas distinta de la de entrenamiento."""
    e9 = res.json("calibracion", "metricas.json")["E9_prior"]
    por_tasa = pd.DataFrame(e9["por_tasa"]).T.sort_values("tasa")
    diferencia = pd.DataFrame(list(por_tasa.diferencia_corregido_menos_sin))
    fig, (ax_ece, ax_diferencia, ax_umbral) = plt.subplots(1, 3, figsize=(15, 3.6))

    ax_ece.plot(
        por_tasa.tasa,
        por_tasa.ece_ponderado_sin_corregir,
        "o-",
        color=COLOR["fp"],
        label="sin corregir",
    )
    ax_ece.plot(
        por_tasa.tasa,
        por_tasa.ece_ponderado_corregido,
        "o-",
        color=COLOR["curso"],
        label="con prior explícito",
    )
    ax_ece.set_xscale("log")
    ax_ece.set_xlabel("tasa base de planetas declarada")
    ax_ece.set_ylabel("ECE ponderado")
    ax_ece.set_title("Error de calibración según la tasa base")
    ax_ece.legend()

    mejora = [COLOR["si"] if superior < 0 else COLOR["no"] for superior in diferencia.ic_sup]
    bosque(
        ax_diferencia,
        [f"tasa {t:g}" for t in por_tasa.tasa],
        diferencia.estimacion,
        diferencia.ic_inf,
        diferencia.ic_sup,
        mejora,
        referencia=0,
    )
    ax_diferencia.set_title("ECE corregido − sin corregir (verde = mejora distinguible)")

    # cada entrada es [antes de corregir, después de corregir]: se dibuja el "después"
    for columna, etiqueta in (
        ("candidatos_con_q_mayor_0.5", "P(planeta) > 0,5"),
        ("candidatos_con_q_mayor_0.9", "P(planeta) > 0,9"),
    ):
        ax_umbral.plot(por_tasa.tasa, [par[1] for par in por_tasa[columna]], "o-", label=etiqueta)
    ax_umbral.set_xscale("log")
    ax_umbral.set_xlabel("tasa base declarada")
    ax_umbral.set_title("Señales que superan el umbral tras corregir")
    ax_umbral.legend()
    plt.tight_layout()
    plt.show()
    print(
        "Tasa base medida en DR25 (periodo 1–30 d):",
        e9["tasa_dr25"],
        "· tasa de entrenamiento:",
        e9["tasa_entrenamiento_media"],
    )


def incertidumbre(res: Resultados, cobertura: float = 1.0) -> pd.DataFrame:
    """¿El modelo duda cuando se equivoca? `cobertura` = fracción de casos que se aceptan."""
    pred = res.npz("incertidumbre", "predicciones.npz")
    fig, (ax_riesgo, ax_entropia, ax_metricas) = plt.subplots(1, 3, figsize=(15, 3.7))

    lectura = []
    for nombre, clave, color in METODOS_DE_INCERTIDUMBRE:
        p = pred[clave]
        acierto = (p.argmax(axis=1) == res.y).astype(float)
        coberturas, riesgo, _ = riesgo_cobertura(-entropia(p), acierto)
        ax_riesgo.plot(coberturas, riesgo, color=color, label=nombre)
        aceptados = max(int(round(cobertura * len(p))), 1)
        lectura.append(f"{nombre}: {riesgo[aceptados - 1]:.1%}")
    ax_riesgo.axvline(cobertura, color="k", ls="--", lw=0.8)
    ax_riesgo.set_xlabel("cobertura: fracción de casos aceptados, de más a menos seguros")
    ax_riesgo.set_ylabel("error entre los aceptados")
    ax_riesgo.set_title("Riesgo frente a cobertura")
    ax_riesgo.legend()

    p_ensamble = pred["p_ensamble"]
    acierto = p_ensamble.argmax(axis=1) == res.y
    duda = entropia(p_ensamble)
    ax_entropia.hist(
        duda[acierto],
        bins=30,
        alpha=0.6,
        color=COLOR["si"],
        density=True,
        label=f"aciertos ({acierto.sum()})",
    )
    ax_entropia.hist(
        duda[~acierto],
        bins=15,
        alpha=0.6,
        color=COLOR["fp"],
        density=True,
        label=f"errores ({(~acierto).sum()})",
    )
    ax_entropia.set_xlabel("entropía predictiva (ensamble)")
    ax_entropia.set_title("La duda es mayor en los errores")
    ax_entropia.legend()

    tabla = res.csv("p3_incertidumbre").set_index("metodo")
    tabla[["nll", "ece_top", "aurc"]].plot.bar(ax=ax_metricas, rot=0)
    ax_metricas.set_xticklabels(["determinista", "MC Dropout", "ensamble"])
    ax_metricas.set_title("Menor es mejor")
    plt.tight_layout()
    plt.show()

    print(f"Aceptando el {cobertura:.0%} más seguro, el error es → " + " · ".join(lectura))
    diferencia = res.json("incertidumbre", "metricas.json")["diferencia_nll_ensamble_menos_mc"]
    print("NLL ensamble − MC Dropout:", diferencia)
    return tabla.round(4)


# ---------------------------------------------------------------------------------------------
# 8 · Veredicto
# ---------------------------------------------------------------------------------------------


def hipotesis(res: Resultados) -> pd.DataFrame:
    """Todas las hipótesis en una figura. Devuelve la tabla, con una marca en las que se sostienen."""
    tabla = res.csv("resumen_hipotesis")
    se_sostiene = tabla.veredicto == "se sostiene"

    fig, ax = plt.subplots(figsize=(11, 4.2))
    etiquetas = [f"{h} · {c}" for h, c in zip(tabla.hipotesis, tabla.cantidad, strict=True)]
    bosque(
        ax,
        etiquetas,
        tabla.estimacion,
        tabla.ic_inf,
        tabla.ic_sup,
        color_si_distingue(se_sostiene),
        referencia=0,
    )
    ax.set_title("Todas las hipótesis: diferencia con IC 95 % (verde = se sostiene)")
    plt.tight_layout()
    plt.show()

    tabla.insert(0, "", np.where(se_sostiene, "✔", "·"))
    return tabla.round(4)
