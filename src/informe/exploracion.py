"""Figuras del notebook `01_exploracion_datos.ipynb`: qué muestran los datos y qué rama motivan.

Todo es descriptivo: se mira el dataset completo con sus etiquetas para entender el problema.
Nada de aquí elige un modelo ni un hiperparámetro.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comun import metricas
from informe.datos import CLASES, Resultados
from informe.graficos import CANALES_VISTA_LOCAL, COLOR, COLOR_DE_CLASE

NOMBRE_DE_ESCALAR = {
    "log10_periodo_d": "log10 del periodo (d)",
    "log10_duracion_h": "log10 de la duración (h)",
    "profundidad_medida_ppm": "profundidad medida (ppm)",
    "snr_medido": "señal/ruido medida",
    "log10_n_transitos_validos": "log10 de tránsitos válidos",
    "koi_dikco_msky": "centroide respecto al catálogo (″)",
    "koi_dicco_msky": "centroide fuera−dentro (″)",
}
NOMBRE_DE_CONSISTENCIA = (
    "dispersión de la profundidad entre tránsitos",
    "diferencia impar − par",
    "log10(1 + tránsitos válidos)",
    "tránsito más desviado",
)
FUERA_DEL_TRANSITO = slice(0, 15)  # primeros puntos de la ventana de 61: antes del tránsito


def _log_con_signo(x: np.ndarray) -> np.ndarray:
    """sign(x)·log10(1 + |x|): deja ver a la vez valores de 1 y de 10 000."""
    return np.sign(x) * np.log10(1 + np.abs(x))


def _senales_reales(res: Resultados) -> np.ndarray:
    """Máscara de los ejemplos con señal real (planeta o falso positivo)."""
    return res.y < 2


# ---------------------------------------------------------------------------------------------
# Rama 4: ¿qué dicen los escalares?
# ---------------------------------------------------------------------------------------------


def escalares_por_clase(res: Resultados) -> pd.DataFrame:
    """Distribución de cada escalar en planetas y falsos positivos, y cuántas veces falta."""
    nombres = [str(n) for n in res.dataset["nombres_escalares"]]
    valores, faltantes = res.dataset["escalares"], res.dataset["faltantes"].astype(bool)
    real = _senales_reales(res)

    fig, ejes = plt.subplots(2, 4, figsize=(15, 5.6))
    filas = []
    for j, (ax, nombre) in enumerate(zip(ejes.ravel(), nombres, strict=False)):
        es_centroide_o_snr = "msky" in nombre or "snr" in nombre or "ppm" in nombre
        transformar = _log_con_signo if es_centroide_o_snr else (lambda x: x)
        con_dato = real & ~faltantes[:, j]
        limites = np.percentile(transformar(valores[con_dato, j]), [1, 99])
        for clase in (0, 1):
            x = transformar(valores[con_dato & (res.y == clase), j])
            ax.hist(x, bins=25, range=tuple(limites), alpha=0.6, color=COLOR_DE_CLASE[clase])
        escala = " · escala log" if es_centroide_o_snr else ""
        ax.set_title(NOMBRE_DE_ESCALAR[nombre] + escala, fontsize=9)
        filas.append(
            {
                "escalar": NOMBRE_DE_ESCALAR[nombre],
                "falta en planetas": int((faltantes[:, j] & (res.y == 0)).sum()),
                "falta en falsos positivos": int((faltantes[:, j] & (res.y == 1)).sum()),
                "falta en 'sin señal'": int((faltantes[:, j] & (res.y == 2)).sum()),
            }
        )
    ejes.ravel()[-1].set_axis_off()
    ejes[0, 0].legend(CLASES[:2], fontsize=8)
    plt.suptitle("Escalares de la rama 4: planetas (azul) frente a falsos positivos (rojo)", y=1.0)
    plt.tight_layout()
    plt.show()
    return pd.DataFrame(filas)


def poder_de_cada_variable(res: Resultados) -> pd.DataFrame:
    """AUC planeta vs FP de cada variable por sí sola (0,5 = no separa; 1 = separa perfecto).

    Es una descripción con todos los datos, no un resultado: sirve para ver qué información
    separa las clases antes de entrenar nada.
    """
    real = _senales_reales(res)
    es_planeta = res.y[real] == 0
    variables = {}
    faltantes = res.dataset["faltantes"].astype(bool)
    for j, nombre in enumerate(res.dataset["nombres_escalares"]):
        valores = np.where(faltantes[:, j], np.nan, res.dataset["escalares"][:, j])
        variables[NOMBRE_DE_ESCALAR[str(nombre)]] = ("escalar (rama 4)", valores[real])
    for j, nombre in enumerate(NOMBRE_DE_CONSISTENCIA):
        valores = np.abs(res.dataset["consistencia"][:, j])
        variables[nombre] = ("entre tránsitos (rama 2)", valores[real])

    filas = []
    for nombre, (origen, valores) in variables.items():
        con_dato = np.isfinite(valores)
        auc = metricas.auc(valores[con_dato], es_planeta[con_dato])
        filas.append(
            {
                "variable": nombre,
                "origen": origen,
                "auc": round(max(auc, 1 - auc), 3),  # sin importar el sentido
                "n": int(con_dato.sum()),
            }
        )
    tabla = pd.DataFrame(filas).sort_values("auc")

    fig, ax = plt.subplots(figsize=(9, 3.8))
    colores = [COLOR["curso"] if "rama 2" in o else COLOR["base"] for o in tabla.origen]
    ax.barh(tabla.variable, tabla.auc, color=colores)
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("AUC planeta vs FP de la variable sola")
    ax.set_title("Qué separa las clases: escalares (gris) y estadísticos entre tránsitos (morado)")
    plt.tight_layout()
    plt.show()
    return tabla.iloc[::-1].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Rama 1: ¿la forma del tránsito distingue las clases?
# ---------------------------------------------------------------------------------------------


def forma_media_por_clase(res: Resultados) -> None:
    """Vista local típica de cada clase: mediana y franja entre los cuartiles, por canal."""
    vista = res.dataset["vista_local"]
    disponible = res.dataset["disponible_vistas"].astype(bool)
    duraciones = np.linspace(-4, 4, vista.shape[1])

    fig, ejes = plt.subplots(1, len(CANALES_VISTA_LOCAL), figsize=(15, 3.3), sharey=True)
    for canal, (ax, (nombre, _)) in enumerate(zip(ejes, CANALES_VISTA_LOCAL, strict=True)):
        for clase, color in enumerate(COLOR_DE_CLASE):
            series = vista[disponible & (res.y == clase), :, canal]
            q1, mediana, q3 = np.percentile(series, [25, 50, 75], axis=0)
            if clase == 2:  # sin señal: solo la mediana; su franja de ruido taparía a las demás
                ax.plot(duraciones, mediana, color=color, lw=0.8, ls=":", label=CLASES[clase])
                continue
            ax.plot(duraciones, mediana, color=color, lw=1.4, label=CLASES[clase])
            ax.fill_between(duraciones, q1, q3, color=color, alpha=0.22)
        ax.set_ylim(-1.9, 0.6)
        ax.set_title(f"Canal «{nombre}»")
        ax.set_xlabel("duraciones del tránsito")
    ejes[0].set_ylabel("brillo en unidades de profundidad")
    ejes[0].legend(fontsize=8)
    plt.suptitle(
        "Forma típica del tránsito por clase (línea = mediana; franja = cuartiles)", y=1.02
    )
    plt.tight_layout()
    plt.show()

    centro = vista.shape[1] // 2
    alrededor = slice(centro - 5, centro + 6)
    for clase in (0, 1):
        series = vista[disponible & (res.y == clase)]
        fondo = np.median(series[:, centro, 0])
        angosto = np.median((series[:, :, 0] < -0.5).mean(axis=1)) * (
            duraciones[-1] - duraciones[0]
        )
        secundario = np.percentile(series[:, alrededor, 3].mean(axis=1), 10)
        print(
            f"{CLASES[clase]}: fondo típico {fondo:+.2f} · ancho a media profundidad "
            f"{angosto:.2f} duraciones · el 10 % con más caída en fase 0,5 baja de {secundario:+.2f}"
        )


# ---------------------------------------------------------------------------------------------
# Rama 2: ¿hay información en los tránsitos uno por uno?
# ---------------------------------------------------------------------------------------------


def transitos_por_senal(res: Resultados, maximo: int = 32, minimo: int = 4) -> None:
    """Cuántos tránsitos tiene cada señal, frente al mínimo y al máximo que usa la rama 2."""
    n = res.dataset["n_transitos_validos"]
    real = _senales_reales(res)
    fig, (ax_hist, ax_periodo) = plt.subplots(1, 2, figsize=(12, 3.3))

    cortes = np.logspace(0, np.log10(max(n.max(), 2)), 30)
    for clase in (0, 1):
        ax_hist.hist(n[res.y == clase], bins=cortes, alpha=0.6, color=COLOR_DE_CLASE[clase])
    for valor, texto in ((minimo, "mínimo"), (maximo, "máximo que entra")):
        ax_hist.axvline(valor, color="k", ls="--", lw=0.8)
        ax_hist.text(valor, ax_hist.get_ylim()[1] * 0.92, f" {texto} ({valor})", fontsize=8)
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel("tránsitos válidos observados en 4 años")
    ax_hist.set_title("Tránsitos por señal")
    ax_hist.legend(CLASES[:2], fontsize=8)

    ax_periodo.scatter(
        res.tabla.periodo_d[real], n[real], s=10, c=np.array(COLOR_DE_CLASE)[res.y[real]]
    )
    ax_periodo.axhline(maximo, color="k", ls="--", lw=0.8)
    ax_periodo.set_xscale("log")
    ax_periodo.set_yscale("log")
    ax_periodo.set_xlabel("periodo (d)")
    ax_periodo.set_ylabel("tránsitos válidos")
    ax_periodo.set_title("A menor periodo, más tránsitos")
    plt.tight_layout()
    plt.show()

    con_mas = (n[real] > maximo).mean()
    print(
        f"Mediana: {int(np.median(n[real]))} tránsitos por señal. El {con_mas:.0%} de las señales "
        f"tiene más de {maximo}: de esas se toma una muestra repartida en el tiempo."
    )


def consistencia_entre_transitos(res: Resultados) -> None:
    """Estadísticos que la curva plegada borra: paridad, dispersión y tránsito más desviado."""
    consistencia = res.dataset["consistencia"]
    disponible = res.dataset["disponible_transitos"].astype(bool)
    indices = (1, 0, 3)
    fig, ejes = plt.subplots(1, len(indices), figsize=(14, 3.2))
    for ax, j in zip(ejes, indices, strict=True):
        valores = np.abs(consistencia[:, j])
        for clase in (0, 1):
            ax.hist(
                valores[disponible & (res.y == clase)],
                bins=30,
                alpha=0.6,
                color=COLOR_DE_CLASE[clase],
            )
        ax.set_title(NOMBRE_DE_CONSISTENCIA[j], fontsize=10)
        ax.set_xlabel("|valor| en log10(1 + σ)")
    ejes[0].legend(CLASES[:2], fontsize=8)
    plt.suptitle("Lo que se ve tránsito por tránsito y no en la curva plegada", y=1.02)
    plt.tight_layout()
    plt.show()


def ruido_de_un_transito(res: Resultados) -> None:
    """Qué tan ruidoso es UN tránsito: ruido fuera del tránsito, en unidades de profundidad.

    Un valor de 1 significa que el ruido de cada punto es tan grande como el tránsito mismo.
    """
    transitos = res.dataset["transitos"]
    mascara = res.dataset["mascara_transitos"].astype(bool)
    disponible = res.dataset["disponible_transitos"].astype(bool)
    ruido = np.full(len(res.ids), np.nan)
    for i in np.flatnonzero(disponible):
        por_transito = transitos[i, mascara[i]][:, FUERA_DEL_TRANSITO].std(axis=1)
        ruido[i] = np.median(por_transito)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    cortes = np.logspace(-3, 0.5, 35)
    for clase in (0, 1):
        ax.hist(ruido[res.y == clase], bins=cortes, alpha=0.6, color=COLOR_DE_CLASE[clase])
    ax.axvline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("ruido por punto de un tránsito ÷ profundidad")
    ax.set_title("¿Se ve un tránsito individual? (a la izquierda de 1: sí)")
    ax.legend(CLASES[:2], fontsize=8)
    plt.tight_layout()
    plt.show()

    real = _senales_reales(res) & np.isfinite(ruido)
    for clase in (0, 1):
        de_clase = real & (res.y == clase)
        ruidosos = (ruido[de_clase] > 0.5).mean()
        print(
            f"{CLASES[clase]}: mediana {np.median(ruido[de_clase]):.2f}; en el {ruidosos:.0%} el "
            "ruido por punto supera la mitad de la profundidad."
        )
