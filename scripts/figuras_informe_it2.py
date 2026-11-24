"""Figuras del informe explicativo de la iteración 2 (casos reales, elegidos al azar con semilla fija).

Lee solo archivos del repositorio y escribe en iteracion_2/evidencias/figuras/informe_*.png.
Uso: python scripts/figuras_informe_it2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

IT2 = Path("iteracion_2")
FIG = IT2 / "evidencias" / "figuras"
RES = IT2 / "evidencias" / "resultados"
SEMILLA = 20261124
CLASES = ("planeta", "falso positivo", "sin señal")
AZUL, ROJO, VERDE, GRIS = "#2166ac", "#b2182b", "#1b7837", "#737373"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def guardar(fig, nombre):
    fig.savefig(FIG / f"informe_{nombre}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def predicciones():
    ps, ids = [], None
    for s in range(5):
        with np.load(RES / "fusion" / "tres_ramas" / f"pred_semilla{s}.npz") as d:
            ids = [str(i) for i in d["id_ejemplo"]]
            ps.append(d["p_prueba"].astype(float))
    return ids, np.mean(ps, axis=0)


def caso(d, i, p, titulo, archivo):
    """Las entradas de un ejemplo real y la respuesta del sistema."""
    fig = plt.figure(figsize=(12, 5.6))
    g = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.45, wspace=0.28)
    a = fig.add_subplot(g[0, :2])
    a.plot(d["vista_global"][i, :, 0], lw=0.6, color=AZUL)
    a.set_title("Vista global: la órbita completa plegada (2.001 puntos)", loc="left")
    a.set_xlabel("fase orbital (el tránsito está en el centro)")
    b = fig.add_subplot(g[0, 2])
    b.bar(CLASES, p, color=[VERDE, ROJO, GRIS])
    for k, v in enumerate(p):
        b.text(k, v + 0.02, f"{v:.2f}", ha="center")
    b.set_ylim(0, 1.12)
    b.set_title("Respuesta de la fusión", loc="left")
    c = fig.add_subplot(g[1, 0])
    loc = d["vista_local"][i]
    c.plot(loc[:, 1], color=AZUL, lw=1, label="tránsitos pares")
    c.plot(loc[:, 2], color=ROJO, lw=1, label="tránsitos impares")
    c.legend(fontsize=8, loc="lower left")
    c.set_title("Vista local: pares frente a impares", loc="left")
    e = fig.add_subplot(g[1, 1])
    e.plot(loc[:, 0], color=AZUL, lw=1, label="tránsito (fase 0)")
    e.plot(loc[:, 3], color=VERDE, lw=1, label="media órbita (fase 0,5)")
    e.legend(fontsize=8, loc="lower left")
    e.set_title("¿Hay un eclipse secundario?", loc="left")
    f = fig.add_subplot(g[1, 2])
    m = d["mascara_transitos"][i] > 0.5
    f.imshow(np.where(m[:, None], d["transitos"][i], np.nan), aspect="auto", cmap="viridis",
             interpolation="nearest")
    f.grid(False)
    f.set_title(f"Rama 2: {int(m.sum())} tránsitos, uno por fila", loc="left")
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=11, fontweight="bold")
    guardar(fig, archivo)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    ej = pd.read_csv(IT2 / "datos" / "ejemplos.csv").set_index("id_ejemplo")
    with np.load(IT2 / "datos" / "dataset_temporal.npz", allow_pickle=False) as a:
        d = {k: a[k] for k in a.files}
    ids_d = {str(x): j for j, x in enumerate(d["id_ejemplo"])}
    ids, p = predicciones()
    t = ej.loc[ids].reset_index()
    y = t["clase"].to_numpy()
    pred = p.argmax(1)
    ok = np.array([d["disponible_vistas"][ids_d[x]] == 1 and d["disponible_transitos"][ids_d[x]] == 1
                   for x in ids])
    rng = np.random.default_rng(SEMILLA)
    elegidos = {}

    def elegir(nombre, mascara):
        j = int(rng.choice(np.flatnonzero(mascara & ok)))
        elegidos[nombre] = {"id": ids[j], "real": CLASES[y[j]], "predicho": CLASES[pred[j]],
                            "p": [round(float(v), 4) for v in p[j]], "periodo_d": float(t.loc[j, "periodo_d"]),
                            "mes": float(t.loc[j, "mes"]), "tipo_fp": str(t.loc[j, "tipo_fp"])}
        return j

    casos = [
        ("planeta_acierto", (y == 0) & (pred == 0), "Caso real 1 · planeta confirmado, bien clasificado"),
        ("eclipse_acierto", (y == 1) & (pred == 1) & (t["tipo_fp"] == "eclipse_secundario").to_numpy(),
         "Caso real 2 · falso positivo (eclipse estelar), bien clasificado"),
        ("error_fp_como_planeta", (y == 1) & (pred == 0), "Caso real 3 · ERROR: falso positivo tomado por planeta"),
        ("error_planeta_como_fp", (y == 0) & (pred == 1), "Caso real 4 · ERROR: planeta tomado por falso positivo"),
    ]
    for nombre, mascara, titulo in casos:
        j = elegir(nombre, mascara)
        e = elegidos[nombre]
        caso(d, ids_d[ids[j]], p[j], f"{titulo}  ·  {ids[j].split('|')[0]}  ·  periodo {e['periodo_d']:.2f} d", nombre)

    # Iteración 1 frente a 2 (cada una con su muestra; no es una diferencia pareada)
    it1 = pd.read_csv("resultados/dl/tablas/lineas_base.csv").set_index("modelo")["auc"]
    f1 = json.load(open("resultados/dl/fusion/e6a_cuatro_ramas/metricas.json"))["semilla_0"]["global"]["auc_planeta_fp"]
    c2 = json.load(open(RES / "fusion" / "comparacion_con_tabulares.json"))["auc"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    filas = [("Tabular sin centroide", it1["S3-B (seleccion)"], c2["tabular_sin_centroide"]["estimacion"]),
             ("Tabular con centroide", it1["S3-C (seleccion)"], c2["tabular_con_centroide"]["estimacion"]),
             ("Fusión de la red", f1["estimacion"], c2["tres_ramas"]["estimacion"])]
    x = np.arange(len(filas))
    ax.bar(x - 0.18, [f[1] for f in filas], 0.36, color="#9ecae1", label="Iteración 1 (245 señales)")
    ax.bar(x + 0.18, [f[2] for f in filas], 0.36, color=AZUL, label="Iteración 2 (4.471 señales)")
    for k, f in enumerate(filas):
        ax.text(k - 0.18, f[1] + 0.003, f"{f[1]:.3f}", ha="center", fontsize=8)
        ax.text(k + 0.18, f[2] + 0.003, f"{f[2]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, [f[0] for f in filas])
    ax.set_ylim(0.84, 1.0)
    ax.set_ylabel("AUC planeta vs. FP")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Lo que cambió al pasar a más datos (muestras distintas: comparar con cautela)", loc="left")
    guardar(fig, "iteraciones")

    # Aporte de cada parte (diferencias pareadas)
    dif = json.load(open(RES / "fusion" / "comparacion_con_tabulares.json"))["diferencias"]
    claves = [("solo_curvas - tabular_sin_centroide", "Solo curvas − tabular sin centroide"),
              ("tres_ramas - tabular_con_centroide", "Fusión − tabular con centroide"),
              ([k for k in dif if "aporte de las curvas" in k][0], "Fusión − solo escalares\n(lo que suman las curvas)"),
              ([k for k in dif if "aporte de los escalares" in k][0], "Fusión − solo curvas\n(lo que suman los escalares)"),
              ([k for k in dif if "efecto del apilado" in k][0], "Fusión − ajuste cruzado\n(efecto del apilado)"),
              ("solo_curvas - tabular_con_centroide", "Solo curvas − tabular con centroide")]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for k, (c, nombre) in enumerate(claves[::-1]):
        v = dif[c]
        color = VERDE if v["ic_inf"] > 0 else (ROJO if v["ic_sup"] < 0 else GRIS)
        ax.plot([v["ic_inf"], v["ic_sup"]], [k, k], color=color, lw=2.5)
        ax.plot(v["estimacion"], k, "o", color=color)
        ax.annotate(f"{v['estimacion']:+.4f}", (v["ic_sup"], k), xytext=(6, -3), textcoords="offset points", fontsize=8)
    ax.axvline(0, color=GRIS, ls="--", lw=1)
    ax.set_yticks(range(len(claves)), [n for _, n in claves[::-1]], fontsize=8)
    ax.set_xlabel("diferencia de AUC (intervalo del 95 %, bootstrap por estrella)")
    ax.set_title("Diferencias pareadas: verde = gana, rojo = pierde, gris = empate", loc="left")
    ax.margins(x=0.15)
    guardar(fig, "diferencias")

    # Prerregistradas frente a falsos positivos normales
    det = pd.read_csv(RES / "prerregistrados" / "detalle.csv")
    q_fp = p[y == 1, 0] / (p[y == 1, 0] + p[y == 1, 1])
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bins = np.linspace(0, 1, 21)
    ax.hist(q_fp, bins=bins, density=True, alpha=0.6, color=GRIS, label=f"falsos positivos normales (n={len(q_fp)})")
    ax.hist(det["q_planeta"], bins=bins, density=True, alpha=0.7, color=VERDE, label=f"señales reabiertas (n={len(det)})")
    ax.set_yscale("log")
    ax.set_xlabel("probabilidad de planeta que da la fusión")
    ax.set_ylabel("densidad (escala log)")
    ax.legend(fontsize=8)
    ax.set_title("Las señales que la comunidad reabrió reciben más probabilidad de planeta", loc="left")
    guardar(fig, "prerregistradas")

    # Clases y tamaño
    fig, ax = plt.subplots(figsize=(6, 2.6))
    cnt = pd.Series(y).value_counts().sort_index()
    ax.barh([CLASES[i] for i in cnt.index], cnt.values, color=[VERDE, ROJO, GRIS])
    for k, v in enumerate(cnt.values):
        ax.text(v + 40, k, f"{v:,}".replace(",", "."), va="center")
    ax.set_title("Ejemplos evaluados por clase (iteración 2)", loc="left")
    ax.margins(x=0.15)
    guardar(fig, "clases")

    (FIG / "informe_casos.json").write_text(json.dumps(elegidos, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(elegidos, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
