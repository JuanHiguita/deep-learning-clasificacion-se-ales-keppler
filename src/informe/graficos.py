"""Piezas de dibujo reutilizables del notebook.

bosque          puntos con intervalo de confianza (la figura más usada del informe)
confiabilidad   diagrama de confiabilidad (confianza declarada frente a acierto real)
ver_ejemplo     una señal vista de las tres formas en que la leen las ramas
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from informe.datos import CLASES, Resultados

COLOR = {
    "planeta": "#1f77b4",
    "fp": "#d62728",
    "sin_senal": "#7f7f7f",
    "curso": "#6a3d9a",  # modelos de este curso
    "base": "#9e9e9e",  # líneas base
    "demo": "#e08a00",  # código de la demo (anterior a los cursos)
    "si": "#2a9d8f",  # diferencia que se distingue de cero
    "no": "#888888",  # diferencia que no se distingue de cero
    "par": "#2ca02c",
    "impar": "#ff7f0e",
}
COLOR_DE_CLASE = (COLOR["planeta"], COLOR["fp"], COLOR["sin_senal"])
# Canales de la vista local: (nombre, color)
CANALES_VISTA_LOCAL = (
    ("todos", "#333333"),
    ("pares", COLOR["par"]),
    ("impares", COLOR["impar"]),
    ("fase 0,5", "#9467bd"),
)
# Columnas centrales de la ventana de un tránsito (61 puntos): ahí se mide su profundidad
CENTRO_DEL_TRANSITO = slice(27, 34)


def aplicar_estilo() -> None:
    """Estilo común de todas las figuras."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )


def color_si_distingue(distinguible) -> list[str]:
    """Verde si la diferencia se distingue de cero, gris si no."""
    return [COLOR["si"] if d else COLOR["no"] for d in distinguible]


def bosque(ax, etiquetas, estimacion, ic_inf, ic_sup, colores=None, referencia=None) -> None:
    """Un punto por fila con su intervalo de confianza del 95 %.

    `referencia` dibuja una línea vertical (por ejemplo en 0 para diferencias).
    """
    filas = np.arange(len(etiquetas))[::-1]  # la primera etiqueta queda arriba
    colores = colores or ["#333333"] * len(etiquetas)
    datos = zip(filas, estimacion, ic_inf, ic_sup, colores, strict=True)
    for fila, valor, inferior, superior, color in datos:
        ax.plot([inferior, superior], [fila, fila], color=color, lw=2.2, solid_capstyle="round")
        ax.plot(valor, fila, "o", color=color, ms=6)
        ax.text(superior, fila, f"  {valor:.3f}", va="center", fontsize=8.5, color=color)
    ax.set_yticks(filas)
    ax.set_yticklabels(etiquetas)
    if referencia is not None:
        ax.axvline(referencia, color="k", lw=0.8, ls="--")
    ax.grid(axis="y", visible=False)


def confiabilidad(ax, p: np.ndarray, y: np.ndarray, titulo: str, n_intervalos: int = 10) -> None:
    """Diagrama de confiabilidad de la clase más probable.

    Se agrupan las predicciones por su confianza; en cada grupo se compara la confianza media
    con la fracción de aciertos. Calibrado = puntos sobre la diagonal. El tamaño del punto
    crece con el número de ejemplos del grupo; los grupos con menos de 5 no se dibujan.
    """
    confianza = p.max(axis=1)
    acierto = (p.argmax(axis=1) == y).astype(float)
    grupo = np.clip((confianza * n_intervalos).astype(int), 0, n_intervalos - 1)
    xs, ys, tamanos = [], [], []
    for g in range(n_intervalos):
        en_grupo = grupo == g
        if en_grupo.sum() >= 5:
            xs.append(confianza[en_grupo].mean())
            ys.append(acierto[en_grupo].mean())
            tamanos.append(np.sqrt(en_grupo.sum()) * 12)
    ax.plot([0.3, 1], [0.3, 1], "k--", lw=0.8)
    ax.scatter(xs, ys, s=tamanos, color=COLOR["curso"], alpha=0.75, zorder=3)
    ax.plot(xs, ys, color=COLOR["curso"], lw=1)
    ax.set_xlabel("confianza declarada")
    ax.set_ylabel("acierto real")
    ax.set_title(titulo)
    ax.set_xlim(0.3, 1.02)
    ax.set_ylim(0.3, 1.02)


def ver_ejemplo(res: Resultados, id_ejemplo: str) -> None:
    """Una señal vista de las tres formas en que la leen las ramas temporales.

    1. Vista global: la órbita completa plegada (rama 1).
    2. Vista local: el tránsito ampliado, con pares, impares y fase 0,5 (rama 1).
    3. Tránsitos uno por uno: cada fila es un tránsito (rama 2).
    4. Profundidad de cada tránsito, coloreada por paridad.
    """
    i = res.posicion(id_ejemplo)
    d = res.dataset
    fig, ejes = plt.subplots(
        1, 4, figsize=(15, 3.3), gridspec_kw={"width_ratios": [1.3, 1, 1.25, 0.9]}
    )
    ax_global, ax_local, ax_transitos, ax_profundidad = ejes

    fase = np.linspace(-0.5, 0.5, d["vista_global"].shape[1])
    ax_global.plot(fase, d["vista_global"][i, :, 0], lw=0.7, color="#333333")
    ax_global.set_title("Vista global (órbita completa)")
    ax_global.set_xlabel("fase")

    duraciones = np.linspace(-4, 4, d["vista_local"].shape[1])
    for canal, (nombre, color) in enumerate(CANALES_VISTA_LOCAL):
        ax_local.plot(duraciones, d["vista_local"][i, :, canal], lw=1.1, color=color, label=nombre)
    ax_local.set_title("Vista local")
    ax_local.set_xlabel("duraciones del tránsito")
    ax_local.legend(fontsize=7)

    n = int(d["mascara_transitos"][i].sum())  # tránsitos que entran al modelo (máximo 32)
    n_filas = max(n, 1)
    imagen = ax_transitos.imshow(
        d["transitos"][i, :n_filas],
        aspect="auto",
        cmap="RdBu_r",
        vmin=-2,
        vmax=2,
        extent=[-2.5, 2.5, n_filas, 0],
    )
    total = int(d["n_transitos_validos"][i])
    ax_transitos.set_title(f"Tránsitos uno por uno ({n} de {total})")
    ax_transitos.set_xlabel("duraciones")
    ax_transitos.set_ylabel("tránsito")
    plt.colorbar(imagen, ax=ax_transitos, fraction=0.045)

    profundidad = -d["transitos"][i, :n, CENTRO_DEL_TRANSITO].mean(axis=1)
    es_par = d["paridad_transitos"][i, :n] == 0
    ax_profundidad.barh(
        np.arange(n), profundidad, color=np.where(es_par, COLOR["par"], COLOR["impar"])
    )
    ax_profundidad.invert_yaxis()
    ax_profundidad.set_title("Profundidad por tránsito\n(verde = par, naranja = impar)")
    ax_profundidad.set_xlabel("profundidad relativa")

    fila = res.tabla.iloc[i]
    par_menos_impar = d["consistencia"][i, 1]
    fig.suptitle(
        f"{id_ejemplo} · clase real: {CLASES[fila.clase]} · periodo {fila.periodo_d:.2f} d · "
        f"par−impar = {par_menos_impar:+.2f} (escala log)",
        y=1.04,
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()
