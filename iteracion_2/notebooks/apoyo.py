"""Apoyo de los cuadernos de la iteración 2: rutas, lectura de evidencias y figuras.

Todo se lee de archivos que están dentro del repositorio (carpeta `iteracion_2/`), con rutas
relativas a su raíz. Solo hace falta numpy, pandas y matplotlib: los cuadernos NO entrenan nada,
muestran y analizan lo que los pasos 1 a 5 dejaron guardado. Toda cifra sale de esos archivos.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AUTORES = (
    "Thomas Monnier Granda · Esteban Felipe Genes Noya · Juan Sebastian Higuita Riascos · "
    "Alexandra Arias Mazo · Eliel Segura Perez"
)
CLASES = ("planeta", "falso positivo", "sin señal")
COLORES = ("#1b7837", "#b2182b", "#737373")
REDES = {
    "m4_escalares": "Rama 4 · escalares",
    "m1_astronet": "Rama 1 · vistas plegadas",
    "m2_promedio": "Rama 2 · promedio",
    "m2_atencion": "Rama 2 · atención",
    "m2_gru": "Rama 2 · GRU",
}
FUSIONES = {
    "tres_ramas": "Fusión de tres ramas",
    "tres_ramas_ajuste_cruzado": "Fusión con ajuste cruzado",
    "solo_curvas": "Solo curvas (ramas 1 y 2)",
    "solo_vistas": "Solo rama 1",
    "solo_transitos": "Solo rama 2",
    "solo_escalares": "Solo rama 4",
    "tabular_sin_centroide": "Tabular sin centroide",
    "tabular_con_centroide": "Tabular con centroide",
}


def _raiz() -> Path:
    for carpeta in [Path.cwd(), *Path.cwd().parents]:
        if (carpeta / "iteracion_2" / "config" / "protocolo.json").is_file():
            return carpeta
    raise FileNotFoundError("No encuentro la raíz del repositorio (falta iteracion_2/config).")


RAIZ = _raiz()
IT2 = RAIZ / "iteracion_2"
DATOS = IT2 / "datos"
EVIDENCIAS = IT2 / "evidencias"
RESULTADOS = EVIDENCIAS / "resultados"
FIGURAS = EVIDENCIAS / "figuras"
MODELOS = IT2 / "modelos_finales"


def estilo() -> None:
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


def leer_json(ruta: Path | str):
    return json.loads((RAIZ / ruta).read_text(encoding="utf-8"))


def protocolo() -> dict:
    return leer_json(IT2 / "config" / "protocolo.json")


def ejemplos() -> pd.DataFrame:
    return pd.read_csv(DATOS / "ejemplos.csv")


def dataset() -> tuple[dict, str]:
    """El dataset completo si está en disco; si no, la muestra versionada (60 ejemplos al azar).

    El completo pesa cientos de MB y no se versiona: lo generan los pasos 1 y 2 (`.bat`).
    """
    completo, muestra = DATOS / "dataset_temporal.npz", DATOS / "muestra_dataset.npz"
    ruta = completo if completo.is_file() else muestra
    with np.load(ruta, allow_pickle=False) as archivo:
        datos = {clave: archivo[clave] for clave in archivo.files}
    origen = "dataset completo" if ruta == completo else "muestra versionada (semilla fija)"
    return datos, origen


def guardar(figura, nombre: str) -> None:
    """Guarda la figura como evidencia en `iteracion_2/evidencias/figuras/`."""
    FIGURAS.mkdir(parents=True, exist_ok=True)
    figura.savefig(FIGURAS / f"{nombre}.png", dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------------------------
# Búsqueda de hiperparámetros
# ---------------------------------------------------------------------------------------------


def configuraciones(red: str) -> pd.DataFrame:
    return pd.read_csv(RESULTADOS / "busqueda" / red / "configuraciones.csv")


def resumen_busqueda(red: str) -> dict:
    return leer_json(RESULTADOS / "busqueda" / red / "resumen.json")


def sensibilidad(red: str) -> dict:
    return leer_json(RESULTADOS / "busqueda" / red / "sensibilidad_seleccion.json")


def con_intervalo(v: dict, decimales: int = 4, signo: bool = False) -> str:
    f = f"{{:{'+' if signo else ''}.{decimales}f}}"
    return f"{f.format(v['estimacion'])} [{f.format(v['ic_inf'])}; {f.format(v['ic_sup'])}]"


def distinguible(v: dict) -> str:
    """Una diferencia es distinguible de cero si su intervalo no lo contiene."""
    return "sí" if v["ic_inf"] > 0 or v["ic_sup"] < 0 else "no"


def tabla_busqueda() -> pd.DataFrame:
    filas = []
    for red, nombre in REDES.items():
        r = resumen_busqueda(red)
        filas.append(
            {
                "red": nombre,
                "AUC protocolo fijo": con_intervalo(r["auc_prueba"]["fijo"]),
                "AUC con búsqueda": con_intervalo(r["auc_prueba"]["con_busqueda"]),
                "diferencia": con_intervalo(r["auc_prueba"]["diferencia"], signo=True),
                "¿distinguible?": distinguible(r["auc_prueba"]["diferencia"]),
                "elegida por partición": " ".join(r["elegida_por_particion"]),
            }
        )
    return pd.DataFrame(filas).set_index("red")


# ---------------------------------------------------------------------------------------------
# Fusión y comparación
# ---------------------------------------------------------------------------------------------


def comparacion(carpeta: str = "fusion") -> dict:
    return leer_json(RESULTADOS / carpeta / "comparacion_con_tabulares.json")


def predicciones(configuracion: str, carpeta: str = "fusion") -> tuple[pd.DataFrame, np.ndarray]:
    """Tabla de ejemplos alineada y probabilidad media de las cinco semillas de la cabeza."""
    ps, ids = [], None
    for semilla in protocolo()["semillas_cabeza_fusion"]:
        ruta = RESULTADOS / carpeta / configuracion / f"pred_semilla{semilla}.npz"
        with np.load(ruta, allow_pickle=False) as d:
            ids = [str(i) for i in d["id_ejemplo"]]
            ps.append(d["p_prueba"].astype(float))
    tabla = ejemplos().set_index("id_ejemplo").loc[ids].reset_index()
    return tabla, np.mean(ps, axis=0)


def bosque(valores: dict[str, dict], titulo: str, referencia: float | None = None, eje=None):
    """Gráfico de bosque: una fila por modelo, con su estimación y su intervalo del 95 %."""
    eje = eje or plt.subplots(figsize=(7, 0.45 * len(valores) + 1.2))[1]
    nombres = list(valores)[::-1]
    for i, nombre in enumerate(nombres):
        v = valores[nombre]
        eje.plot([v["ic_inf"], v["ic_sup"]], [i, i], color="#2166ac", lw=2)
        eje.plot(v["estimacion"], i, "o", color="#2166ac")
        eje.annotate(
            f"{v['estimacion']:.4f}",
            (v["ic_sup"], i),
            xytext=(6, -3),
            textcoords="offset points",
            fontsize=9,
        )
    if referencia is not None:
        eje.axvline(referencia, color="#737373", ls="--", lw=1)
    eje.set_yticks(range(len(nombres)), [FUSIONES.get(n, REDES.get(n, n)) for n in nombres])
    eje.set_title(titulo, loc="left")
    eje.margins(x=0.12)
    eje.grid(axis="y", visible=False)
    return eje


def confiabilidad(q: np.ndarray, es_planeta: np.ndarray, intervalos: int = 15, eje=None):
    """Diagrama de confiabilidad: probabilidad predicha frente a frecuencia observada."""
    eje = eje or plt.subplots(figsize=(4.2, 4.2))[1]
    bordes = np.linspace(0, 1, intervalos + 1)
    donde = np.clip(np.digitize(q, bordes) - 1, 0, intervalos - 1)
    xs, ys, ece = [], [], 0.0
    for b in range(intervalos):
        en = donde == b
        if en.sum() == 0:
            continue
        xs.append(q[en].mean())
        ys.append(es_planeta[en].mean())
        ece += en.mean() * abs(xs[-1] - ys[-1])
    eje.plot([0, 1], [0, 1], color="#737373", ls="--", lw=1)
    eje.plot(xs, ys, "o-", color="#2166ac")
    eje.set_xlabel("probabilidad predicha de planeta")
    eje.set_ylabel("fracción real de planetas")
    eje.set_title(f"ECE = {ece:.3f}", loc="left")
    return ece


def ver_ejemplo(datos: dict, i: int, titulo: str = ""):
    """Las tres formas en que la red ve una misma señal: vista global, vista local y tránsitos."""
    figura, ejes = plt.subplots(1, 3, figsize=(12, 2.8))
    ejes[0].plot(datos["vista_global"][i, :, 0], lw=0.8, color="#2166ac")
    ejes[0].set_title("Rama 1 · vista global (órbita completa)", loc="left")
    ejes[1].plot(datos["vista_local"][i, :, 0], lw=0.9, color="#2166ac")
    ejes[1].set_title("Rama 1 · vista local (el tránsito)", loc="left")
    presentes = datos["mascara_transitos"][i] > 0.5
    imagen = np.where(presentes[:, None], datos["transitos"][i], np.nan)
    ejes[2].imshow(imagen, aspect="auto", cmap="viridis", interpolation="nearest")
    ejes[2].set_title(f"Rama 2 · {int(presentes.sum())} tránsitos, uno por fila", loc="left")
    ejes[2].grid(False)
    figura.suptitle(titulo, x=0.01, ha="left", fontsize=10)
    figura.tight_layout()
    return figura
