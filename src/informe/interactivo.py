"""Controles interactivos del notebook y explorador de señales.

`controles(funcion, parametro=Lista(...) | Deslizador(...))` conecta una figura con sus controles:
al mover un control, la figura se vuelve a dibujar.

Modo fijo. Si `ipywidgets` no está instalado, o si la variable de entorno INFORME_ESTATICO vale
"1" (así se genera la copia del repositorio, que debe poder leerse sin ejecutar nada), la figura
se dibuja una vez con los valores iniciales.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from comun import metricas
from comun.metricas import entropia
from informe.datos import CLASES, Resultados
from informe.graficos import COLOR, COLOR_DE_CLASE, ver_ejemplo

VARIABLE_MODO_FIJO = "INFORME_ESTATICO"
N_MAS_DUDOSOS = 40


@dataclass(frozen=True)
class Lista:
    """Control desplegable. `opciones`: valores, o pares (texto, valor)."""

    etiqueta: str
    opciones: Sequence
    inicial: object = None

    def valor_inicial(self):
        if self.inicial is not None:
            return self.inicial
        primera = self.opciones[0]
        return primera[1] if isinstance(primera, tuple) else primera


@dataclass(frozen=True)
class Deslizador:
    """Control deslizante numérico."""

    etiqueta: str
    minimo: float
    maximo: float
    paso: float
    inicial: float

    def valor_inicial(self):
        return self.inicial


def hay_controles() -> bool:
    """¿Se pueden mostrar controles? (ipywidgets instalado y no se pidió el modo fijo)."""
    if os.environ.get(VARIABLE_MODO_FIJO) == "1":
        return False
    try:
        import ipywidgets  # noqa: F401
    except ImportError:
        return False
    return True


def _crear_control(control: Lista | Deslizador):
    import ipywidgets as widgets

    estilo = {"description_width": "initial"}
    if isinstance(control, Lista):
        return widgets.Dropdown(
            options=list(control.opciones),
            value=control.valor_inicial(),
            description=control.etiqueta,
            style=estilo,
            layout=widgets.Layout(width="auto"),
        )
    entero = all(float(v).is_integer() for v in (control.minimo, control.maximo, control.paso))
    tipo = widgets.IntSlider if entero else widgets.FloatSlider
    return tipo(
        min=control.minimo,
        max=control.maximo,
        step=control.paso,
        value=control.inicial,
        description=control.etiqueta,
        style=estilo,
        continuous_update=False,  # redibuja al soltar, no en cada paso
        layout=widgets.Layout(width="420px"),
    )


def _mostrar(resultado) -> None:
    """Las figuras devuelven a veces una tabla: se muestra debajo."""
    if resultado is not None:
        display(resultado)


def controles(funcion: Callable, **parametros: Lista | Deslizador) -> None:
    """Dibuja `funcion(**valores)` y la redibuja cada vez que cambia un control."""
    if not hay_controles():
        iniciales = {nombre: control.valor_inicial() for nombre, control in parametros.items()}
        _mostrar(funcion(**iniciales))
        elegidos = ", ".join(f"{c.etiqueta} {c.valor_inicial()}" for c in parametros.values())
        print(f"(Vista fija con {elegidos}. Ejecuta el notebook para usar los controles.)")
        return

    import ipywidgets as widgets

    creados = {nombre: _crear_control(control) for nombre, control in parametros.items()}
    salida = widgets.interactive_output(lambda **valores: _mostrar(funcion(**valores)), creados)
    display(widgets.HBox(list(creados.values())), salida)


# ---------------------------------------------------------------------------------------------
# Explorador de señales (sección 7)
# ---------------------------------------------------------------------------------------------


class Explorador:
    """Reúne lo que opinó cada rama y el sistema completo sobre cada señal."""

    def __init__(self, res: Resultados):
        self.res = res
        calibracion = res.json("calibracion", "metricas.json")["E9_prior"]
        self.tasa_entrenamiento = calibracion["tasa_entrenamiento_media"]
        self.tasa_dr25 = calibracion["tasa_dr25"]["valor"]
        self.por_rama = {
            "Rama 1 · vistas": res.predicciones("m1_astronet"),
            "Rama 2 · tránsitos (atención)": res.predicciones("m2_atencion"),
            "Rama 3 · imagen v0 (demo)": res.predicciones("imagen_v0"),
            "Rama 4 · escalares": res.predicciones("m4_escalares"),
            "FUSIÓN (calibrada)": res.npz("calibracion", "pred_calibrada.npz")["p_despues"],
        }
        incertidumbre = res.npz("incertidumbre", "predicciones.npz")
        self.p_ensamble = incertidumbre["p_ensamble"]
        self.desacuerdo = incertidumbre["im_ensamble"]
        self.entropia = entropia(self.p_ensamble)
        self.es_error = self.p_ensamble.argmax(axis=1) != res.y

    def grupos(self) -> dict[str, list[str]]:
        """Listas de señales para el primer desplegable."""
        ids, y = self.res.ids, self.res.y
        mas_dudosos = np.argsort(-self.entropia)[:N_MAS_DUDOSOS]
        return {
            "Errores de la fusión": sorted(ids[self.es_error]),
            "Planetas": sorted(ids[y == 0]),
            "Falsos positivos": sorted(ids[y == 1]),
            "Sin señal": sorted(ids[y == 2]),
            "Más dudosos (entropía alta)": list(ids[mas_dudosos]),
        }

    def _opinion_de_cada_rama(self, ax, i: int) -> None:
        """Barras apiladas: la probabilidad que cada rama dio a cada clase."""
        nombres = list(self.por_rama)
        p = np.array([self.por_rama[nombre][i] for nombre in nombres], dtype=float)
        izquierda = np.zeros(len(nombres))
        for clase, color in enumerate(COLOR_DE_CLASE):
            ancho = np.nan_to_num(p[:, clase])
            ax.barh(nombres, ancho, left=izquierda, color=color, label=CLASES[clase])
            izquierda += ancho
        for fila in np.flatnonzero(np.isnan(p).any(axis=1)):
            ax.text(0.5, fila, "sin dato para este ejemplo", ha="center", va="center", fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.legend(ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0))
        ax.grid(False)

    def _efecto_de_la_tasa_base(self, ax, i: int, tasa: float) -> None:
        """P(planeta | hay señal) de la fusión, antes y después de corregir el prior."""
        p_fusion = np.asarray(self.por_rama["FUSIÓN (calibrada)"][i : i + 1], dtype=float)
        q = metricas.q_planeta(p_fusion)
        q_corregida = metricas.corregir_prior(q, self.tasa_entrenamiento, tasa)
        valores = [float(q[0]), float(q_corregida[0])]
        ax.bar(
            ["sin corregir", f"tasa base {tasa:g}"], valores, color=[COLOR["base"], COLOR["curso"]]
        )
        for j, valor in enumerate(valores):
            ax.text(j, valor + 0.02, f"{valor:.3f}", ha="center")
        ax.set_ylim(0, 1.05)
        ax.set_title("P(planeta | hay señal)")

    def _ficha(self, i: int) -> None:
        """Dos líneas de texto: qué pasó con la señal y sus datos básicos."""
        fila = self.res.tabla.iloc[i]
        escalares = self.res.dataset["escalares"][i]
        predicha = CLASES[int(self.p_ensamble[i].argmax())]
        marca = "✗ ERROR" if self.es_error[i] else "✓"
        print(
            f"Real: {CLASES[fila.clase]} · predicho por la fusión: {predicha} {marca} · "
            f"entropía {self.entropia[i]:.3f} · desacuerdo entre cabezas {self.desacuerdo[i]:.4f}"
        )
        print(
            f"Periodo {fila.periodo_d:.3f} d · duración {fila.duracion_h:.2f} h · "
            f"profundidad medida {escalares[2]:.0f} ppm · SNR medido {escalares[3]:.1f} · "
            f"MES {fila.mes:.1f} · banderas del catálogo (solo informativas, nunca entradas): "
            f"nt={fila.fp_nt} ss={fila.fp_ss} co={fila.fp_co} ec={fila.fp_ec}"
        )

    def ver(self, id_ejemplo: str, tasa: float = 0.5) -> None:
        """Ficha completa de una señal: sus datos, la opinión de cada rama y la decisión."""
        i = self.res.posicion(id_ejemplo)
        ver_ejemplo(self.res, id_ejemplo)
        fig, (ax_ramas, ax_tasa) = plt.subplots(
            1, 2, figsize=(13, 2.9), gridspec_kw={"width_ratios": [2.2, 1]}
        )
        self._opinion_de_cada_rama(ax_ramas, i)
        self._efecto_de_la_tasa_base(ax_tasa, i, tasa)
        plt.tight_layout()
        plt.show()
        self._ficha(i)

    def mostrar(self, caso_fijo: str, tasa_fija: float = 0.05) -> None:
        """Explorador con tres controles: grupo, señal y tasa base declarada."""
        if not hay_controles():
            self.ver(caso_fijo, tasa_fija)
            print("(Vista fija. Ejecuta el notebook para elegir cualquier señal.)")
            return

        import ipywidgets as widgets

        grupos = self.grupos()
        tasas = sorted({0.02, 0.05, 0.1, 0.2, 0.5, round(self.tasa_dr25, 3)})
        elegir_grupo = widgets.Dropdown(options=list(grupos), description="Grupo:")
        elegir_senal = widgets.Dropdown(options=grupos[elegir_grupo.value], description="Señal:")
        elegir_tasa = widgets.SelectionSlider(options=tasas, value=0.5, description="Tasa base:")

        def al_cambiar_de_grupo(_cambio) -> None:
            elegir_senal.options = grupos[elegir_grupo.value]

        elegir_grupo.observe(al_cambiar_de_grupo, names="value")
        salida = widgets.interactive_output(
            self.ver, {"id_ejemplo": elegir_senal, "tasa": elegir_tasa}
        )
        display(widgets.HBox([elegir_grupo, elegir_senal, elegir_tasa]), salida)
