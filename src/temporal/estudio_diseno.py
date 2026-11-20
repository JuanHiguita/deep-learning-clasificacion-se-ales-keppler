"""Estudio de decisiones de diseño de las ramas 1, 2 y 4 (spec 010). Reanudable.

Cada variante cambia una sola cosa respecto al protocolo: activación, normalización, optimizador,
profundidad, función de pérdida o preprocesamiento. Incluye los experimentos pendientes del plan
E5 (pooling de la rama 1) y E2b (rama 2 sin los estadísticos de consistencia).

Uso:
  python -I src/temporal/estudio_diseno.py --dataset data/dataset_temporal.npz \
      --ejemplos compartido/ejemplos.csv --salida resultados/dl/diseno --referencia resultados/dl \
      --arquitecturas m2_atencion [--presupuesto 150]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from comun import entrenamiento as ent  # noqa: E402
from comun import estudio  # noqa: E402
from comun.protocolo import cargar_protocolo  # noqa: E402
from temporal import modelos  # noqa: E402
from temporal.exp_ramas import EXPERIMENTOS  # noqa: E402

_CONSTRUCTOR = {
    "m4_escalares": modelos.m4_escalares,
    "m1_astronet": modelos.m1_astronet,
    "m2_promedio": partial(modelos.m2, agregador="promedio"),
    "m2_gru": partial(modelos.m2, agregador="gru"),
    "m2_atencion": partial(modelos.m2, agregador="atencion"),
}
CLAVES_DE_SERIES = ("vista_global", "vista_local", "transitos")


@dataclass(frozen=True)
class Variante:
    """Qué cambia una variante respecto al protocolo."""

    factor: str
    cambio: str
    modelo: dict = field(default_factory=dict)  # argumentos del constructor del modelo
    entrenamiento: dict = field(default_factory=dict)  # claves de protocolo["entrenamiento"]
    aumentacion: bool = True
    estandarizar: bool = True
    series_en_zscore: bool = False
    sin_entradas: tuple[str, ...] = ()  # entradas del protocolo que esta variante no recibe
    guardar_modelos: bool = False


_COMUNES = {
    "act_gelu": Variante("activación", "GELU en vez de ReLU", modelo={"activacion": "gelu"}),
    "act_tanh": Variante("activación", "tanh en vez de ReLU", modelo={"activacion": "tanh"}),
    "norm_lote": Variante(
        "normalización", "normalización por lote (BatchNorm)", modelo={"normalizacion": "lote"}
    ),
    "norm_capa": Variante(
        "normalización", "normalización por capa (LayerNorm)", modelo={"normalizacion": "capa"}
    ),
    "opt_adam": Variante(
        "optimizador", "Adam (sin decaimiento de pesos)", entrenamiento={"optimizador": "adam"}
    ),
    "opt_sgd": Variante(
        "optimizador", "SGD con momento de Nesterov, tasa ×10", entrenamiento={"optimizador": "sgd"}
    ),
    "capas_menos": Variante("profundidad", "una capa o bloque menos", modelo={"capas_extra": -1}),
    "capas_mas": Variante("profundidad", "una capa o bloque más", modelo={"capas_extra": 1}),
    "perdida_sin_pesos": Variante(
        "pérdida", "entropía cruzada sin pesos por clase", entrenamiento={"pesos_de_clase": False}
    ),
    "perdida_suavizado": Variante(
        "pérdida", "suavizado de etiquetas 0,1", entrenamiento={"perdida": "suavizado"}
    ),
    "perdida_focal": Variante(
        "pérdida", "pérdida focal (γ = 2)", entrenamiento={"perdida": "focal"}
    ),
}
_DE_SERIES = {
    "pre_sin_aumentacion": Variante(
        "preprocesamiento", "sin aumentación de datos", aumentacion=False
    ),
    "pre_zscore": Variante(
        "preprocesamiento",
        "cada serie con media 0 y desviación 1 (se pierde la escala de profundidad)",
        series_en_zscore=True,
    ),
}
_REFERENCIA = Variante("referencia", "protocolo fijo (experimento publicado)")

VARIANTES = {
    "m4_escalares": {
        estudio.REFERENCIA: _REFERENCIA,
        **_COMUNES,
        "pre_sin_estandarizar": Variante(
            "preprocesamiento", "escalares sin estandarizar", estandarizar=False
        ),
    },
    "m1_astronet": {
        estudio.REFERENCIA: _REFERENCIA,
        **_COMUNES,
        **_DE_SERIES,
        # E5 del plan: cómo reduce la torre convolucional
        "pool_promedio": Variante(
            "pooling (E5)", "pooling promedio", modelo={"pooling": "promedio"}
        ),
        "pool_paso": Variante(
            "pooling (E5)", "sin pooling: convolución con paso 2", modelo={"pooling": "paso"}
        ),
    },
    "m2_atencion": {
        estudio.REFERENCIA: _REFERENCIA,
        **_COMUNES,
        **_DE_SERIES,
        # E2b del plan; se guardan los modelos porque E4 (menos tránsitos) los necesita
        "sin_consistencia": Variante(
            "entradas (E2b)",
            "sin los 4 estadísticos de consistencia",
            modelo={"usar_consistencia": False},
            sin_entradas=("consistencia",),
            guardar_modelos=True,
        ),
    },
    **{
        arquitectura: {
            estudio.REFERENCIA: _REFERENCIA,
            "sin_consistencia": Variante(
                "entradas (E2b)",
                "sin los 4 estadísticos de consistencia",
                modelo={"usar_consistencia": False},
                guardar_modelos=True,
            ),
        }
        for arquitectura in ("m2_promedio", "m2_gru")
    },
}


def descripciones(arquitectura: str) -> dict[str, tuple[str, str]]:
    return {nombre: (v.factor, v.cambio) for nombre, v in VARIANTES[arquitectura].items()}


def series_en_zscore(dataset: dict) -> dict:
    """Copia del dataset con cada serie estandarizada por separado (media 0, desviación 1).

    Es la alternativa a la normalización por profundidad del proyecto (DD-04): la red deja de
    saber cuán profundo es el tránsito respecto al ruido.
    """
    copia = dict(dataset)
    for clave in CLAVES_DE_SERIES:
        x = dataset[clave].astype(np.float32)
        eje = 1 if clave != "transitos" else 2  # eje del tiempo
        media = x.mean(axis=eje, keepdims=True)
        desviacion = x.std(axis=eje, keepdims=True)
        copia[clave] = np.where(desviacion > 1e-6, (x - media) / np.maximum(desviacion, 1e-6), 0.0)
    return copia


def experimento_de(arquitectura: str, nombre: str, variante: Variante) -> ent.Experimento:
    base = EXPERIMENTOS[arquitectura][0]
    estandarizar = base.estandarizar if variante.estandarizar else ()
    return ent.Experimento(
        nombre,
        partial(_CONSTRUCTOR[arquitectura], **variante.modelo),
        tuple(c for c in base.claves_entrada if c not in variante.sin_entradas),
        base.clave_disponible,
        base.aumentar if variante.aumentacion else None,
        tuple(c for c in estandarizar if c not in variante.sin_entradas),
        base.clave_faltantes,
    )


def _copiar_referencia(origen: Path, destino: Path) -> None:
    """La referencia es el experimento publicado (semilla 0): se copia en vez de reentrenarla."""
    if (destino / "metricas.json").is_file():
        return
    destino.mkdir(parents=True, exist_ok=True)
    for archivo in ("pred_semilla0.npz", "historial_semilla0.json", "metricas.json"):
        shutil.copyfile(origen / archivo, destino / archivo)


def ejecutar(
    arquitectura: str,
    dataset: dict,
    ejemplos: pd.DataFrame,
    protocolo: dict,
    salida: Path,
    referencia: Path,
    presupuesto_s: float | None = None,
) -> pd.DataFrame:
    """Entrena las variantes que falten (semilla 0) y devuelve la tabla comparativa."""
    carpeta = Path(salida) / arquitectura
    carpeta.mkdir(parents=True, exist_ok=True)
    _copiar_referencia(Path(referencia) / arquitectura, carpeta / estudio.REFERENCIA)
    inicio = time.perf_counter()
    en_zscore = None
    for nombre, variante in VARIANTES[arquitectura].items():
        if (carpeta / nombre / "metricas.json").is_file():
            continue
        restante = None if presupuesto_s is None else presupuesto_s - (time.perf_counter() - inicio)
        if restante is not None and restante <= 0:
            raise ent.Incompleto(f"{arquitectura}: pausa antes de {nombre}")
        entradas = dataset
        if variante.series_en_zscore:
            en_zscore = en_zscore or series_en_zscore(dataset)
            entradas = en_zscore
        con_cambios = json.loads(json.dumps(protocolo))
        con_cambios["entrenamiento"].update(variante.entrenamiento)
        ent.entrenar_cv(
            experimento_de(arquitectura, nombre, variante),
            entradas,
            ejemplos,
            con_cambios,
            [0],
            carpeta,
            con_intervalos=False,
            presupuesto_s=restante,
            guardar_modelos=variante.guardar_modelos,
        )
    return estudio.resumir(carpeta, ejemplos, protocolo, descripciones(arquitectura))


def confirmar(
    arquitectura: str,
    variantes: list[str],
    semillas: list[int],
    dataset: dict,
    ejemplos: pd.DataFrame,
    protocolo: dict,
    salida: Path,
    referencia: Path,
    presupuesto_s: float | None = None,
) -> pd.DataFrame:
    """Entrena las variantes indicadas con más semillas y las compara con el protocolo.

    La semilla 0 ya está entrenada: `entrenar_cv` la reutiliza y solo entrena las que faltan.
    """
    carpeta = Path(salida) / arquitectura
    en_zscore = None
    inicio = time.perf_counter()
    for nombre in variantes:
        variante = VARIANTES[arquitectura][nombre]
        if all((carpeta / nombre / f"pred_semilla{s}.npz").is_file() for s in semillas):
            continue
        entradas = dataset
        if variante.series_en_zscore:
            en_zscore = en_zscore or series_en_zscore(dataset)
            entradas = en_zscore
        con_cambios = json.loads(json.dumps(protocolo))
        con_cambios["entrenamiento"].update(variante.entrenamiento)
        restante = None if presupuesto_s is None else presupuesto_s - (time.perf_counter() - inicio)
        ent.entrenar_cv(
            experimento_de(arquitectura, nombre, variante),
            entradas,
            ejemplos,
            con_cambios,
            semillas,
            carpeta,
            con_intervalos=False,
            presupuesto_s=restante,
        )
    return estudio.confirmar(
        carpeta, Path(referencia) / arquitectura, variantes, semillas, ejemplos, protocolo
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--ejemplos", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--referencia", type=Path, required=True, help="resultados del protocolo fijo")
    ap.add_argument("--arquitecturas", nargs="+", choices=list(VARIANTES), required=True)
    ap.add_argument("--presupuesto", type=float, default=None, help="segundos; pausa y reanuda")
    ap.add_argument("--protocolo", type=Path, default=Path("config/protocolo.json"))
    ap.add_argument(
        "--confirmar", nargs="+", default=None, help="variantes a repetir con 3 semillas"
    )
    a = ap.parse_args(argv)

    protocolo = cargar_protocolo(a.protocolo)
    with np.load(a.dataset, allow_pickle=False) as archivo:
        dataset = {clave: archivo[clave] for clave in archivo.files}
    ejemplos = pd.read_csv(a.ejemplos)
    if a.confirmar:
        try:
            tabla = confirmar(
                a.arquitecturas[0],
                a.confirmar,
                protocolo["semillas"],
                dataset,
                ejemplos,
                protocolo,
                a.salida,
                a.referencia,
                a.presupuesto,
            )
        except ent.Incompleto as pausa:
            print("INCOMPLETO:", pausa)
            return 3
        print(tabla.to_string(index=False))
        return 0
    inicio = time.perf_counter()
    for arquitectura in a.arquitecturas:
        restante = None if a.presupuesto is None else a.presupuesto - (time.perf_counter() - inicio)
        try:
            tabla = ejecutar(
                arquitectura, dataset, ejemplos, protocolo, a.salida, a.referencia, restante
            )
        except ent.Incompleto as pausa:
            print("INCOMPLETO:", pausa)
            return 3
        columnas = [
            "variante",
            "nll_validacion",
            "auc_prueba",
            "diferencia_auc",
            "ic_inf",
            "ic_sup",
        ]
        print(arquitectura, "terminado\n", tabla[columnas].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
