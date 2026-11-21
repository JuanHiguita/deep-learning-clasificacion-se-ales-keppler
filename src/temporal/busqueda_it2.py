"""Iteración 2 (spec 014): búsqueda ampliada de hiperparámetros con más datos y en GPU. Reanudable.

Qué cambia frente a la búsqueda de la spec 009 (`busqueda_ramas.py`), y qué no:

· Espacio más amplio: redes hasta 4 veces más anchas y con 1 o 2 bloques más, activación,
  normalización y pérdida (entropía cruzada o con suavizado de etiquetas). Con 4.547 señales hay
  datos para redes mayores; con 262 no los había.
· Más configuraciones por arquitectura (12 en vez de 8), las mismas para todas.
· El modelo de cada partición se guarda mientras se busca: en GPU reentrenar no da los mismos pesos.
· Las reglas NO cambian: el espacio se declara aquí antes de entrenar, la configuración 0 es el
  protocolo fijo, la selección es anidada (cada partición elige con SU validación interna) y la
  prueba nunca decide.

Uso (desde la raíz del repositorio, con el entorno de GPU):
  python -I src/temporal/busqueda_it2.py --arquitecturas m4_escalares m1_astronet \
      [--dataset iteracion_2/datos/dataset_temporal.npz] [--n 12]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MOSTRAR_PROGRESO", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from comun.protocolo import cargar_protocolo  # noqa: E402
from temporal import busqueda_ramas, modelos  # noqa: E402

N_CONFIGURACIONES = 12  # la del protocolo y 11 sorteadas, igual para todas las arquitecturas
RAIZ = Path("iteracion_2")

_ENTRENAMIENTO = {
    "tasa_aprendizaje": ("log", 3e-4, 5e-3),
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    "lote": [128, 256],
    "perdida": ["ce", "suavizado"],
}
_FIJO = {"tasa_aprendizaje": 1e-3, "weight_decay": 1e-4, "lote": 128, "perdida": "ce"}
_RED = {"activacion": ["relu", "tanh"], "capas_extra": [0, 1, 2]}
_RED_FIJA = {"activacion": "relu", "capas_extra": 0}
# Rama 2: sin normalización por lote, porque los tránsitos vacíos (ceros) entrarían en sus medias.
_RAMA_2 = {
    **_ENTRENAMIENTO,
    **_RED,
    "dropout": [0.1, 0.3, 0.5],
    "ancho": [1.0, 2.0, 4.0],
    "normalizacion": [None, "capa"],
}
_RAMA_2_FIJA = {**_FIJO, **_RED_FIJA, "dropout": 0.3, "ancho": 1.0, "normalizacion": None}

# arquitectura → (espacio, configuración fija del protocolo). Declarado antes de entrenar.
ESPACIOS = {
    "m4_escalares": (
        {
            **_ENTRENAMIENTO,
            **_RED,
            "dropout": [0.1, 0.2, 0.4],
            "oculta": [32, 64, 128, 256],
            "normalizacion": [None, "lote", "capa"],
        },
        {**_FIJO, **_RED_FIJA, "dropout": 0.2, "oculta": 32, "normalizacion": None},
    ),
    "m1_astronet": (
        {
            **_ENTRENAMIENTO,
            **_RED,
            "dropout": [0.1, 0.3, 0.5],
            "ancho": [1.0, 2.0, 4.0],
            "pooling": ["max", "promedio"],
            "normalizacion": [None, "lote"],
        },
        {
            **_FIJO,
            **_RED_FIJA,
            "dropout": 0.3,
            "ancho": 1.0,
            "pooling": "max",
            "normalizacion": None,
        },
    ),
    "m2_promedio": (_RAMA_2, _RAMA_2_FIJA),
    "m2_gru": (
        {**_RAMA_2, "unidades_gru": [32, 64, 128]},
        {**_RAMA_2_FIJA, "unidades_gru": 32},
    ),
    "m2_atencion": (
        {**_RAMA_2, "cabezas": [2, 4, 8]},
        {**_RAMA_2_FIJA, "cabezas": 2},
    ),
}
# La rama 2 aplica φ con los tránsitos plegados en el lote: misma función, mucho más rápida en GPU.
CONSTRUCTORES = {
    "m4_escalares": modelos.m4_escalares,
    "m1_astronet": modelos.m1_astronet,
    "m2_promedio": partial(modelos.m2, agregador="promedio", plegado=True),
    "m2_gru": partial(modelos.m2, agregador="gru", plegado=True),
    "m2_atencion": partial(modelos.m2, agregador="atencion", plegado=True),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arquitecturas", nargs="+", choices=list(ESPACIOS), required=True)
    ap.add_argument("--dataset", type=Path, default=RAIZ / "datos" / "dataset_temporal.npz")
    ap.add_argument("--ejemplos", type=Path, default=RAIZ / "datos" / "ejemplos.csv")
    ap.add_argument("--protocolo", type=Path, default=RAIZ / "config" / "protocolo.json")
    ap.add_argument("--salida", type=Path, default=RAIZ / "evidencias" / "resultados" / "busqueda")
    ap.add_argument(
        "--referencia",
        type=Path,
        default=RAIZ / "evidencias" / "resultados",
        help="resultados del protocolo fijo: su corrida es la configuración 0",
    )
    ap.add_argument("--n", type=int, default=N_CONFIGURACIONES)
    a = ap.parse_args(argv)

    protocolo = cargar_protocolo(a.protocolo)
    with np.load(a.dataset, allow_pickle=False) as archivo:
        dataset = {clave: archivo[clave] for clave in archivo.files}
    ejemplos = pd.read_csv(a.ejemplos)
    for arquitectura in a.arquitecturas:
        inicio = time.perf_counter()
        print(f"=== {arquitectura}: {a.n} configuraciones ===", flush=True)
        resumen = busqueda_ramas.ejecutar(
            arquitectura,
            dataset,
            ejemplos,
            protocolo,
            a.salida,
            a.n,
            referencia=a.referencia,
            espacios=ESPACIOS,
            constructores=CONSTRUCTORES,
            guardar_modelos=True,
        )
        minutos = (time.perf_counter() - inicio) / 60
        print(
            arquitectura,
            json.dumps(resumen["auc_prueba"]),
            resumen["elegida_por_particion"],
            f"({minutos:.0f} min)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
