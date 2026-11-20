"""Volver a usar un modelo guardado: cargarlo y predecir con él (spec 009, RF de despliegue).

Un modelo guardado en `modelos/mejores/<arquitectura>/` consta de:
    particion_<k>.keras                  modelo completo (entradas → 3 probabilidades)
    estandarizacion_particion_<k>.npz    media y desviación con que se estandarizaron sus entradas
    manifiesto.json                      hiperparámetros, configuración y huella SHA-256

El modelo de la partición k nunca vio las estrellas de la partición k: sobre ellas su predicción
es una predicción honesta, y debe coincidir con la que se reportó.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras  # noqa: E402
import numpy as np  # noqa: E402

from comun import entrenamiento as ent  # noqa: E402
from comun.utilidades import sha256_archivo  # noqa: E402
from temporal import modelos  # noqa: E402, F401  (registra las capas propias al importarse)
from temporal.exp_ramas import EXPERIMENTOS  # noqa: E402


@dataclass(frozen=True)
class ModeloGuardado:
    """Un modelo listo para predecir, con la estandarización que necesita."""

    arquitectura: str
    particion: int
    modelo: keras.Model
    estandarizacion: dict[str, tuple[np.ndarray, np.ndarray]]  # clave → (media, desviación)
    hiperparametros: dict

    def predecir(self, entradas: dict[str, np.ndarray]) -> np.ndarray:
        """Probabilidades (n, 3) para entradas con el formato de `dataset_temporal.npz`."""
        experimento = EXPERIMENTOS[self.arquitectura][0]
        faltantes = entradas[experimento.clave_faltantes] if experimento.clave_faltantes else None
        lote = {}
        for clave in experimento.claves_entrada:
            if clave in self.estandarizacion:
                media, desviacion = self.estandarizacion[clave]
                lote[clave] = ent.aplicar_estandarizacion(
                    entradas[clave], media, desviacion, faltantes
                )
            else:
                lote[clave] = entradas[clave]
        return np.asarray(self.modelo.predict(lote, verbose=0, batch_size=256))


def cargar(carpeta: str | Path, arquitectura: str, particion: int) -> ModeloGuardado:
    """Carga un modelo guardado y comprueba su huella contra el manifiesto."""
    carpeta = Path(carpeta) / arquitectura
    manifiesto = json.loads((carpeta / "manifiesto.json").read_text(encoding="utf-8"))
    datos = manifiesto["particiones"][str(particion)]
    ruta = carpeta / datos["archivo"]
    if sha256_archivo(ruta) != datos["sha256"]:
        raise ValueError(f"{ruta} no coincide con la huella del manifiesto: no se carga")
    modelo = keras.saving.load_model(ruta, safe_mode=True)

    estandarizacion = {}
    ruta_estandarizacion = carpeta / f"estandarizacion_particion_{particion}.npz"
    if ruta_estandarizacion.is_file():
        with np.load(ruta_estandarizacion, allow_pickle=False) as archivo:
            claves = {
                nombre.removesuffix("_media")
                for nombre in archivo.files
                if nombre.endswith("_media")
            }
            for clave in claves:
                estandarizacion[clave] = (archivo[f"{clave}_media"], archivo[f"{clave}_desviacion"])
    return ModeloGuardado(
        arquitectura, particion, modelo, estandarizacion, datos["hiperparametros"]
    )
