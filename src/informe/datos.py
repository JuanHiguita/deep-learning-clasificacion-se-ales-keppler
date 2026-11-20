"""Carga única de todo lo que el notebook necesita leer.

El notebook crea un objeto `Resultados` con `cargar()` y se lo pasa a cada figura. Así ninguna
celda repite rutas ni lógica de lectura.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CLASES = ("Planeta", "Falso positivo", "Sin señal")
N_CLASES = len(CLASES)

# Nombre legible de cada experimento de rama (carpeta en resultados/dl/)
NOMBRE_DE_RAMA = {
    "m4_escalares": "Rama 4 · escalares",
    "m1_astronet": "Rama 1 · CNN tipo AstroNet (vista plegada)",
    "m2_promedio": "Rama 2 · tránsitos + promedio",
    "m2_gru": "Rama 2 · tránsitos + GRU",
    "m2_atencion": "Rama 2 · tránsitos + atención",
}
NOMBRE_DE_METRICA = {
    "auc_planeta_fp": "AUC planeta vs FP (mayor es mejor)",
    "nll": "NLL: log-verosimilitud negativa (menor es mejor)",
    "exactitud": "Exactitud en 3 clases (mayor es mejor)",
    "ece_top": "ECE: error de calibración (menor es mejor)",
}


def buscar_raiz(desde: Path | None = None) -> Path:
    """Sube desde `desde` hasta la carpeta del repositorio (la que tiene config/protocolo.json)."""
    inicio = (desde or Path.cwd()).resolve()
    for carpeta in (inicio, *inicio.parents):
        if (carpeta / "config" / "protocolo.json").is_file():
            return carpeta
    raise FileNotFoundError("No se encontró la raíz del repositorio (config/protocolo.json)")


@dataclass(frozen=True)
class Resultados:
    """Datos y resultados del proyecto, alineados por `id_ejemplo`.

    dataset   arreglos de data/dataset_temporal.npz (vistas, tránsitos, escalares…)
    ids       identificador de cada ejemplo, en el orden del dataset
    tabla     fila de compartido/ejemplos.csv de cada ejemplo, en ese mismo orden
    y         clase real: 0 planeta, 1 falso positivo, 2 sin señal
    """

    raiz: Path
    dataset: dict
    ids: np.ndarray
    tabla: pd.DataFrame
    y: np.ndarray

    @property
    def carpeta(self) -> Path:
        return self.raiz / "resultados" / "dl"

    def posicion(self, id_ejemplo: str) -> int:
        """Índice de un ejemplo dentro del dataset."""
        coincidencias = np.flatnonzero(self.ids == id_ejemplo)
        if len(coincidencias) == 0:
            raise KeyError(f"No existe el ejemplo {id_ejemplo!r}")
        return int(coincidencias[0])

    def json(self, *partes: str):
        """Lee resultados/dl/<partes…> como JSON."""
        return json.loads(self.carpeta.joinpath(*partes).read_text(encoding="utf-8"))

    def csv(self, nombre: str) -> pd.DataFrame:
        """Lee resultados/dl/tablas/<nombre>.csv."""
        return pd.read_csv(self.carpeta / "tablas" / f"{nombre}.csv")

    def npz(self, *partes: str) -> dict:
        """Lee resultados/dl/<partes…> como diccionario de arreglos."""
        with np.load(self.carpeta.joinpath(*partes), allow_pickle=False) as archivo:
            return {clave: archivo[clave] for clave in archivo.files}

    def predicciones(self, experimento: str, semilla: int = 0) -> np.ndarray:
        """Probabilidades fuera de muestra de un experimento, alineadas con `ids`.

        Devuelve (n, 3). Queda NaN en los ejemplos que ese experimento no tiene (por ejemplo,
        una estrella sin imagen en la rama 3).
        """
        archivo = self.npz(experimento, f"pred_semilla{semilla}.npz")
        posicion = {ident: i for i, ident in enumerate(self.ids)}
        p = np.full((len(self.ids), N_CLASES), np.nan)
        filas = [posicion[ident] for ident in archivo["id_ejemplo"].astype(str)]
        p[filas] = archivo["p_prueba"]
        return p


def cargar(raiz: Path | None = None) -> Resultados:
    """Lee el dataset y la tabla de ejemplos. Las tablas de resultados se leen al pedirlas."""
    raiz = raiz or buscar_raiz()
    with np.load(raiz / "data" / "dataset_temporal.npz", allow_pickle=False) as archivo:
        dataset = {clave: archivo[clave] for clave in archivo.files}
    ids = dataset["id_ejemplo"].astype(str)
    ejemplos = pd.read_csv(raiz / "compartido" / "ejemplos.csv")
    tabla = ejemplos.set_index("id_ejemplo").loc[ids].reset_index()
    return Resultados(raiz, dataset, ids, tabla, tabla["clase"].to_numpy())
