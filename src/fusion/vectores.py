"""Carga de los vectores congelados de cada rama, unidos por id_ejemplo (contrato §5 y §6)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

RAMAS = ("vistas", "transitos", "imagen", "escalares")
DISPONIBLE = {"vistas": "disponible_vistas", "transitos": "disponible_transitos"}


def cargar_vectores(modelos: Path, versiones: dict[str, str], k: int, ids: np.ndarray, dataset: dict
                    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Vectores de la partición k alineados con `ids` (ceros donde la rama no está) y banderas (n, 4)."""
    ids = np.array([str(i) for i in ids])
    pos_ds = {str(i): j for j, i in enumerate(dataset["id_ejemplo"])}
    vectores, banderas = {}, np.zeros((len(ids), len(RAMAS)), np.float32)
    for r, rama in enumerate(RAMAS):
        with np.load(Path(modelos) / rama / versiones[rama] / f"vectores_particion_{k}.npz", allow_pickle=False) as d:
            pos = {str(i): j for j, i in enumerate(d["id_ejemplo"])}
            z = d["z"]
        v = np.zeros((len(ids), z.shape[1]), np.float32)
        for i, ident in enumerate(ids):
            if ident not in pos:
                continue
            if rama in DISPONIBLE and not dataset[DISPONIBLE[rama]][pos_ds[ident]]:
                continue
            v[i], banderas[i, r] = z[pos[ident]], 1.0
        vectores[rama] = v
    return vectores, banderas


def armar_entrada(vectores: dict[str, np.ndarray], banderas: np.ndarray, ramas: tuple[str, ...]) -> np.ndarray:
    """Concatena los vectores de las ramas pedidas y sus banderas de disponibilidad."""
    cols = [RAMAS.index(r) for r in ramas]
    return np.concatenate([vectores[r] for r in ramas] + [banderas[:, cols]], axis=1).astype(np.float32)
