"""Convierte la salida de src/demo/demo_dataset.py al formato del proyecto (id_ejemplo, sin prerregistrados).

Paso 1 (script de la demo, sin modificar):
  python src/demo/demo_dataset.py data/tpf data/seleccion_demo.csv data/demo_crudo.npz
Paso 2:
  python -I src/fusion/dataset_demo.py --crudo data/demo_crudo.npz \
      --ejemplos compartido/ejemplos.csv --salida data/dataset_demo.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from comun.utilidades import escribir_manifiesto

TIPO = {"real": "transito", "nulo": "sin_senal"}


def convertir(crudo: Path, ejemplos: pd.DataFrame) -> dict:
    with np.load(crudo, allow_pickle=False) as d:
        img, seq = d["img"], d["seq"]
    meta = pd.read_csv(crudo.with_name(crudo.stem + "_meta.csv"))
    ids = (meta["koi"] + "|" + meta["tipo"].map(TIPO)).to_numpy()
    validos = set(ejemplos.loc[~ejemplos["prerregistrado"].astype(bool), "id_ejemplo"])
    conservar = np.array([i in validos for i in ids])
    if not (np.isfinite(img[conservar]).all() and np.isfinite(seq[conservar]).all()):
        raise ValueError("El dataset de la demo contiene NaN")
    return {
        "id_ejemplo": np.array(ids[conservar], dtype="U"),
        "img": img[conservar].astype(np.float32),
        "seq": seq[conservar].astype(np.float32),
        "n_transitos_q5": meta["n_transitos"].to_numpy()[conservar],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--crudo", type=Path, required=True)
    ap.add_argument("--ejemplos", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    a = ap.parse_args(argv)
    ejemplos = pd.read_csv(a.ejemplos)
    datos = convertir(a.crudo, ejemplos)
    np.savez_compressed(a.salida, **datos)
    n_total = int((~ejemplos["prerregistrado"].astype(bool)).sum())
    escribir_manifiesto(
        a.salida.with_suffix(".manifiesto.json"),
        vars(a),
        entradas=[a.crudo, a.ejemplos],
        extra={
            "origen": "src/demo/demo_dataset.py sin modificar (commit 6ffa66f)",
            "n_ejemplos": int(len(datos["id_ejemplo"])),
            "n_sin_imagen": n_total - int(len(datos["id_ejemplo"])),
        },
    )
    print(
        f"{len(datos['id_ejemplo'])} ejemplos con imagen de {n_total}; "
        f"img {datos['img'].shape}, seq {datos['seq'].shape}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
