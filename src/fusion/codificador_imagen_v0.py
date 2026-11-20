"""Rama de imagen v0 (código de la demo, congelada) y línea base M0, con el motor y las particiones del curso.

Uso:
  python -I src/fusion/codificador_imagen_v0.py --dataset data/dataset_demo.npz --ejemplos compartido/ejemplos.csv \
      --salida resultados/dl --modelos modelos --experimentos imagen_v0 m0_demo_curva m0_demo_completo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from comun import entrenamiento as ent
from comun.protocolo import cargar_protocolo
from fusion import modelos_demo as md

ORIGEN = "v0 · código de la demo (base común, commit 6ffa66f) · no es resultado de ningún curso"
EXPERIMENTOS = {
    "imagen_v0": ent.Experimento("imagen_v0", md.imagen_v0, ("img",), aumentar=md.aumentar_demo),
    "m0_demo_curva": ent.Experimento(
        "m0_demo_curva", md.demo_curva, ("seq",), aumentar=md.aumentar_demo
    ),
    "m0_demo_completo": ent.Experimento(
        "m0_demo_completo", md.demo_completo, ("img", "seq"), aumentar=md.aumentar_demo
    ),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--ejemplos", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--modelos", type=Path, default=Path("modelos"))
    ap.add_argument("--experimentos", nargs="+", choices=sorted(EXPERIMENTOS), required=True)
    ap.add_argument("--presupuesto", type=float, default=None)
    ap.add_argument("--protocolo", type=Path, default=Path("config/protocolo.json"))
    a = ap.parse_args(argv)
    protocolo = cargar_protocolo(a.protocolo)
    protocolo["terciles_mes"] = json.loads(
        a.ejemplos.with_name("manifiesto_base.json").read_text(encoding="utf-8")
    )["resumen"]["terciles_mes"]
    with np.load(a.dataset, allow_pickle=False) as d:
        entradas = {c: d[c] for c in d.files}
    for nombre in a.experimentos:
        es_v0 = nombre == "imagen_v0"
        try:
            r = ent.entrenar_cv(
                EXPERIMENTOS[nombre],
                entradas,
                pd.read_csv(a.ejemplos),
                protocolo,
                protocolo["semillas"],
                a.salida,
                exportar=(a.modelos / "imagen") if es_v0 else None,
                version="v0",
                guardar_modelos=False,
                presupuesto_s=a.presupuesto,
            )
        except ent.Incompleto as e:
            print("INCOMPLETO:", e)
            return 3
        if es_v0:
            ruta = a.modelos / "imagen" / "v0" / "manifiesto.json"
            m = json.loads(ruta.read_text(encoding="utf-8"))
            m["origen"] = ORIGEN
            ruta.write_text(
                json.dumps(m, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(
            json.dumps(
                {
                    k: r[k]
                    for k in (
                        "experimento",
                        "parametros",
                        "auc_media_semillas",
                        "auc_desviacion_semillas",
                    )
                },
                ensure_ascii=False,
            ),
            json.dumps(r["semilla_0"]["global"]["auc_planeta_fp"]),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
