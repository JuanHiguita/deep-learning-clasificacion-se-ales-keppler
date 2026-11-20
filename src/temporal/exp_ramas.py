"""Experimentos de las ramas (spec 004): pruebas de cordura (E0), comparación (E1, E2) y exportación v1.

Uso:
  python -I src/temporal/exp_ramas.py --dataset data/dataset_temporal.npz --ejemplos compartido/ejemplos.csv \
      --salida resultados/dl --modelos modelos --experimentos m4_escalares [--cordura] [--exportar]
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
from temporal import modelos

EXPERIMENTOS = {
    "m4_escalares": (ent.Experimento("m4_escalares", modelos.m4_escalares, ("escalares", "faltantes"),
                                     estandarizar=("escalares",), clave_faltantes="faltantes"), "escalares"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--ejemplos", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--modelos", type=Path, default=Path("modelos"))
    ap.add_argument("--experimentos", nargs="+", choices=sorted(EXPERIMENTOS), required=True)
    ap.add_argument("--cordura", action="store_true", help="solo las pruebas de cordura (E0)")
    ap.add_argument("--exportar", action="store_true", help="exporta la semilla congelada como codificador v1")
    ap.add_argument("--protocolo", type=Path, default=Path("config/protocolo.json"))
    a = ap.parse_args(argv)

    protocolo = cargar_protocolo(a.protocolo)
    ejemplos = pd.read_csv(a.ejemplos)
    base = json.loads((a.ejemplos.with_name("manifiesto_base.json")).read_text(encoding="utf-8"))
    protocolo["terciles_mes"] = base["resumen"]["terciles_mes"]
    with np.load(a.dataset, allow_pickle=False) as d:
        entradas = {c: d[c] for c in d.files}
    for nombre in a.experimentos:
        exp, rama = EXPERIMENTOS[nombre]
        if a.cordura:
            auc = ent.prueba_etiquetas_barajadas(exp, entradas, ejemplos, protocolo, a.salida / "cordura")
            acc = ent.prueba_sobreajuste_lote(exp, entradas, ejemplos)
            ok = 0.40 <= auc <= 0.60 and acc >= 0.95
            print(f"{nombre}: AUC con etiquetas barajadas = {auc:.3f}; exactitud al memorizar un lote = {acc:.3f} "
                  f"→ {'PASA' if ok else 'FALLA: detenerse'}")
            if not ok:
                return 1
            continue
        r = ent.entrenar_cv(exp, entradas, ejemplos, protocolo, protocolo["semillas"], a.salida,
                            exportar=(a.modelos / rama) if a.exportar else None, guardar_modelos=True)
        print(json.dumps({k: r[k] for k in ("experimento", "parametros", "auc_media_semillas",
                                             "auc_desviacion_semillas", "nll_validacion_media")}, ensure_ascii=False))
        print(json.dumps(r["semilla_0"]["global"], ensure_ascii=False))
        if r["auc_media_semillas"] > 0.99:
            print("ATENCIÓN: AUC > 0,99. Detenerse y auditar fugas antes de continuar.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
