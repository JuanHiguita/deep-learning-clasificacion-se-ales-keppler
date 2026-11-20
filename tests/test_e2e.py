"""Prueba de extremo a extremo con datos reales (spec 011). Se salta si los datos no están.

Ejecutar con:  pytest -q -m e2e
"""

import importlib.util
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
NECESARIOS = [
    RAIZ / "data" / "llc",
    RAIZ / "data" / "dataset_temporal.npz",
    RAIZ / "data" / "dataset_demo.npz",
    RAIZ / "modelos" / "sistema" / "manifiesto.json",
    RAIZ / "resultados" / "dl" / "calibracion" / "pred_calibrada.npz",
]


@pytest.mark.e2e
@pytest.mark.skipif(not all(r.exists() for r in NECESARIOS), reason="faltan datos o modelos reales")
def test_de_los_fits_a_la_probabilidad_calibrada():
    ruta = RAIZ / "scripts" / "prueba_e2e.py"
    spec = importlib.util.spec_from_file_location("prueba_e2e", ruta)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["prueba_e2e"] = modulo
    spec.loader.exec_module(modulo)

    informe = modulo.probar(estrellas_por_particion=1)
    assert len(informe["estrellas"]) == 5
    for estrella in informe["estrellas"]:
        assert estrella["correcto"], estrella
