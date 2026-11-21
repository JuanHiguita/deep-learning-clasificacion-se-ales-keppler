"""Iteración 2: elección anidada entre redes y rama ausente en la fusión."""

import numpy as np
import pandas as pd
import pytest

from fusion import iteracion_2, vectores


def _busqueda(carpeta, red, nll):
    (carpeta / red).mkdir(parents=True)
    (carpeta / red / "resumen.json").write_text("{}", encoding="utf-8")
    filas = [
        {"configuracion": f"cfg{i:02d}", "parametros": 10, "nll_val_k0": a, "nll_val_k1": b}
        for i, (a, b) in enumerate(nll)
    ]
    pd.DataFrame(filas).to_csv(carpeta / red / "configuraciones.csv", index=False)


def test_cada_particion_elige_con_su_validacion(tmp_path):
    _busqueda(tmp_path, "m2_promedio", [(0.5, 0.2), (0.4, 0.9)])
    _busqueda(tmp_path, "m2_gru", [(0.3, 0.6), (0.7, 0.7)])
    _busqueda(tmp_path, "m2_atencion", [(0.6, 0.6), (0.6, 0.6)])
    elegidas = iteracion_2.elegir(tmp_path, "transitos", 2)
    assert [(e["red"], e["configuracion"]) for e in elegidas] == [
        ("m2_gru", "cfg00"),
        ("m2_promedio", "cfg00"),
    ]


def test_no_elige_si_falta_una_busqueda(tmp_path):
    _busqueda(tmp_path, "m2_promedio", [(0.5, 0.2)])
    with pytest.raises(FileNotFoundError):
        iteracion_2.elegir(tmp_path, "transitos", 2)


def test_rama_ausente_queda_en_cero_con_bandera_cero(tmp_path):
    ids = np.array(["a", "b"])
    for rama in ("vistas", "transitos", "escalares"):
        (tmp_path / rama / "v1").mkdir(parents=True)
        np.savez(
            tmp_path / rama / "v1" / "vectores_particion_0.npz",
            id_ejemplo=ids,
            z=np.ones((2, vectores.DIMENSION[rama]), np.float32),
        )
    dataset = {
        "id_ejemplo": ids,
        "disponible_vistas": np.array([True, True]),
        "disponible_transitos": np.array([True, False]),
    }
    versiones = {"vistas": "v1", "transitos": "v1", "imagen": None, "escalares": "v1"}
    v, banderas = vectores.cargar_vectores(tmp_path, versiones, 0, ids, dataset)
    assert v["imagen"].shape == (2, 32) and not v["imagen"].any()
    assert banderas.tolist() == [[1, 1, 0, 1], [1, 0, 0, 1]]
