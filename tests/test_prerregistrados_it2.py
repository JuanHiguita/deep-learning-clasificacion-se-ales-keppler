"""Iteración 2: el estadístico declarado para las prerregistradas."""

import numpy as np

from fusion import it2_prerregistrados as pre

CFG = {"bootstrap_remuestreos": 200, "bootstrap_semilla": 0}


def test_A_alto_si_las_prerregistradas_puntuan_mas():
    rng = np.random.default_rng(0)
    r = pre.resumir(
        rng.uniform(0.6, 1.0, 40),
        np.arange(40),
        rng.uniform(0.0, 0.4, 300),
        1000 + np.arange(300),
        CFG,
    )
    assert r["A"]["estimacion"] == 1.0
    assert r["hipotesis_A_mayor_que_0.5"].startswith("respaldada")
    assert r["n"] == {"prerregistradas": 40, "falsos_positivos_normales": 300}


def test_no_concluyente_si_los_grupos_son_iguales():
    rng = np.random.default_rng(1)
    r = pre.resumir(
        rng.uniform(0, 1, 40), np.arange(40), rng.uniform(0, 1, 300), 1000 + np.arange(300), CFG
    )
    assert r["hipotesis_A_mayor_que_0.5"].startswith("no concluyente")
