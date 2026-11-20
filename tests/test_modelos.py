"""Pruebas de los modelos (spec 004): formas, máscaras, invariancia al orden y serialización."""

import numpy as np
import pytest

keras = pytest.importorskip("keras")
from temporal import aumentacion, modelos  # noqa: E402


def _lote(n=6, semilla=0):
    rng = np.random.default_rng(semilla)
    tr = rng.normal(size=(n, 32, 61)).astype("float32")
    mk = np.zeros((n, 32), "float32")
    for i, k in enumerate([32, 20, 8, 4, 32, 12][:n]):
        mk[i, :k] = 1
    return {
        "transitos": tr * mk[:, :, None],
        "mascara_transitos": mk,
        "consistencia": rng.normal(size=(n, 4)).astype("float32"),
    }


def _permutar(x, semilla=1):
    rng = np.random.default_rng(semilla)
    y = {k: v.copy() for k, v in x.items()}
    for i in range(len(x["transitos"])):
        k = int(x["mascara_transitos"][i].sum())
        p = rng.permutation(k)
        y["transitos"][i, :k] = x["transitos"][i, :k][p]
    return y


@pytest.mark.parametrize("agregador", modelos.AGREGADORES)
def test_m2_formas_mascara_y_orden(agregador, tmp_path):
    m = modelos.m2(0, agregador)
    x = _lote()
    p = m.predict(x, verbose=0)
    assert p.shape == (6, 3) and np.allclose(p.sum(1), 1, atol=1e-5) and np.isfinite(p).all()
    assert m.get_layer("embedding").output.shape[-1] == 32 and m.count_params() < 200_000
    # el contenido de un tránsito vacío no cambia la salida
    sucio = {k: v.copy() for k, v in x.items()}
    sucio["transitos"][2, 8:] = 99.0
    assert np.allclose(m.predict(sucio, verbose=0), p, atol=1e-5)
    # invariancia al orden: promedio y atención sí; la GRU no
    cambio = np.abs(m.predict(_permutar(x), verbose=0) - p).max()
    assert (cambio > 1e-4) if agregador == "gru" else (cambio < 1e-5)
    # guardar y cargar en modo seguro conserva las predicciones
    m.save(tmp_path / "m.keras")
    m2 = keras.saving.load_model(tmp_path / "m.keras", safe_mode=True)
    assert np.allclose(m2.predict(x, verbose=0), p, atol=1e-5)


def test_pesos_de_atencion_suman_uno_y_respetan_la_mascara():
    m = modelos.m2(0, "atencion")
    x = _lote()
    w = modelos.submodelo_pesos_atencion(m).predict(x, verbose=0)
    assert w.shape == (6, 32) and np.allclose(w.sum(1), 1, atol=1e-5)
    assert np.all(w[x["mascara_transitos"] == 0] == 0)


@pytest.mark.parametrize("pooling", ["max", "promedio", "paso"])
def test_m1_y_m4(pooling):
    m = modelos.m1_astronet(0, pooling)
    p = m.predict(
        {
            "vista_global": np.zeros((3, 2001, 1), "float32"),
            "vista_local": np.zeros((3, 201, 4), "float32"),
        },
        verbose=0,
    )
    assert (
        p.shape == (3, 3)
        and m.get_layer("embedding").output.shape[-1] == 32
        and m.count_params() < 200_000
    )
    m4 = modelos.m4_escalares(0)
    assert m4.get_layer("embedding").output.shape[-1] == 16


def test_aumentacion_de_transitos():
    x = _lote()
    a = aumentacion.aumentar_transitos(x, np.random.default_rng(0))
    b = aumentacion.aumentar_transitos(x, np.random.default_rng(0))
    assert np.array_equal(a["transitos"], b["transitos"])  # reproducible
    n = a["mascara_transitos"].sum(1)
    assert np.all(n <= x["mascara_transitos"].sum(1)) and np.all(n >= 4)
    for i in range(len(n)):  # vacíos al final y en cero
        k = int(n[i])
        assert a["mascara_transitos"][i, :k].all() and not a["mascara_transitos"][i, k:].any()
        assert not a["transitos"][i, k:].any()
    v = aumentacion.aumentar_vistas(
        {
            "vista_global": np.zeros((2, 2001, 1), "float32"),
            "vista_local": np.zeros((2, 201, 4), "float32"),
        },
        np.random.default_rng(0),
    )
    assert v["vista_local"].shape == (2, 201, 4) and v["vista_global"].std() > 0
