"""Iteración 2: el espacio ampliado es válido y el plegado de tránsitos no cambia la función."""

import numpy as np
import pytest

from comun import busqueda
from temporal import busqueda_it2, busqueda_ramas, modelos


@pytest.mark.parametrize("arquitectura", list(busqueda_it2.ESPACIOS))
def test_espacio_valido_y_construible(arquitectura):
    configuraciones = busqueda_ramas.configuraciones_de(arquitectura, 4, busqueda_it2.ESPACIOS)
    assert configuraciones[0] == busqueda_it2.ESPACIOS[arquitectura][1]
    assert len({str(c) for c in configuraciones}) == 4
    for configuracion in configuraciones:
        modelo = busqueda_it2.CONSTRUCTORES[arquitectura](
            0, **busqueda.de_arquitectura(configuracion)
        )
        assert modelo.output_shape == (None, 3)


def test_la_configuracion_fija_es_la_red_del_protocolo():
    for arquitectura, (_, fija) in busqueda_it2.ESPACIOS.items():
        del_protocolo = busqueda_ramas._CONSTRUCTOR[arquitectura](0)
        de_la_busqueda = busqueda_it2.CONSTRUCTORES[arquitectura](
            0, **busqueda.de_arquitectura(fija)
        )
        assert de_la_busqueda.count_params() == del_protocolo.count_params()


@pytest.mark.parametrize("agregador", modelos.AGREGADORES)
def test_plegar_los_transitos_da_la_misma_salida(agregador):
    normal = modelos.m2(0, agregador)
    plegado = modelos.m2(0, agregador, plegado=True)
    plegado.set_weights(normal.get_weights())
    rng = np.random.default_rng(0)
    mascara = (rng.random((5, 32)) < 0.6).astype("float32")
    mascara[:, 0] = 1.0
    entradas = [
        rng.normal(size=(5, 32, 61)).astype("float32") * mascara[..., None],
        mascara,
        rng.normal(size=(5, 4)).astype("float32"),
    ]
    diferencia = np.abs(normal.predict(entradas, verbose=0) - plegado.predict(entradas, verbose=0))
    assert diferencia.max() < 1e-5


def test_profundidad_extra():
    assert modelos._con_profundidad((16, 32), 2) == (16, 32, 64, 128)
    with pytest.raises(ValueError):
        modelos._con_profundidad((16, 32), 4)
