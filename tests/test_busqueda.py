"""Búsqueda de hiperparámetros (spec 009): muestreo, selección anidada y espacios declarados."""

import numpy as np
import pytest

from comun import busqueda
from temporal import busqueda_ramas, modelos

ESPACIO = {"tasa_aprendizaje": ("log", 3e-4, 3e-3), "lote": [16, 32, 64], "ancho": [0.5, 1.0]}
FIJA = {"tasa_aprendizaje": 1e-3, "lote": 32, "ancho": 1.0}


def test_muestreo_reproducible_y_la_primera_es_la_del_protocolo():
    a = busqueda.muestrear(ESPACIO, FIJA, 6, 0)
    assert a == busqueda.muestrear(ESPACIO, FIJA, 6, 0)
    assert a[0] == FIJA
    assert a[:4] == busqueda.muestrear(ESPACIO, FIJA, 4, 0)  # pedir menos no cambia las primeras
    assert a != busqueda.muestrear(ESPACIO, FIJA, 6, 1)  # otra arquitectura, otro sorteo
    assert len({tuple(sorted(c.items())) for c in a}) == 6  # sin repetidas


def test_las_configuraciones_caen_dentro_del_espacio():
    for c in busqueda.muestrear(ESPACIO, FIJA, 20, 3)[1:]:
        assert 3e-4 <= c["tasa_aprendizaje"] <= 3e-3
        assert c["lote"] in ESPACIO["lote"] and c["ancho"] in ESPACIO["ancho"]


def test_la_configuracion_fija_debe_cubrir_el_espacio():
    with pytest.raises(ValueError):
        busqueda.muestrear(ESPACIO, {"lote": 32}, 3, 0)


def test_protocolo_con_no_modifica_el_original():
    protocolo = {"entrenamiento": {"tasa_aprendizaje": 1e-3, "lote": 32, "paciencia": 15}}
    nuevo = busqueda.protocolo_con(protocolo, {"tasa_aprendizaje": 2e-3, "ancho": 2.0})
    assert nuevo["entrenamiento"] == {"tasa_aprendizaje": 2e-3, "lote": 32, "paciencia": 15}
    assert protocolo["entrenamiento"]["tasa_aprendizaje"] == 1e-3
    assert busqueda.de_arquitectura({"lote": 16, "ancho": 2.0, "dropout": 0.1}) == {
        "ancho": 2.0,
        "dropout": 0.1,
    }


def test_seleccion_anidada_elige_por_particion_y_conserva_la_fija_en_empates():
    #                 k0   k1   k2
    nll = np.array([[0.5, 0.4, 0.3], [0.4, 0.4, 0.9], [0.6, 0.1, 0.3]])
    assert busqueda.seleccion_anidada(nll).tolist() == [1, 2, 0]


def test_nll_de_validacion_usa_solo_la_validacion_de_cada_modelo():
    y = np.array([0, 0, 1, 1])
    pred = {
        "p_validacion": np.array([[0.9, 0.1, 0], [0.9, 0.1, 0], [0.5, 0.5, 0], [0.5, 0.5, 0]]),
        "particion_modelo_validacion": np.array([0, 0, 1, 1]),
    }
    k0, k1 = busqueda.nll_de_validacion_por_particion(pred, y, 2)
    assert k0 == pytest.approx(-np.log(0.9)) and k1 == pytest.approx(-np.log(0.5))


def test_espacios_declarados_incluyen_la_configuracion_del_protocolo():
    for arquitectura in busqueda_ramas.ESPACIOS:
        lista = busqueda_ramas.configuraciones_de(arquitectura)
        assert len(lista) == busqueda_ramas.N_CONFIGURACIONES
        assert lista[0] == busqueda_ramas.ESPACIOS[arquitectura][1]


def test_los_valores_por_defecto_de_los_modelos_son_los_del_protocolo():
    # si cambia un conteo, la configuración 0 dejó de ser el experimento publicado
    assert modelos.m4_escalares(0).count_params() == 1059
    assert modelos.m2(0, "promedio").count_params() == 5027
    assert modelos.m2(0, "atencion").count_params() == 13604
    assert modelos.m2(0, "atencion", ancho=2.0, cabezas=4).count_params() > 13604


def test_cada_configuracion_construye_un_modelo_con_el_vector_del_contrato():
    for arquitectura, dimension in (("m2_gru", 32), ("m2_atencion", 32), ("m4_escalares", 16)):
        for indice, configuracion in enumerate(busqueda_ramas.configuraciones_de(arquitectura, 4)):
            modelo = busqueda_ramas.experimento_de(arquitectura, indice, configuracion).constructor(
                0
            )
            assert modelo.get_layer("embedding").output.shape[-1] == dimension
