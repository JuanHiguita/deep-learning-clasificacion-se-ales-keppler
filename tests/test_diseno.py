"""Estudio de diseño (spec 010): optimizadores, pérdidas, variantes, E4 y preprocesamiento."""

import keras
import numpy as np
import pytest

from comun import entrenamiento as ent
from comun import metricas
from temporal import e4_menos_transitos as e4
from temporal import estudio_diseno

CFG = {"tasa_aprendizaje": 1e-3, "weight_decay": 1e-4}


def test_el_optimizador_por_defecto_es_el_del_protocolo():
    assert isinstance(ent.crear_optimizador(CFG), keras.optimizers.AdamW)
    assert isinstance(ent.crear_optimizador({**CFG, "optimizador": "adam"}), keras.optimizers.Adam)
    sgd = ent.crear_optimizador({**CFG, "optimizador": "sgd"})
    assert isinstance(sgd, keras.optimizers.SGD) and float(sgd.learning_rate) == pytest.approx(1e-2)
    with pytest.raises(ValueError):
        ent.crear_optimizador({**CFG, "optimizador": "otro"})


def test_la_perdida_por_defecto_es_la_entropia_cruzada():
    assert ent.crear_perdida({}) == "sparse_categorical_crossentropy"
    with pytest.raises(ValueError):
        ent.crear_perdida({"perdida": "otra"})


@pytest.mark.parametrize("nombre", ["suavizado", "focal"])
def test_las_perdidas_alternativas_dan_un_valor_por_ejemplo(nombre):
    y = np.array([0, 1, 2, 1])
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.6, 0.3, 0.1]], "float32")
    valores = np.asarray(ent.crear_perdida({"perdida": nombre})(y, p))
    assert valores.shape == (4,) and np.isfinite(valores).all() and (valores > 0).all()
    assert valores[3] == valores.max()  # el único ejemplo mal clasificado es el que más cuesta


def test_cada_variante_cambia_un_solo_factor_y_tiene_referencia():
    for arquitectura, variantes in estudio_diseno.VARIANTES.items():
        nombres = list(variantes)
        assert nombres[0] == "referencia", arquitectura
        for nombre, v in variantes.items():
            if nombre == "referencia":
                continue
            cambios = [
                bool(v.modelo),
                bool(v.entrenamiento),
                not v.aumentacion,
                not v.estandarizar,
                v.series_en_zscore,
            ]
            assert sum(cambios) == 1, f"{arquitectura}/{nombre} debe cambiar una sola cosa"


def test_ninguna_variante_quita_las_banderas_de_dato_faltante():
    for variantes in estudio_diseno.VARIANTES.values():
        for v in variantes.values():
            assert "faltantes" not in v.sin_entradas and "mascara_transitos" not in v.sin_entradas


def test_las_variantes_de_la_rama_4_construyen_y_respetan_el_vector_del_contrato():
    for nombre, v in estudio_diseno.VARIANTES["m4_escalares"].items():
        modelo = estudio_diseno.experimento_de("m4_escalares", nombre, v).constructor(0)
        assert modelo.get_layer("embedding").output.shape[-1] == 16
        assert modelo.output_shape[-1] == 3


def test_la_rama_2_sin_consistencia_no_recibe_esa_entrada():
    v = estudio_diseno.VARIANTES["m2_atencion"]["sin_consistencia"]
    experimento = estudio_diseno.experimento_de("m2_atencion", "sin_consistencia", v)
    assert experimento.claves_entrada == ("transitos", "mascara_transitos")
    assert experimento.estandarizar == ()
    assert [e.name for e in experimento.constructor(0).inputs] == ["transitos", "mascara_transitos"]


def test_series_en_zscore_no_toca_el_original():
    rng = np.random.default_rng(0)
    dataset = {
        "vista_global": rng.normal(3, 2, (4, 50, 1)).astype("float32"),
        "vista_local": rng.normal(-1, 5, (4, 20, 4)).astype("float32"),
        "transitos": np.concatenate(
            [rng.normal(0, 3, (4, 5, 30)), np.zeros((4, 1, 30))], axis=1
        ).astype("float32"),
        "escalares": np.ones((4, 7), "float32"),
    }
    copia = estudio_diseno.series_en_zscore(dataset)
    assert np.abs(copia["vista_local"].mean(axis=1)).max() < 1e-5
    assert np.abs(copia["vista_local"].std(axis=1) - 1).max() < 1e-4
    assert (copia["transitos"][:, 5] == 0).all()  # una serie constante queda en cero, no en NaN
    assert copia["escalares"] is dataset["escalares"]
    assert dataset["vista_global"].mean() > 2


def test_conservar_deja_como_mucho_n_transitos_y_solo_validos():
    mascara = np.array([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]], "float32")
    nueva = e4.conservar(mascara, 3, np.random.default_rng(0))
    assert nueva.sum(axis=1).tolist() == [3, 2]
    assert ((nueva == 1) <= (mascara == 1)).all()


def test_el_auc_por_rangos_coincide_con_el_del_proyecto():
    rng = np.random.default_rng(1)
    puntaje = np.round(rng.random(200), 1)  # con empates
    positivo = rng.random(200) > 0.4
    assert e4._auc_por_rangos(puntaje, positivo) == pytest.approx(metricas.auc(puntaje, positivo))
