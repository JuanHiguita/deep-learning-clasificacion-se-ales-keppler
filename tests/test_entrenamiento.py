"""Pruebas del motor de entrenamiento (spec 003). Las que entrenan están marcadas como lentas."""
import json

import numpy as np
import pandas as pd
import pytest

from comun.protocolo import cargar_protocolo

keras = pytest.importorskip("keras")
from comun import entrenamiento as ent  # noqa: E402
from comun import verificar_contrato  # noqa: E402

PROTOCOLO = cargar_protocolo("config/protocolo.json")


def test_estandarizacion_ignora_faltantes_y_no_mira_la_prueba():
    x = np.array([[1.0, 100.0], [3.0, 0.0], [5.0, 300.0]])
    falt = np.array([[0, 0], [0, 1], [0, 0]])
    media, desv = ent.estadisticas_estandarizacion(x, falt)
    assert media == pytest.approx([3.0, 200.0]) and desv == pytest.approx([np.sqrt(8 / 3), 100.0])
    z = ent.aplicar_estandarizacion(x, media, desv, falt)
    assert z[1, 1] == 0 and z[0, 1] == pytest.approx(-1) and z[2, 1] == pytest.approx(1)
    # columna constante: no divide por cero
    _, d = ent.estadisticas_estandarizacion(np.ones((4, 1)))
    assert d[0] == 1.0


def test_pesos_por_clase():
    assert ent.pesos_por_clase(np.array([0, 0, 1, 2, 2, 2])) == pytest.approx({0: 1.0, 1: 2.0, 2: 2 / 3})


def _datos(n_estrellas=60, semilla=0):
    rng = np.random.default_rng(semilla)
    filas = []
    for i in range(n_estrellas):
        clase = i % 2
        comun = dict(kepid=1000 + i, kepoi_name=f"K{i + 1:05d}.01", particion=i % 5, prerregistrado=False, mes=10.0 + i,
                     fp_nt=0, fp_ss=int(clase == 1), fp_co=0, fp_ec=0)
        filas.append(dict(id_ejemplo=f"K{i + 1:05d}.01|transito", tipo="transito", clase=clase, **comun))
        filas.append(dict(id_ejemplo=f"K{i + 1:05d}.01|sin_senal", tipo="sin_senal", clase=2, **comun))
    ejemplos = pd.DataFrame(filas)
    y = ejemplos["clase"].to_numpy()
    centros = np.array([[2.0, 0.0], [-2.0, 0.0], [0.0, 3.0]]) * 100          # escala grande: exige estandarizar
    x = centros[y] + 60 * rng.standard_normal((len(y), 2))
    faltantes = np.zeros_like(x)
    disponible = np.ones(len(y))
    disponible[:4] = 0
    entradas = {"id_ejemplo": ejemplos["id_ejemplo"].to_numpy().astype("U"), "x": x.astype(np.float32),
                "faltantes": faltantes.astype(np.float32), "disponible": disponible}
    return ejemplos, entradas


def _constructor(semilla):
    a = keras.Input((2,), name="x")
    h = keras.layers.Dense(16, activation="relu")(a)
    h = keras.layers.Dense(16, activation="relu", name="embedding")(h)
    return keras.Model(a, keras.layers.Dense(3, activation="softmax")(h))


def _experimento(nombre="prueba"):
    return ent.Experimento(nombre, _constructor, ("x",), clave_disponible="disponible", estandarizar=("x",),
                           clave_faltantes="faltantes")


@pytest.mark.lenta
def test_entrenar_cv_de_punta_a_punta(tmp_path):
    ejemplos, entradas = _datos()
    protocolo = json.loads(json.dumps(PROTOCOLO))
    protocolo["entrenamiento"].update(epocas_max=25, paciencia=5)
    protocolo["metricas"]["bootstrap_remuestreos"] = 100
    protocolo["dimension_vector"]["escalares"] = 16
    r = ent.entrenar_cv(_experimento(), entradas, ejemplos, protocolo, [0, 1], tmp_path / "resultados",
                        exportar=tmp_path / "modelos" / "escalares", guardar_modelos=True)
    assert r["auc_media_semillas"] > 0.95 and r["n_no_disponibles"] == 4 and r["parametros"] > 0
    d = np.load(tmp_path / "resultados" / "prueba" / "pred_semilla0.npz")
    assert np.isfinite(d["p_prueba"]).all() and np.isfinite(d["p_validacion"]).all()
    assert np.allclose(d["p_prueba"].sum(axis=1), 1, atol=1e-4)
    # cada ejemplo es validación interna del modelo de la partición anterior a la suya
    assert (d["particion_modelo_validacion"] == (d["particion"] - 1) % 5).all()
    # reproducible con la misma semilla
    r2 = ent.entrenar_cv(_experimento("prueba2"), entradas, ejemplos, protocolo, [0], tmp_path / "resultados")
    d2 = np.load(tmp_path / "resultados" / "prueba2" / "pred_semilla0.npz")
    assert np.allclose(d["p_prueba"], d2["p_prueba"], atol=1e-5) and r2["por_semilla"]["0"] == r["por_semilla"]["0"]
    # exportación según el contrato y vector = activación de la capa embedding
    carpeta = tmp_path / "modelos" / "escalares" / "v1"
    assert verificar_contrato.verificar_rama(carpeta, "escalares", ejemplos, protocolo) == []
    assert (tmp_path / "resultados" / "prueba" / "modelos" / "particion_0_semilla_1.keras").is_file()
    cod = keras.saving.load_model(carpeta / "particion_0.keras", safe_mode=True)
    est = np.load(carpeta / "estandarizacion_particion_0.npz")
    z = cod.predict(ent.aplicar_estandarizacion(entradas["x"], est["x_media"], est["x_desviacion"]), verbose=0)
    assert np.allclose(z, np.load(carpeta / "vectores_particion_0.npz")["z"], atol=1e-4)


@pytest.mark.lenta
def test_pruebas_de_cordura(tmp_path):
    ejemplos, entradas = _datos(n_estrellas=100)
    protocolo = json.loads(json.dumps(PROTOCOLO))
    protocolo["entrenamiento"].update(epocas_max=20, paciencia=5)
    auc = ent.prueba_etiquetas_barajadas(_experimento(), entradas, ejemplos, protocolo, tmp_path)
    assert 0.3 < auc < 0.7                              # sin fuga, las etiquetas barajadas no se pueden aprender
    assert ent.prueba_sobreajuste_lote(_experimento(), entradas, ejemplos) >= 0.95
