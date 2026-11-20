"""Pruebas del motor ya implementado: protocolo, métricas y estadística."""

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from comun import estadistica, metricas, protocolo


def _datos(n=400, semilla=0):
    rng = np.random.default_rng(semilla)
    y = rng.integers(0, 3, n)
    logits = rng.normal(0, 1, (n, 3))
    logits[np.arange(n), y] += 1.5
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return p, y


# ---------- protocolo ----------


def test_protocolo_se_carga():
    pr = protocolo.cargar_protocolo("config/protocolo.json")
    assert pr["particiones"]["n"] == 5
    assert sum(pr["dimension_vector"].values()) == 112


def test_separacion_es_disjunta_y_completa():
    particion = np.repeat(np.arange(5), 20)
    for k in range(5):
        s = protocolo.separar(particion, k)
        assert not np.any(s.entrenamiento & s.validacion)
        assert not np.any(s.entrenamiento & s.prueba)
        assert not np.any(s.validacion & s.prueba)
        assert np.all(s.entrenamiento | s.validacion | s.prueba)
        assert set(particion[s.validacion]) == {(k + 1) % 5}
        assert s.entrenamiento.sum() == 60


def test_prerregistrados_quedan_fuera():
    particion = np.repeat(np.arange(5), 4)
    pre = np.zeros(20, bool)
    pre[[0, 5, 10]] = True
    s = protocolo.separar(particion, 0, prerregistrado=pre)
    assert not np.any((s.entrenamiento | s.validacion | s.prueba) & pre)


def test_detecta_estrella_repetida():
    particion = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    kepid = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    protocolo.comprobar_sin_solape(kepid, protocolo.separar(particion, 0))
    kepid[2] = 1  # la estrella 1 pasa a estar en prueba y en validación
    with pytest.raises(ValueError, match="Fuga"):
        protocolo.comprobar_sin_solape(kepid, protocolo.separar(particion, 0))


# ---------- métricas ----------


def test_auc_coincide_con_sklearn_incluso_con_empates():
    rng = np.random.default_rng(1)
    s = np.round(rng.random(300), 1)  # muchos empates
    pos = rng.random(300) < 0.4
    assert metricas.auc(s, pos) == pytest.approx(roc_auc_score(pos, s), abs=1e-12)
    w = rng.random(300) + 0.1
    assert metricas.auc(s, pos, w) == pytest.approx(
        roc_auc_score(pos, s, sample_weight=w), abs=1e-12
    )


def test_auc_planeta_fp_ignora_sin_senal():
    p, y = _datos()
    real = y < 2
    q = p[real, 0] / (p[real, 0] + p[real, 1])
    assert metricas.auc_planeta_fp(p, y) == pytest.approx(roc_auc_score(y[real] == 0, q), abs=1e-9)


def test_auc_sin_una_clase_es_nan():
    assert np.isnan(metricas.auc(np.array([0.1, 0.9]), np.array([True, True])))


def test_nll_y_brier_coinciden_con_sklearn():
    p, y = _datos()
    assert metricas.nll(p, y) == pytest.approx(log_loss(y, p, labels=[0, 1, 2]), abs=1e-9)
    q = np.array([0.2, 0.7, 0.9, 0.4])
    pos = np.array([0, 1, 1, 0])
    p2 = np.stack([q, 1 - q], axis=1)
    # Brier multiclase de 2 clases = 2 × Brier binario
    assert metricas.brier(p2, 1 - pos) == pytest.approx(2 * brier_score_loss(pos, q), abs=1e-12)


def test_ece_caso_a_mano():
    # Dos intervalos: confianza 0.75 con acierto 0.5 (mitad de los datos) y 0.95 con acierto 1.0.
    p = np.array([[0.75, 0.25, 0.0], [0.75, 0.25, 0.0], [0.95, 0.05, 0.0], [0.95, 0.05, 0.0]])
    y = np.array([0, 1, 0, 0])
    esperado = 0.5 * abs(0.5 - 0.75) + 0.5 * abs(1.0 - 0.95)
    assert metricas.ece_top(p, y, n_intervalos=10) == pytest.approx(esperado, abs=1e-12)


def test_ece_de_modelo_perfectamente_calibrado_es_pequeno():
    rng = np.random.default_rng(3)
    q = rng.random(200_000)
    pos = rng.random(200_000) < q
    assert metricas.ece_binario(q, pos) < 0.01


def test_probabilidades_invalidas_fallan():
    with pytest.raises(ValueError):
        metricas.nll(np.array([[0.5, 0.6, 0.1]]), np.array([0]))
    with pytest.raises(ValueError):
        metricas.nll(np.array([[np.nan, 0.5, 0.5]]), np.array([0]))


def test_correccion_de_prior():
    q = np.array([0.1, 0.5, 0.9])
    # Sin cambio de tasa no cambia nada.
    assert np.allclose(metricas.corregir_prior(q, 0.5, 0.5), q)
    # Con tasa objetivo menor, la probabilidad baja y conserva el orden.
    q2 = metricas.corregir_prior(q, 0.5, 0.05)
    assert np.all(q2 < q) and np.all(np.diff(q2) > 0)
    # Verificación con la regla de Bayes en un caso exacto.
    assert q2[1] == pytest.approx(0.05, abs=1e-12)


def test_correccion_de_prior_recupera_la_calibracion():
    # Datos sintéticos con cambio de tasa base puro: la corrección debe reducir el ECE ponderado.
    rng = np.random.default_rng(5)
    n = 100_000
    pos = rng.random(n) < 0.5
    x = rng.normal(np.where(pos, 1.0, -1.0), 1.0)
    q = 1 / (1 + np.exp(-2 * x))  # posterior exacto con tasa 0.5
    w = metricas.pesos_para_tasa_base(pos, 0.05)
    assert np.sum(w * pos) / w.sum() == pytest.approx(0.05)
    sin = metricas.ece_binario(q, pos, pesos=w)
    con = metricas.ece_binario(metricas.corregir_prior(q, 0.5, 0.05), pos, pesos=w)
    assert con < 0.01 < sin


# ---------- estadística ----------


def test_bootstrap_de_modelos_identicos_da_cero():
    p, y = _datos()
    grupos = np.repeat(np.arange(200), 2)
    est = estadistica.diferencia_media_semillas(metricas.auc_planeta_fp, y, [p, p], [p, p])
    r = estadistica.bootstrap_por_estrella(grupos, est, n_remuestreos=200)
    assert r.estimacion == 0 and r.ic_inf == 0 and r.ic_sup == 0 and r.incluye(0)


def test_bootstrap_detecta_una_diferencia_real():
    p, y = _datos(n=600)
    rng = np.random.default_rng(9)
    ruido = rng.random(p.shape)
    peor = 0.4 * p + 0.6 * ruido / ruido.sum(axis=1, keepdims=True)
    grupos = np.repeat(np.arange(300), 2)
    est = estadistica.diferencia_media_semillas(metricas.auc_planeta_fp, y, [p], [peor])
    r = estadistica.bootstrap_por_estrella(grupos, est, n_remuestreos=500)
    assert r.estimacion > 0 and not r.incluye(0)


def test_bootstrap_es_reproducible_y_agrupa_por_estrella():
    grupos = np.array([10, 10, 20, 20, 30, 30, 40, 40])
    vistos = []

    def est(indices):
        vistos.append(indices.copy())
        return float(len(indices))

    a = estadistica.bootstrap_por_estrella(grupos, est, n_remuestreos=50, semilla=1)
    b = estadistica.bootstrap_por_estrella(grupos, est, n_remuestreos=50, semilla=1)
    assert a == b
    for indices in vistos:
        # cada estrella entra completa: sus dos ejemplos aparecen el mismo número de veces
        cuenta = np.bincount(indices, minlength=8)
        assert np.all(cuenta[0::2] == cuenta[1::2])


def test_bootstrap_falla_si_el_estrato_es_demasiado_pequeno():
    grupos = np.arange(4)
    y = np.array([0, 1, 1, 1])
    s = np.array([0.9, 0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="demasiado pequeño"):
        estadistica.bootstrap_por_estrella(
            grupos, lambda i: metricas.auc(s[i], y[i] == 0), n_remuestreos=200
        )
