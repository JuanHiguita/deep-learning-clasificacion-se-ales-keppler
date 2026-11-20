"""Piezas del notebook que no dependen de los resultados: controles en modo fijo y gráficos base."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from informe import datos, graficos, interactivo  # noqa: E402
from informe.interactivo import Deslizador, Lista, controles  # noqa: E402


def test_valor_inicial_de_los_controles():
    assert Lista("Métrica:", [("AUC", "auc"), ("NLL", "nll")]).valor_inicial() == "auc"
    assert Lista("Semilla:", [0, 1, 2]).valor_inicial() == 0
    assert Lista("Semilla:", [0, 1, 2], inicial=2).valor_inicial() == 2
    assert Deslizador("T:", 0.4, 3.0, 0.1, 1.0).valor_inicial() == 1.0


def test_modo_fijo_dibuja_una_vez_con_los_valores_iniciales(monkeypatch, capsys):
    monkeypatch.setenv(interactivo.VARIABLE_MODO_FIJO, "1")
    llamadas = []
    controles(
        lambda metrica, n: llamadas.append((metrica, n)),
        metrica=Lista("Métrica:", [("AUC", "auc")]),
        n=Deslizador("Intervalos:", 5, 20, 1, 10),
    )
    assert not interactivo.hay_controles()
    assert llamadas == [("auc", 10)]
    assert "Vista fija" in capsys.readouterr().out


def test_bosque_dibuja_un_punto_y_un_intervalo_por_fila():
    fig, ax = plt.subplots()
    graficos.bosque(ax, ["a", "b"], [0.9, 0.8], [0.85, 0.7], [0.95, 0.9], referencia=0.5)
    etiquetas = [t.get_text() for t in ax.get_yticklabels()]
    altura = dict(zip(etiquetas, ax.get_yticks(), strict=True))
    assert altura["a"] > altura["b"]  # la primera etiqueta queda arriba
    assert len(ax.lines) == 2 * 2 + 1  # intervalo y punto por fila, más la línea de referencia
    plt.close(fig)
    with pytest.raises(ValueError):
        graficos.bosque(plt.subplots()[1], ["a", "b"], [0.9], [0.8], [1.0])


def test_confiabilidad_omite_los_grupos_con_pocos_ejemplos():
    rng = np.random.default_rng(0)
    confianza = np.r_[np.full(200, 0.95), np.full(3, 0.55)]  # el segundo grupo tiene 3 ejemplos
    p = np.c_[confianza, 1 - confianza, np.zeros_like(confianza)]
    y = (rng.random(len(p)) > 0.9).astype(int)
    fig, ax = plt.subplots()
    graficos.confiabilidad(ax, p, y, "prueba", n_intervalos=10)
    assert len(ax.collections[0].get_offsets()) == 1
    plt.close(fig)


def test_buscar_raiz(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "protocolo.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notebooks").mkdir()
    assert datos.buscar_raiz(tmp_path / "notebooks") == tmp_path
    with pytest.raises(FileNotFoundError):
        datos.buscar_raiz(tmp_path.parent)
