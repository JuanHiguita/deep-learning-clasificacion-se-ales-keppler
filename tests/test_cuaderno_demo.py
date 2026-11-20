"""Pruebas del notebook demostrativo (spec 012): paquete, extracción segura, modos y celdas."""

import importlib.util
import json
from pathlib import Path

import nbformat
import pytest

from cuaderno_demo import extraccion, flujo

RAIZ_REPO = Path(__file__).resolve().parents[1]


def _constructor():
    """Carga scripts/construir_notebook_demo.py como módulo (scripts/ no es un paquete)."""
    ruta = RAIZ_REPO / "scripts" / "construir_notebook_demo.py"
    spec = importlib.util.spec_from_file_location("construir_notebook_demo", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


ARCHIVOS = {"src/a.py": b"x = 1\n", "data/b.csv": b"1,2\n3,4\n"}


def test_extraer_verifica_y_es_idempotente(tmp_path):
    paquete, manifiesto = _constructor().empaquetar(dict(ARCHIVOS), "abc1234")
    primera = extraccion.extraer(paquete, manifiesto, tmp_path)
    segunda = extraccion.extraer(paquete, manifiesto, tmp_path)
    total = sum(len(v) for v in ARCHIVOS.values())
    assert primera == {"escritos": 2, "ya_estaban": 0, "bytes": total}
    assert segunda == {"escritos": 0, "ya_estaban": 2, "bytes": total}
    assert (tmp_path / "data" / "b.csv").read_bytes() == ARCHIVOS["data/b.csv"]


def test_el_paquete_es_determinista():
    constructor = _constructor()
    primero = constructor.empaquetar(dict(ARCHIVOS), "c")[0]
    assert primero == constructor.empaquetar(dict(ARCHIVOS), "c")[0]


def test_rechaza_paquete_alterado(tmp_path):
    paquete, manifiesto = _constructor().empaquetar(dict(ARCHIVOS), "abc1234")
    manifiesto["archivos"]["src/a.py"] = "0" * 64
    with pytest.raises(extraccion.PaqueteInvalido):
        extraccion.extraer(paquete, manifiesto, tmp_path)


def test_rechaza_huella_global_distinta(tmp_path):
    paquete, manifiesto = _constructor().empaquetar(dict(ARCHIVOS), "abc1234")
    manifiesto["sha256_paquete"] = "f" * 64
    with pytest.raises(extraccion.PaqueteInvalido):
        extraccion.extraer(paquete, manifiesto, tmp_path)


def test_rechaza_archivos_que_no_estan_en_el_manifiesto(tmp_path):
    paquete, manifiesto = _constructor().empaquetar(dict(ARCHIVOS), "abc1234")
    del manifiesto["archivos"]["data/b.csv"]
    with pytest.raises(extraccion.PaqueteInvalido):
        extraccion.extraer(paquete, manifiesto, tmp_path)


@pytest.mark.parametrize("nombre", ["../fuera.py", "/abs.py", "a\\b.py", "C:x.py", "a/../../b.py"])
def test_rechaza_rutas_peligrosas(tmp_path, nombre):
    with pytest.raises(extraccion.PaqueteInvalido):
        extraccion.ruta_segura(nombre, tmp_path)


def test_ruta_segura_normal(tmp_path):
    assert extraccion.ruta_segura("src/x/y.py", tmp_path) == (tmp_path / "src/x/y.py").resolve()


def test_modo_rapido_solo_cambia_las_semillas(tmp_path):
    (tmp_path / "config").mkdir()
    oficial = {"semillas": [0, 1, 2], "semillas_cabeza_fusion": [0, 1, 2, 3, 4], "otra": 1}
    (tmp_path / "config" / "protocolo.json").write_text(json.dumps(oficial), encoding="utf-8")
    ruta = flujo.preparar_modo(tmp_path, "rapido")
    rapido = json.loads((tmp_path / ruta).read_text(encoding="utf-8"))
    assert rapido["semillas"] == [0]
    assert rapido["semillas_cabeza_fusion"] == oficial["semillas_cabeza_fusion"]
    assert rapido["otra"] == 1


def test_una_carpeta_no_mezcla_modos(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "protocolo.json").write_text('{"semillas": [0, 1, 2]}', encoding="utf-8")
    flujo.preparar_modo(tmp_path, "rapido")
    with pytest.raises(RuntimeError):
        flujo.preparar_modo(tmp_path, "completo")
    with pytest.raises(ValueError):
        flujo.preparar_modo(tmp_path, "otro")


def test_pasos_completos_y_en_orden():
    nombres = [paso.nombre for paso in flujo.PASOS]
    assert nombres.index("analisis_ramas") > nombres.index("m2_atencion")
    assert nombres.index("fusion") > nombres.index("imagen_v0")
    assert nombres[-1] == "consolidar"
    for paso in flujo.PASOS:
        assert "--protocolo" in paso.argumentos("config/protocolo.json")


def test_celdas_validas_y_sin_codigo_dinamico():
    constructor = _constructor()
    paquete, manifiesto = constructor.empaquetar(dict(ARCHIVOS), "abc1234")
    fuente = (RAIZ_REPO / "src" / "cuaderno_demo" / "extraccion.py").read_text(encoding="utf-8")
    celdas = constructor.celdas(paquete, manifiesto, fuente)
    notebook = nbformat.v4.new_notebook(cells=celdas)
    nbformat.validate(notebook)
    codigo = "\n".join(c.source for c in celdas if c.cell_type == "code")
    assert "MODO = " in celdas[1].source
    for prohibido in ("eval(", "exec(", "shell=True", "allow_pickle=True"):
        assert prohibido not in codigo
    oculta = [c for c in celdas if c.cell_type == "code" and "PAQUETE_BASE64 =" in c.source]
    assert len(oculta) == 1 and oculta[0].metadata["cellView"] == "form"
