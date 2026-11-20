"""Pruebas de la base común (spec 001) con tablas pequeñas construidas en memoria. Sin red."""
import json

import numpy as np
import pandas as pd
import pytest

from comun import construir_ejemplos as ce
from comun import descargar_koi, lineas_base_tabulares, utilidades, verificar_contrato
from comun.protocolo import cargar_protocolo
from temporal import descargar_mast

PROTOCOLO = cargar_protocolo("config/protocolo.json")


def tabla_dr25(n=60, semilla=0):
    rng = np.random.default_rng(semilla)
    disp = np.array(["CONFIRMED", "FALSE POSITIVE"])[np.arange(n) % 2]
    es_fp = disp == "FALSE POSITIVE"
    df = pd.DataFrame({
        "kepid": 1_000_000 + np.arange(n),
        "kepoi_name": [f"K{i + 1:05d}.01" for i in range(n)],
        "koi_disposition": disp, "koi_pdisposition": disp, "koi_score": np.where(es_fp, 0.0, 1.0),
        "koi_fpflag_nt": 0, "koi_fpflag_ss": (es_fp & (np.arange(n) % 4 == 1)).astype(int),
        "koi_fpflag_co": (es_fp & (np.arange(n) % 4 == 3)).astype(int), "koi_fpflag_ec": 0,
        "koi_period": rng.uniform(1, 30, n), "koi_time0bk": rng.uniform(130, 160, n),
        "koi_duration": rng.uniform(1.5, 6, n), "koi_depth": np.where(es_fp, 5000.0, 800.0) + rng.normal(0, 50, n),
        "koi_model_snr": rng.uniform(20, 300, n), "koi_max_mult_ev": rng.uniform(10, 500, n),
        "koi_num_transits": rng.integers(40, 900, n), "koi_kepmag": rng.uniform(11, 15.5, n),
        "koi_dikco_msky": np.where(es_fp, 2.0, 0.1) + rng.normal(0, 0.05, n),
        "koi_dicco_msky": np.where(es_fp, 2.0, 0.1) + rng.normal(0, 0.05, n),
    })
    df.loc[3, "koi_dikco_msky"] = np.nan
    return df


def tablas():
    dr25 = tabla_dr25()
    seleccion = dr25[["kepid", "kepoi_name", "koi_disposition"]].iloc[:50].copy()
    acumulada = dr25[["kepid", "kepoi_name", "koi_disposition"]].copy()
    return seleccion, dr25, acumulada


# ---------- utilidades ----------

@pytest.mark.parametrize("malo", ["12; drop", "abc", -5, 0, 10**9, 3.7, None, True, "../1"])
def test_validar_kepid_rechaza(malo):
    with pytest.raises(ValueError):
        utilidades.validar_kepid(malo)


def test_validar_kepid_acepta():
    assert utilidades.validar_kepid("7440748") == 7440748
    assert utilidades.validar_kepid(np.int64(757450)) == 757450


def test_ruta_dentro_de(tmp_path):
    assert utilidades.ruta_dentro_de(tmp_path, "a/b.fits").parent == tmp_path / "a"
    with pytest.raises(ValueError):
        utilidades.ruta_dentro_de(tmp_path, "../fuera.fits")


@pytest.mark.parametrize("url", [
    "http://archive.stsci.edu/x", "https://ejemplo.com/x", "https://archive.stsci.edu.malo.com/x", "file:///etc/passwd"])
def test_url_no_permitida(url):
    with pytest.raises(ValueError):
        utilidades.comprobar_url(url)


def test_manifiesto_y_hash(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hola", encoding="utf-8")
    m = utilidades.escribir_manifiesto(tmp_path / "m.json", {"x": 1}, entradas=[f])
    assert list(m["entradas"].values()) == [utilidades.sha256_archivo(f)]
    assert json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))["argumentos"] == {"x": "1"}


# ---------- descarga (sin red) ----------

def test_consulta_tap_es_de_lista_cerrada():
    q = descargar_koi.construir_consulta("q1_q17_dr25_koi")
    assert q.startswith("select kepid,kepoi_name,koi_disposition") and q.endswith("from q1_q17_dr25_koi")
    assert descargar_koi.construir_url("cumulative").startswith("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?")
    with pytest.raises(ValueError):
        descargar_koi.construir_consulta("cumulative; drop table x")


def test_columnas_faltantes_fallan():
    df = tabla_dr25(1200).drop(columns=["koi_model_snr"])
    with pytest.raises(KeyError, match="koi_model_snr"):
        descargar_koi.comprobar_columnas(df, "q1_q17_dr25_koi")


def test_rutas_de_mast():
    assert descargar_mast.carpeta_remota("lightcurves", 7440748) == \
        "https://archive.stsci.edu/missions/kepler/lightcurves/0074/007440748/"
    assert descargar_mast.nombre_tpf_q5(757450) == "kplr000757450-2010174085026_lpd-targ.fits.gz"
    html = ('<a href="kplr007440748-2009166043257_llc.fits">x</a> <a href="kplr007440748-2009166043257_slc.fits">'
            '<a href="kplr007440748-2009131105131_llc.fits"> <a href="kplr000000001-2009131105131_llc.fits">')
    assert descargar_mast.archivos_llc(html, 7440748) == [
        "kplr007440748-2009131105131_llc.fits", "kplr007440748-2009166043257_llc.fits"]


# ---------- ejemplos ----------

def test_tipo_fp():
    base = dict(koi_disposition="FALSE POSITIVE", koi_fpflag_nt=0, koi_fpflag_ss=0, koi_fpflag_co=0, koi_fpflag_ec=0)
    assert ce.tipo_fp({**base, "koi_fpflag_ss": 1}) == "eclipse_secundario"
    assert ce.tipo_fp({**base, "koi_fpflag_co": 1}) == "centroide"
    assert ce.tipo_fp({**base, "koi_fpflag_nt": 1}) == "no_transito"
    assert ce.tipo_fp({**base, "koi_fpflag_ec": 1}) == "otro"
    assert ce.tipo_fp({**base, "koi_fpflag_ss": 1, "koi_fpflag_co": 1}) == "otro"
    assert ce.tipo_fp({**base, "koi_disposition": "CONFIRMED"}) == ""


def test_construir_ejemplos():
    seleccion, dr25, acumulada = tablas()
    acumulada.loc[acumulada.kepoi_name == "K00002.01", "koi_disposition"] = "CONFIRMED"  # cambió después de DR25
    dr25.loc[dr25.kepoi_name == "K00005.01", "koi_disposition"] = "CANDIDATE"            # sin etiqueta en DR25
    ej, resumen = ce.construir(seleccion, dr25, acumulada, PROTOCOLO)
    assert len(ej) == 2 * 49 and ej["id_ejemplo"].is_unique
    assert resumen["sin_etiqueta_en_dr25"] == ["K00005.01"]
    assert resumen["prerregistrados"] == ["K00002.01"]
    assert ej.loc[ej.kepoi_name == "K00002.01", "prerregistrado"].all()
    # las dos filas de un KOI comparten partición y la clase 2 es exactamente 'sin_senal'
    assert (ej.groupby("kepoi_name")["particion"].nunique() == 1).all()
    assert ((ej["clase"] == 2) == (ej["tipo"] == "sin_senal")).all()
    # balance por partición
    cuenta = ej[ej.tipo == "transito"].groupby(["particion", "clase"]).size().unstack()
    assert (cuenta.max() - cuenta.min()).max() <= 1
    assert ej.groupby("particion").size().max() - ej.groupby("particion").size().min() <= 2
    assert verificar_contrato.verificar_ejemplos(ej, PROTOCOLO) == []


def test_construir_es_determinista():
    a, _ = ce.construir(*tablas(), PROTOCOLO)
    b, _ = ce.construir(*tablas(), PROTOCOLO)
    pd.testing.assert_frame_equal(a, b)


def test_demasiados_prerregistrados_detiene():
    seleccion, dr25, acumulada = tablas()
    acumulada.loc[:9, "koi_disposition"] = "CANDIDATE"
    with pytest.raises(RuntimeError, match="prerregistro"):
        ce.construir(seleccion, dr25, acumulada, PROTOCOLO)


# ---------- verificación del contrato ----------

def _repo(tmp_path):
    ej, resumen = ce.construir(*tablas(), PROTOCOLO)
    (tmp_path / "compartido").mkdir()
    (tmp_path / "data").mkdir()
    ruta = tmp_path / "compartido" / "ejemplos.csv"
    ej.to_csv(ruta, index=False, lineterminator="\n")
    utilidades.escribir_manifiesto(tmp_path / "compartido" / "manifiesto_base.json", {},
                                   extra={"sha256_ejemplos": utilidades.sha256_archivo(ruta)})
    return ej


def _dataset(ej, nombres=None, **cambios):
    n = len(ej)
    d = {c: np.zeros((n, *f), dtype=np.float32) for c, f in PROTOCOLO["formas"].items() if c != "img"}
    d["id_ejemplo"] = np.array(ej["id_ejemplo"].tolist(), dtype="U")
    d["nombres_escalares"] = np.array(nombres or PROTOCOLO["escalares_permitidos"])
    d.update(cambios)
    return d


def test_contrato_correcto(tmp_path):
    ej = _repo(tmp_path)
    np.savez(tmp_path / "data" / "dataset_temporal.npz", **_dataset(ej))
    fallos, hechas = verificar_contrato.verificar(tmp_path, PROTOCOLO)
    assert fallos == [] and "dataset_temporal.npz" in hechas


def test_contrato_detecta_fuga_en_escalares(tmp_path):
    ej = _repo(tmp_path)
    nombres = PROTOCOLO["escalares_permitidos"][:-1] + ["koi_score"]
    np.savez(tmp_path / "data" / "dataset_temporal.npz", **_dataset(ej, nombres))
    fallos, _ = verificar_contrato.verificar(tmp_path, PROTOCOLO)
    assert any("FUGA" in f for f in fallos)


def test_contrato_detecta_nan_forma_y_manifiesto(tmp_path):
    ej = _repo(tmp_path)
    malo = np.zeros((len(ej), 32, 61), np.float32)
    malo[0, 0, 0] = np.nan
    np.savez(tmp_path / "data" / "dataset_temporal.npz",
             **_dataset(ej, transitos=malo, vista_global=np.zeros((len(ej), 100, 1), np.float32)))
    with open(tmp_path / "compartido" / "ejemplos.csv", "a", encoding="utf-8") as f:
        f.write("\n")
    fallos, _ = verificar_contrato.verificar(tmp_path, PROTOCOLO)
    texto = " | ".join(fallos)
    assert "NaN" in texto and "vista_global tiene forma" in texto and "SHA-256" in texto


def test_contrato_detecta_estrella_en_dos_particiones():
    ej, _ = ce.construir(*tablas(), PROTOCOLO)
    ej.loc[ej.index[0], "particion"] = (ej.loc[ej.index[0], "particion"] + 2) % 5
    assert any("más de una partición" in f or "Fuga" in f for f in verificar_contrato.verificar_ejemplos(ej, PROTOCOLO))


def test_contrato_rama(tmp_path):
    ej = _repo(tmp_path)
    carpeta = tmp_path / "modelos" / "escalares" / "v1"
    carpeta.mkdir(parents=True)
    (carpeta / "manifiesto.json").write_text("{}", encoding="utf-8")
    for k in range(5):
        (carpeta / f"particion_{k}.keras").write_bytes(b"x")
        np.savez(carpeta / f"vectores_particion_{k}.npz", id_ejemplo=np.array(ej["id_ejemplo"].tolist(), dtype="U"),
                 z=np.zeros((len(ej), 16 if k else 8), np.float32))
    fallos, _ = verificar_contrato.verificar(tmp_path, PROTOCOLO)
    assert len(fallos) == 1 and "vectores_particion_0" in fallos[0]


# ---------- líneas base ----------

def test_lineas_base_sin_fuga_y_con_senal():
    dr25 = tabla_dr25(240)
    seleccion = dr25[["kepid", "kepoi_name", "koi_disposition"]].iloc[:200]
    ej, _ = ce.construir(seleccion, dr25, dr25[["kepid", "kepoi_name", "koi_disposition"]], PROTOCOLO)
    sel = ej[ej.tipo == "transito"][["kepoi_name", "kepid", "clase", "particion"]].merge(
        dr25.drop(columns="kepid"), on="kepoi_name").reset_index(drop=True)
    with pytest.raises(ValueError, match="FUGA"):
        lineas_base_tabulares.predecir_oof(sel, lineas_base_tabulares.S3B + ["koi_score"], PROTOCOLO)
    p = lineas_base_tabulares.predecir_oof(sel, lineas_base_tabulares.S3C, PROTOCOLO)
    assert np.isfinite(p).all()
    r = lineas_base_tabulares.evaluar(p, sel, {**PROTOCOLO, "metricas": {**PROTOCOLO["metricas"], "bootstrap_remuestreos": 100}})
    assert r["auc"] > 0.9 and r["ic_inf"] <= r["auc"] <= r["ic_sup"]


def test_particiones_estables_y_por_estrella():
    kois = pd.DataFrame({"kepid": [5, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
                         "kepoi_name": [f"K{i:05d}.01" for i in range(11)],
                         "clase": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]})
    p = ce.asignar_particiones(kois, 5, 0)
    assert p[0] == p[1]                                   # la estrella con dos KOIs no se parte
    assert list(p) == list(ce.asignar_particiones(kois.sample(frac=1, random_state=3), 5, 0)[
        np.argsort(kois.sample(frac=1, random_state=3).index)])   # no depende del orden de las filas
    assert list(p) != list(ce.asignar_particiones(kois, 5, 1))    # sí depende de la semilla
    # valor fijo: si esto cambia, cambiaron las particiones del proyecto
    assert list(ce.asignar_particiones(pd.DataFrame({"kepid": [1, 2, 3, 4, 5, 6], "kepoi_name": list("abcdef"),
                                                     "clase": [0] * 6}), 5, 0)) == [0, 2, 1, 0, 4, 3]
