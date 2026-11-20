"""Pruebas del extractor temporal (spec 002) con curvas sintéticas."""

import numpy as np
import pytest

from comun.protocolo import cargar_protocolo
from temporal import curvas
from temporal.curvas import Efemerides
from tests import sinteticos

PROTOCOLO = cargar_protocolo("config/protocolo.json")
CFG = PROTOCOLO["extractor"]
EF = Efemerides(periodo=5.0, t0=133.0, duracion=4.0 / 24)


def ejemplo(ef=EF, otras=(), efs_estrella=None, epoca=None, **kw):
    trimestres = sinteticos.curva(ef, **kw)
    t, f, s = curvas.preparar_curva(trimestres, list(efs_estrella or [ef]), CFG)
    return curvas.construir_ejemplo(t, f, s, epoca or ef, list(otras), CFG), (t, f, s)


def test_aplanado_quita_la_tendencia_y_conserva_el_transito():
    _, (t, f, s) = ejemplo()
    fuera = ~curvas.mascara_transitos(t, [EF])
    assert abs(np.median(f[fuera]) - 1) < 2e-5
    assert curvas.ruido_robusto(f[fuera]) == pytest.approx(200e-6, rel=0.15)
    prof, err, n = curvas.medir_profundidad(t, f, s, EF)
    assert prof == pytest.approx(1000e-6, rel=0.10) and n > 500 and err < 2e-5


def test_formas_y_escala():
    ej, _ = ejemplo()
    for clave, forma in PROTOCOLO["formas"].items():
        if clave != "img":
            assert ej[clave].shape == tuple(forma), clave
            assert np.all(np.isfinite(ej[clave])), clave
    assert ej["disponible_vistas"] == 1 and ej["disponible_transitos"] == 1
    centro = ej["vista_local"][95:106, 0]
    assert -1.2 < np.median(centro) < -0.8  # un tránsito real vale cerca de −1
    assert abs(np.median(ej["vista_local"][:40, 0])) < 0.1  # fuera del tránsito, cerca de 0
    assert abs(np.median(ej["vista_local"][95:106, 3])) < 0.15  # sin eclipse secundario
    assert ej["escalares"][2] == pytest.approx(1000, rel=0.10) and ej["faltantes"][:5].sum() == 0
    assert ej["faltantes"][5:].sum() == 2  # centroide no dado → faltante, con aviso


def test_planeta_es_consistente():
    ej, _ = ejemplo()
    assert abs(ej["consistencia"][1]) < 0.6  # |par − impar| < 3 sigmas (escala logarítmica)
    assert ej["mascara_transitos"].sum() == 32
    par = ej["paridad_transitos"]
    assert (par == 0).sum() == 16 and (par == 1).sum() == 16


def test_estrella_doble_alterna():
    ej, _ = ejemplo(profundidad=600e-6, profundidad_impar=1400e-6)
    assert ej["consistencia"][1] > 1.0  # más de 9 sigmas
    local = ej["vista_local"]
    assert (
        np.median(local[95:106, 2]) < np.median(local[95:106, 1]) - 0.4
    )  # impares más profundos que pares
    prof = -ej["transitos"][:, 27:34].mean(axis=1)
    assert (
        prof[ej["paridad_transitos"] == 1].mean() > prof[ej["paridad_transitos"] == 0].mean() + 0.4
    )


def test_eclipse_secundario_aparece_en_el_canal_de_fase_media():
    ej, _ = ejemplo(profundidad=2000e-6, secundario=800e-6)
    assert np.median(ej["vista_local"][95:106, 3]) < -0.2
    assert (
        ej["vista_global"][:20, 0].mean() < -0.1
    )  # el secundario cae en los extremos de la vista global


def test_hueco_invalida_el_transito():
    tc = EF.t0 + 20 * EF.periodo
    trimestres = sinteticos.curva(EF, borrar=(tc - 0.1, tc + 0.5))
    t, f, s = curvas.preparar_curva(trimestres, [EF], CFG)
    v = {x.indice: x for x in curvas.ventanas_por_transito(t, f, s, EF, [], CFG)}
    assert not v[20].valida and v[20].motivo == "cadencias"
    assert v[19].valida and v[21].valida


def test_otro_koi_invalida_los_transitos_que_coinciden():
    otro = Efemerides(periodo=15.0, t0=EF.t0, duracion=3.0 / 24)  # coincide cada 3 tránsitos
    ej, (t, f, s) = ejemplo(otras=[otro], efs_estrella=[EF, otro], otro=(otro, 3000e-6))
    v = curvas.ventanas_por_transito(t, f, s, EF, [otro], CFG)
    assert all(
        (not x.valida and x.motivo in ("otro_koi", "cadencias")) for x in v if x.indice % 3 == 0
    )
    assert all(x.valida for x in v if x.indice % 3 == 1 and x.motivo != "cadencias")
    assert ej["descartes"]["otro_koi"] > 50
    assert ej["escalares"][2] == pytest.approx(1000, rel=0.35)


def test_sin_senal():
    desplazada = Efemerides(EF.periodo, EF.t0 + CFG["desfase_sin_senal"] * EF.periodo, EF.duracion)
    ej, (t, f, s) = ejemplo(epoca=desplazada, otras=[EF])
    assert ej["escalares"][3] < 3 and ej["aviso_sin_senal"] == 0  # SNR medido bajo
    assert abs(ej["vista_local"][95:106, 0].mean()) < 0.5  # el ruido no se infla a −1
    assert np.abs(ej["vista_local"]).max() <= CFG["recorte_relativo"]
    # la señal real queda oculta: ninguna ventana válida usa cadencias del tránsito real
    assert ej["disponible_transitos"] == 1 and ej["mascara_transitos"].sum() == 32
    assert np.abs(np.median(ej["transitos"][:, 20:41])) < 0.5


def test_sin_senal_con_duracion_larga_respecto_al_periodo():
    # Caso real (20-nov): con duración/periodo > 1/13 la ventana desplazada toca el tránsito real.
    ef = Efemerides(periodo=1.3, t0=133.0, duracion=4.0 / 24)
    desplazada = Efemerides(ef.periodo, ef.t0 + 0.25 * ef.periodo, ef.duracion)
    ej, _ = ejemplo(ef=ef, epoca=desplazada, otras=[ef], profundidad=5000e-6)
    assert ej["disponible_transitos"] == 1 and ej["disponible_vistas"] == 1
    # sin rastro de la señal real: con 5000 ppm, cualquier resto saturaría el recorte en −5
    assert ej["escalares"][3] < 5 and (ej["vista_local"][:, 0] <= -4.99).sum() == 0
    assert (ej["transitos"] <= -4.99).mean() < 0.01


def test_companero_de_periodo_muy_corto_no_anula_la_rama():
    # Caso real K00191.01: un compañero con periodo de 0,7 d cae dentro de todas las ventanas.
    otro = Efemerides(periodo=0.7, t0=133.2, duracion=1.5 / 24)
    ef = Efemerides(periodo=15.36, t0=133.0, duracion=4.1 / 24)
    ej, _ = ejemplo(ef=ef, otras=[otro], efs_estrella=[ef, otro], otro=(otro, 300e-6))
    assert ej["disponible_transitos"] == 1 and ej["n_transitos_validos"] > 50
    assert ej["escalares"][2] == pytest.approx(1000, rel=0.15)


def test_seleccion_es_determinista_y_cronologica():
    _, (t, f, s) = ejemplo()
    v = curvas.ventanas_por_transito(t, f, s, EF, [], CFG)
    a, b = curvas.seleccionar_transitos(v, 32), curvas.seleccionar_transitos(v, 32)
    assert [x.indice for x in a] == [x.indice for x in b] == sorted(x.indice for x in a)
    assert len(a) == 32 and a[-1].indice - a[0].indice > 200  # repartidos en toda la misión
    pocos = [x for x in v if x.valida][:5]
    assert len(curvas.seleccionar_transitos(pocos, 32)) == 5


def test_curva_vacia_o_pocos_transitos_no_revientan():
    vacio = curvas.construir_ejemplo(np.empty(0), np.empty(0), np.empty(0), EF, [], CFG)
    assert (
        vacio["disponible_vistas"] == 0
        and vacio["disponible_transitos"] == 0
        and vacio["faltantes"].all()
    )
    largo = Efemerides(periodo=700.0, t0=300.0, duracion=8.0 / 24)  # solo 2 tránsitos en la misión
    ej, _ = ejemplo(ef=largo)
    assert ej["disponible_transitos"] == 0 and ej["mascara_transitos"].sum() == 0


def test_senal_profunda_no_se_satura():
    # Caso K00022.01: con la normalización por forma, una señal de 1 % sigue valiendo −1.
    ej, _ = ejemplo(profundidad=10000e-6, ruido=100e-6)
    assert -1.1 < np.median(ej["vista_local"][95:106, 0]) < -0.9
    assert ej["escalares"][3] > 1000


def test_dataset_completo_desde_fits(tmp_path):
    """Corrida completa con FITS sintéticos: lectura, extractor, guardado y verificación del contrato."""
    fits = pytest.importorskip("astropy.io.fits")
    import pandas as pd

    from comun import construir_ejemplos as ce
    from comun import utilidades, verificar_contrato
    from temporal import construir_dataset_temporal as cdt

    n = 10
    dr25 = pd.DataFrame(
        {
            "kepid": 2_000_000 + np.arange(n),
            "kepoi_name": [f"K{i + 1:05d}.01" for i in range(n)],
            "koi_disposition": ["CONFIRMED", "FALSE POSITIVE"] * (n // 2),
            "koi_fpflag_nt": 0,
            "koi_fpflag_ss": [0, 1] * (n // 2),
            "koi_fpflag_co": 0,
            "koi_fpflag_ec": 0,
            "koi_period": 3.0 + np.arange(n),
            "koi_time0bk": 133.0 + 0.3 * np.arange(n),
            "koi_duration": 4.0,
            "koi_max_mult_ev": 50.0 + np.arange(n),
            "koi_dikco_msky": [0.1, np.nan] * (n // 2),
            "koi_dicco_msky": 0.2,
        }
    )
    ejemplos, _ = ce.construir(
        dr25[["kepoi_name", "koi_disposition"]],
        dr25,
        dr25[["kepid", "kepoi_name", "koi_disposition"]],
        PROTOCOLO,
    )
    for k in dr25.itertuples():
        ef = Efemerides(k.koi_period, k.koi_time0bk, k.koi_duration / 24)
        impar = 1600e-6 if k.koi_disposition == "FALSE POSITIVE" else None
        carpeta = tmp_path / "data" / "llc" / f"{k.kepid:09d}"
        carpeta.mkdir(parents=True)
        for q, (t, f) in enumerate(
            sinteticos.curva(ef, semilla=k.kepid, n_trimestres=6, profundidad_impar=impar)
        ):
            calidad = np.zeros(len(t), dtype=np.int32)
            calidad[:5] = 16  # cadencias marcadas: deben descartarse
            f = f.copy()
            f[10] = np.nan
            tabla = fits.BinTableHDU.from_columns(
                [
                    fits.Column("TIME", "D", array=t),
                    fits.Column("PDCSAP_FLUX", "E", array=f),
                    fits.Column("SAP_QUALITY", "J", array=calidad),
                ]
            )
            fits.HDUList([fits.PrimaryHDU(), tabla]).writeto(
                carpeta / f"kplr{k.kepid:09d}-20090000000{q:02d}_llc.fits"
            )

    datos, registro = cdt.construir(ejemplos, dr25, tmp_path / "data" / "llc", PROTOCOLO)
    assert len(datos["id_ejemplo"]) == 2 * n and datos["id_ejemplo"].dtype.kind == "U"
    assert registro["disponible_vistas"].all() and registro["disponible_transitos"].all()
    y = ejemplos.set_index("id_ejemplo").loc[datos["id_ejemplo"], "clase"].values
    diferencia = datos["consistencia"][:, 1]
    assert (
        diferencia[y == 1].min() > 1.0 and np.abs(diferencia[y == 0]).max() < 0.8
    )  # la doble alterna; el planeta no
    assert datos["escalares"][y == 2, 3].max() < 5  # sin señal: SNR bajo
    assert datos["faltantes"][y == 2, 5:].all()  # sin señal: centroide faltante
    assert (
        datos["faltantes"][y == 1, 5].all() and not datos["faltantes"][y == 0, 5].any()
    )  # NaN del catálogo → aviso

    (tmp_path / "compartido").mkdir()
    ruta = tmp_path / "compartido" / "ejemplos.csv"
    ejemplos.to_csv(ruta, index=False, lineterminator="\n")
    utilidades.escribir_manifiesto(
        tmp_path / "compartido" / "manifiesto_base.json",
        {},
        extra={"sha256_ejemplos": utilidades.sha256_archivo(ruta)},
    )
    np.savez_compressed(tmp_path / "data" / "dataset_temporal.npz", **datos)
    fallos, hechas = verificar_contrato.verificar(tmp_path, PROTOCOLO)
    assert fallos == [] and "dataset_temporal.npz" in hechas
