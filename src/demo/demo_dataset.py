"""Construye el dataset de la demo desde TPFs reales de Kepler (un trimestre por estrella).

Por cada KOI se generan dos ejemplos:
  - el real (clase PLANETA o FALSO POSITIVO según koi_disposition), centrado en su época;
  - un NEGATIVO (sin señal): la misma estrella, con una época falsa desplazada 0.25 en fase,
    donde no hay tránsito (prueba nula del plan).

Entradas del modelo por ejemplo:
  img  (11, 11, 4): SNR de la imagen de diferencia apilada, imagen fuera de tránsito,
                    posición del objetivo (gaussiana), máscara de píxeles válidos
  seq  (61, 2):     vista local (±2 duraciones) y vista en fase 0.5 (eclipse secundario),
                    ambas en unidades de SNR por punto

Uso: python -I demo_dataset.py <carpeta_tpf> <seleccion.csv> <salida.npz>
"""

import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
import lightkurve as lk

warnings.filterwarnings("ignore")
CARP, SEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
N = 11
NB = 61
CLASES = {"CONFIRMED": 0, "FALSE POSITIVE": 1}  # 2 = negativo (sin señal)


def imagen_diferencia(t, cubo, epocas, dur):
    """Apila la imagen de diferencia de todos los tránsitos (ponderación por varianza)."""
    num = np.zeros(cubo.shape[1:])
    den = np.zeros(cubo.shape[1:])
    usados = 0
    for tc in epocas:
        dt = t - tc
        dentro = np.abs(dt) < 0.35 * dur
        fuera = (np.abs(dt) > 0.75 * dur) & (np.abs(dt) < 0.75 * dur + max(1.0, 1.5 * dur))
        if dentro.sum() < 2 or fuera.sum() < 10:
            continue
        X = np.vstack([np.ones(fuera.sum()), dt[fuera]]).T
        Y = np.nan_to_num(cubo[fuera].reshape(fuera.sum(), -1))
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        sig2 = (Y - X @ coef).var(axis=0, ddof=2)
        modelo = np.vstack([np.ones(dentro.sum()), dt[dentro]]).T @ coef
        Yd = np.nan_to_num(cubo[dentro].reshape(dentro.sum(), -1))
        d = (modelo - Yd).mean(axis=0)
        var = sig2 * (1 / dentro.sum() + 1 / fuera.sum())
        ok = var > 0
        w = np.zeros_like(var)
        w[ok] = 1 / var[ok]
        num += (d * w).reshape(cubo.shape[1:])
        den += w.reshape(cubo.shape[1:])
        usados += 1
    snr = np.where(den > 0, num / np.sqrt(np.where(den > 0, den, 1)), 0.0)
    return snr, usados


def vista(fase, flujo, centro, dur, sigma):
    bordes = np.linspace(centro - 2 * dur, centro + 2 * dur, NB + 1)
    idx = np.digitize(fase, bordes) - 1
    v = np.array([np.nanmedian(flujo[idx == i]) if np.any(idx == i) else np.nan for i in range(NB)])
    ok = np.isfinite(v)
    if ok.sum() < NB // 3:
        return None
    v = np.interp(np.arange(NB), np.flatnonzero(ok), v[ok])
    cnt = np.array([max((idx == i).sum(), 1) for i in range(NB)])
    return np.clip((v - 1.0) / (sigma / np.sqrt(cnt)), -30, 30) / 10.0


def recortar(mapa, yo, xo, relleno=0.0):
    """Recorta/rellena a N×N con el píxel objetivo en el centro."""
    out = np.full((N, N), relleno, dtype=float)
    c = N // 2
    H, W = mapa.shape
    for y in range(H):
        for x in range(W):
            yy, xx = y - yo + c, x - xo + c
            if 0 <= yy < N and 0 <= xx < N:
                out[yy, xx] = mapa[y, x]
    return out


def ejemplo(tpf, lc_flat, P, t0, dur, ra, dec):
    t = tpf.time.value
    cubo = tpf.flux.value.astype(float)
    buenos = np.isfinite(t) & np.isfinite(cubo).any(axis=(1, 2))
    t, cubo = t[buenos], cubo[buenos]
    n0, n1 = np.ceil((t.min() - t0) / P), np.floor((t.max() - t0) / P)
    epocas = [t0 + n * P for n in range(int(n0), int(n1) + 1)]
    snr, usados = imagen_diferencia(t, cubo, epocas, dur)
    if usados < 2:
        return None
    fuera_img = np.nanmedian(cubo, axis=0)
    valid = np.isfinite(cubo).any(axis=0).astype(float)
    fuera_img = np.nan_to_num(fuera_img) / max(np.nanmax(fuera_img), 1e-9)
    xo_f, yo_f = tpf.wcs.world_to_pixel_values(ra, dec)
    xo, yo = int(round(float(xo_f))), int(round(float(yo_f)))
    sx, sy = float(xo_f) - xo, float(yo_f) - yo
    yy, xx = np.mgrid[0:N, 0:N]
    gauss = np.exp(-(((xx - N // 2 - sx) ** 2) + ((yy - N // 2 - sy) ** 2)) / (2 * 0.7**2))
    img = np.stack(
        [
            np.clip(recortar(snr, yo, xo), -30, 30) / 10.0,
            recortar(fuera_img, yo, xo),
            gauss,
            recortar(valid, yo, xo),
        ],
        axis=-1,
    )
    # vistas 1D desde la curva de apertura aplanada
    tl, fl = lc_flat.time.value, lc_flat.flux.value
    fase = ((tl - t0 + P / 4) % P) - P / 4  # fase 0 en el tránsito, 0.5P cae dentro del rango
    fuera = np.abs(((tl - t0 + P / 2) % P) - P / 2) > dur
    sigma = np.nanstd(fl[fuera])
    v1 = vista(fase, fl, 0.0, dur, sigma)
    v2 = vista(fase, fl, P / 2, dur, sigma)
    if v1 is None or v2 is None:
        return None
    return img.astype(np.float32), np.stack([v1, v2], axis=-1).astype(np.float32), usados


sel = pd.read_csv(SEL)
IMG, SEQ, Y, META = [], [], [], []
for fila in sel.itertuples():
    fs = glob.glob(os.path.join(CARP, f"demo_kplr{fila.kepid:09d}-*_lpd-targ.fits.gz"))
    if not fs:
        continue
    try:
        tpf = lk.read(fs[0], quality_bitmask="default")
        P, t0, dur = fila.koi_period, fila.koi_time0bk, fila.koi_duration / 24
        lc = tpf.to_lightcurve(aperture_mask="pipeline").remove_nans().normalize()
        mask = lc.create_transit_mask(period=P, transit_time=t0, duration=dur * 1.5)
        flat = lc.flatten(window_length=401, mask=mask).remove_outliers(
            sigma_upper=4, sigma_lower=50
        )
        real = ejemplo(tpf, flat, P, t0, dur, tpf.ra, tpf.dec)
        falso = ejemplo(tpf, flat, P, t0 + 0.25 * P, dur, tpf.ra, tpf.dec)  # época sin tránsito
        if real is None or falso is None:
            continue
        for (img, seq, n), y, tipo in [
            (real, CLASES[fila.koi_disposition], "real"),
            (falso, 2, "nulo"),
        ]:
            IMG.append(img)
            SEQ.append(seq)
            Y.append(y)
            META.append(
                dict(
                    kepid=fila.kepid,
                    koi=fila.kepoi_name,
                    disp=fila.koi_disposition,
                    tipo=tipo,
                    n_transitos=n,
                    periodo=P,
                    mes=fila.koi_max_mult_ev,
                    kp=fila.koi_kepmag,
                    fp_nt=fila.koi_fpflag_nt,
                    fp_ss=fila.koi_fpflag_ss,
                    fp_co=fila.koi_fpflag_co,
                    dikco=fila.koi_dikco_msky,
                    dicco=fila.koi_dicco_msky,
                    trimestre=os.path.basename(fs[0])[19:32],
                )
            )
        print("OK", fila.kepoi_name, fila.koi_disposition, real[2], "tránsitos", flush=True)
    except Exception as e:
        print("FAIL", fila.kepoi_name, repr(e)[:80], flush=True)

np.savez_compressed(OUT, img=np.array(IMG), seq=np.array(SEQ), y=np.array(Y))
pd.DataFrame(META).to_csv(OUT.replace(".npz", "_meta.csv"), index=False)
print("total", len(Y), "por clase", np.bincount(np.array(Y)))
