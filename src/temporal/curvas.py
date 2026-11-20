"""Extractor temporal: de la curva de luz a las entradas de las ramas 1, 2 y 4 (spec 002).

Funciones puras sobre arreglos numpy (sin lightkurve), para poder probarlas con curvas sintéticas.
Todos los tiempos están en días (BKJD); el flujo llega normalizado alrededor de 1.

Decisión DD-04: las series se expresan como (flujo − 1) / escala, donde escala es la profundidad
medida del ejemplo (con un piso de 3 sigmas). Un tránsito real vale cerca de −1, y la diferencia
entre eclipses pares e impares no se pierde por saturación.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

CADENCIA_D = 0.0204340  # cadencia larga de Kepler, 29,4 minutos


@dataclass(frozen=True)
class Efemerides:
    periodo: float   # días
    t0: float        # época del primer tránsito, BKJD
    duracion: float  # días

    def __post_init__(self):
        if not (self.periodo > 0 and self.duracion > 0 and np.isfinite(self.t0)):
            raise ValueError(f"Efemérides inválidas: {self}")


@dataclass(frozen=True)
class Ventana:
    indice: int          # número de tránsito respecto a t0
    centro: float
    serie: np.ndarray    # (n_puntos,), flujo − 1 sin escalar
    profundidad: float
    error: float
    valida: bool
    motivo: str          # "", "cadencias", "en_transito" u "otro_koi"


def fase(t: np.ndarray, ef: Efemerides, centro: float = 0.0) -> np.ndarray:
    """Tiempo respecto al tránsito más cercano (o respecto a `centro` días después), en [−P/2, P/2)."""
    return ((t - ef.t0 - centro + ef.periodo / 2) % ef.periodo) - ef.periodo / 2


def numero_de_transito(t: np.ndarray, ef: Efemerides) -> np.ndarray:
    return np.round((t - ef.t0) / ef.periodo).astype(int)


def mascara_transitos(t: np.ndarray, efemerides: list[Efemerides], factor: float = 1.5) -> np.ndarray:
    """Verdadero en las cadencias dentro de `factor` duraciones (ancho total) de cualquier tránsito."""
    m = np.zeros(len(t), dtype=bool)
    for ef in efemerides:
        m |= np.abs(fase(t, ef)) < factor * ef.duracion / 2
    return m


def aplanar(t: np.ndarray, f: np.ndarray, enmascarado: np.ndarray, ventana: int = 401,
            hueco_d: float = 0.5) -> np.ndarray:
    """Divide por una tendencia suave (Savitzky–Golay) ajustada sin los puntos enmascarados.

    Se trabaja por tramos sin huecos mayores de `hueco_d` días. Sobre los puntos enmascarados la
    tendencia se interpola, de modo que el tránsito no se absorbe.
    """
    salida = np.array(f, dtype=float)
    cortes = np.flatnonzero(np.diff(t) > hueco_d) + 1
    for tramo in np.split(np.arange(len(t)), cortes):
        if len(tramo) == 0:
            continue
        libre = ~enmascarado[tramo]
        if libre.sum() < 15:
            tendencia = np.full(len(tramo), np.median(f[tramo][libre]) if libre.any() else np.median(f[tramo]))
        else:
            relleno = np.interp(t[tramo], t[tramo][libre], f[tramo][libre])
            w = min(ventana, len(tramo) if len(tramo) % 2 else len(tramo) - 1)
            tendencia = savgol_filter(relleno, w, 2) if w >= 5 else np.full(len(tramo), np.median(relleno))
        salida[tramo] = f[tramo] / tendencia
    return salida


def ruido_robusto(f: np.ndarray) -> float:
    """1,4826 · MAD."""
    return float(1.4826 * np.median(np.abs(f - np.median(f)))) if len(f) else float("nan")


def preparar_curva(trimestres: list[tuple[np.ndarray, np.ndarray]], efemerides: list[Efemerides], cfg: dict
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Limpia y une los trimestres. Devuelve tiempo, flujo aplanado y ruido por punto, ordenados.

    `efemerides` son las de TODOS los KOIs de la estrella: se enmascaran al aplanar (RF-4).
    """
    ts, fs, ss = [], [], []
    for t, f in trimestres:
        ok = np.isfinite(t) & np.isfinite(f)
        t, f = np.asarray(t, float)[ok], np.asarray(f, float)[ok]
        if len(t) < 50:
            continue
        orden = np.argsort(t)
        t, f = t[orden], f[orden] / np.median(f[orden])
        en_transito = mascara_transitos(t, efemerides, cfg["mascara_transito_duraciones"])
        plano = aplanar(t, f, en_transito, cfg["ventana_aplanado_cadencias"])
        sigma = ruido_robusto(plano[~en_transito]) if (~en_transito).sum() > 20 else ruido_robusto(plano)
        if not (np.isfinite(sigma) and sigma > 0):
            continue
        conservar = plano < 1 + cfg["recorte_atipicos_sigma_superior"] * sigma  # solo hacia arriba
        ts.append(t[conservar])
        fs.append(plano[conservar])
        ss.append(np.full(conservar.sum(), sigma))
    if not ts:
        return np.empty(0), np.empty(0), np.empty(0)
    t, f, s = np.concatenate(ts), np.concatenate(fs), np.concatenate(ss)
    orden = np.argsort(t)
    return t[orden], f[orden], s[orden]


def medir_profundidad(t: np.ndarray, f: np.ndarray, sigma: np.ndarray, ef: Efemerides) -> tuple[float, float, int]:
    """Profundidad (fracción), su error y el número de puntos, con los puntos del centro del tránsito."""
    centro = np.abs(fase(t, ef)) < 0.25 * ef.duracion
    n = int(centro.sum())
    if n == 0:
        return float("nan"), float("nan"), 0
    return float(1.0 - np.median(f[centro])), float(np.median(sigma[centro]) / np.sqrt(n)), n


def vista_plegada(fase_d: np.ndarray, valor: np.ndarray, semiancho: float, n_puntos: int
                  ) -> tuple[np.ndarray, float]:
    """Mediana por intervalo en [−semiancho, semiancho]. Devuelve la serie y la fracción de vacíos."""
    dentro = np.abs(fase_d) <= semiancho
    idx = np.clip(((fase_d[dentro] + semiancho) / (2 * semiancho) * n_puntos).astype(int), 0, n_puntos - 1)
    v = valor[dentro]
    orden = np.argsort(idx, kind="mergesort")
    idx, v = idx[orden], v[orden]
    serie = np.full(n_puntos, np.nan)
    if len(idx):
        inicios = np.flatnonzero(np.r_[True, np.diff(idx) > 0])
        for a, b in zip(inicios, np.r_[inicios[1:], len(idx)]):
            serie[idx[a]] = np.median(v[a:b])
    lleno = np.isfinite(serie)
    vacios = 1.0 - lleno.mean()
    if lleno.sum() >= 2:
        serie = np.interp(np.arange(n_puntos), np.flatnonzero(lleno), serie[lleno])
    else:
        serie = np.zeros(n_puntos)
    return serie, float(vacios)


def ventanas_por_transito(t: np.ndarray, f: np.ndarray, sigma: np.ndarray, ef: Efemerides,
                          otras: list[Efemerides], cfg: dict) -> list[Ventana]:
    """Una ventana por tránsito observado, con su validez (RF-11 y RF-12).

    Las cadencias que caen en el tránsito de OTRO KOI de la estrella se ocultan (no se usan) y la
    ventana se interpola sobre el hueco. La ventana solo se descarta ("otro_koi") si, tras ocultar,
    queda menos de la fracción mínima visible o se pierde el centro del tránsito.
    """
    n_puntos = cfg["transito_puntos"]
    semi = min(cfg["transito_semiancho_duraciones"] * ef.duracion, ef.periodo / 2)
    rejilla = np.linspace(-semi, semi, n_puntos)
    esperadas = 2 * semi / CADENCIA_D
    if len(t) == 0:
        return []
    visible = ~mascara_transitos(t, otras, cfg["mascara_transito_duraciones"]) if otras else np.ones(len(t), bool)
    tv, fv, sv = t[visible], f[visible], sigma[visible]
    n_min, n_max = int(np.ceil((t[0] - ef.t0) / ef.periodo)), int(np.floor((t[-1] - ef.t0) / ef.periodo))
    ventanas = []
    for n in range(n_min, n_max + 1):
        tc = ef.t0 + n * ef.periodo
        a0, b0 = np.searchsorted(t, [tc - semi, tc + semi])      # todas las cadencias observadas
        a, b = np.searchsorted(tv, [tc - semi, tc + semi])       # solo las visibles
        dt, y = tv[a:b] - tc, fv[a:b] - 1.0
        en_centro_total = (np.abs(t[a0:b0] - tc) < 0.5 * ef.duracion).sum()
        en_centro = (np.abs(dt) < 0.5 * ef.duracion).sum()
        motivo = ""
        if (b0 - a0) < cfg["min_fraccion_cadencias"] * esperadas:
            motivo = "cadencias"
        elif en_centro_total < cfg["min_cadencias_en_transito"]:
            motivo = "en_transito"
        elif (b - a) < cfg["min_fraccion_visible"] * esperadas or en_centro < cfg["min_cadencias_en_transito"]:
            motivo = "otro_koi"
        if motivo:
            ventanas.append(Ventana(n, tc, np.zeros(n_puntos), float("nan"), float("nan"), False, motivo))
            continue
        nucleo = np.abs(dt) < 0.25 * ef.duracion
        if nucleo.sum() < 1:
            nucleo = np.abs(dt) < 0.5 * ef.duracion
        serie = np.interp(rejilla, dt, y)
        prof = float(-np.mean(y[nucleo]))
        err = float(np.mean(sv[a:b][nucleo]) / np.sqrt(nucleo.sum()))
        ventanas.append(Ventana(n, tc, serie, prof, err, True, ""))
    return ventanas


def _equiespaciados(n_total: int, k: int) -> np.ndarray:
    return np.unique(np.round(np.linspace(0, n_total - 1, k)).astype(int)) if k > 0 else np.empty(0, int)


def seleccionar_transitos(ventanas: list[Ventana], maximo: int = 32) -> list[Ventana]:
    """Hasta `maximo` tránsitos válidos, mitad pares y mitad impares, repartidos en el tiempo (RF-13)."""
    validas = [v for v in ventanas if v.valida]
    pares = [v for v in validas if v.indice % 2 == 0]
    impares = [v for v in validas if v.indice % 2 != 0]
    mitad = maximo // 2
    k_p, k_i = min(len(pares), mitad), min(len(impares), mitad)
    k_p = min(len(pares), maximo - k_i)
    k_i = min(len(impares), maximo - k_p)
    elegidas = [pares[i] for i in _equiespaciados(len(pares), k_p)] + \
               [impares[i] for i in _equiespaciados(len(impares), k_i)]
    return sorted(elegidas, key=lambda v: v.indice)


def _log_con_signo(x: float) -> float:
    return float(np.sign(x) * np.log10(1.0 + abs(x))) if np.isfinite(x) else 0.0


def estadisticos_consistencia(ventanas: list[Ventana]) -> np.ndarray:
    """Cuatro estadísticos de consistencia entre tránsitos (RF-15), con todos los válidos.

    0: dispersión robusta de la profundidad por tránsito, en unidades de su error típico
    1: (profundidad media impar − par) en unidades del error de esa diferencia
    2: log10(1 + número de tránsitos válidos)
    3: mayor desviación de un tránsito respecto a la mediana, en unidades de su error
    Los estadísticos 0, 1 y 3 se comprimen con sign(x)·log10(1 + |x|): en estrellas dobles
    brillantes valen miles de sigmas.
    """
    validas = [v for v in ventanas if v.valida and np.isfinite(v.error) and v.error > 0]
    if len(validas) < 2:
        return np.zeros(4, dtype=np.float32)
    d = np.array([v.profundidad for v in validas])
    e = np.array([v.error for v in validas])
    par = np.array([v.indice % 2 == 0 for v in validas])
    dispersion = ruido_robusto(d) / np.median(e)
    if par.any() and (~par).any():
        ee = np.sqrt(np.mean(e[par] ** 2) / par.sum() + np.mean(e[~par] ** 2) / (~par).sum())
        diferencia = (d[~par].mean() - d[par].mean()) / ee
    else:
        diferencia = 0.0
    maxima = np.max(np.abs(d - np.median(d)) / e)
    return np.array([_log_con_signo(dispersion), _log_con_signo(diferencia), np.log10(1 + len(validas)),
                     _log_con_signo(maxima)], dtype=np.float32)


def construir_ejemplo(t: np.ndarray, f: np.ndarray, sigma: np.ndarray, ef: Efemerides,
                      otras: list[Efemerides], cfg: dict, centroide: tuple[float, float] = (np.nan, np.nan)
                      ) -> dict:
    """Todas las entradas de un ejemplo (ramas 1, 2 y 4) a partir de la curva ya preparada.

    `otras` son los demás KOIs de la estrella: sus cadencias en tránsito se ocultan.
    Para el ejemplo 'sin_senal' se llama con la época desplazada, `otras` debe incluir la
    efeméride real del KOI (así la señal real queda oculta) y `centroide` se deja en NaN.
    """
    ng, nl, nt, mx = cfg["vista_global_puntos"], cfg["vista_local_puntos"], cfg["transito_puntos"], cfg["max_transitos"]
    recorte = cfg["recorte_relativo"]
    ej = {
        "vista_global": np.zeros((ng, 1), np.float32), "vista_local": np.zeros((nl, 4), np.float32),
        "transitos": np.zeros((mx, nt), np.float32), "mascara_transitos": np.zeros(mx, np.float32),
        "paridad_transitos": np.full(mx, -1, np.int8), "n_transitos_validos": 0,
        "consistencia": np.zeros(4, np.float32), "escalares": np.zeros(7, np.float32),
        "faltantes": np.ones(7, np.float32), "disponible_vistas": 0, "disponible_transitos": 0,
        "aviso_sin_senal": 0, "descartes": {"cadencias": 0, "en_transito": 0, "otro_koi": 0},
    }
    t_todo, f_todo, sigma_todo = t, f, sigma
    if len(t) and otras:   # las cadencias de otros KOIs no entran a la profundidad ni a las vistas
        visible = ~mascara_transitos(t, otras, cfg["mascara_transito_duraciones"])
        t, f, sigma = t[visible], f[visible], sigma[visible]
    prof, sig_local, n_centro = medir_profundidad(t, f, sigma, ef) if len(t) else (np.nan, np.nan, 0)
    if n_centro < 3 or not np.isfinite(sig_local) or sig_local <= 0:
        return ej
    escala = max(prof, cfg["escala_min_sigmas"] * sig_local)
    y = (f - 1.0) / escala

    # ---- rama 1: vistas ----
    ph = fase(t, ef)
    semi_local = min(cfg["vista_local_semiancho_duraciones"] * ef.duracion, ef.periodo / 2)
    paridad = numero_de_transito(t, ef) % 2
    vg, vac_g = vista_plegada(ph, y, ef.periodo / 2, ng)
    canales, vac_l = [], []
    for sel, fase_c in ((slice(None), ph), (paridad == 0, ph[paridad == 0]), (paridad == 1, ph[paridad == 1]),
                        (slice(None), fase(t, ef, ef.periodo / 2))):
        serie, vac = vista_plegada(fase_c, y[sel], semi_local, nl)
        canales.append(serie if vac <= 1 / 3 else np.zeros(nl))
        vac_l.append(vac)
    if vac_g <= 1 / 3 and vac_l[0] <= 1 / 3:
        ej["vista_global"] = np.clip(vg, -recorte, recorte).astype(np.float32)[:, None]
        ej["vista_local"] = np.clip(np.stack(canales, axis=-1), -recorte, recorte).astype(np.float32)
        ej["disponible_vistas"] = 1

    # ---- rama 2: tránsitos uno por uno ----
    ventanas = ventanas_por_transito(t_todo, f_todo, sigma_todo, ef, otras, cfg)
    for v in ventanas:
        if not v.valida:
            ej["descartes"][v.motivo] += 1
    validas = [v for v in ventanas if v.valida]
    ej["n_transitos_validos"] = len(validas)
    if len(validas) >= cfg["min_transitos_para_rama"]:
        # En un tránsito individual el ruido por punto puede superar a la profundidad (planetas
        # pequeños, ejemplos sin señal). La escala de las ventanas es el mayor de los dos: la señal
        # vale −1 si domina, y el ruido tiene desviación 1 si domina él. Sin esto, satura el recorte.
        escala_ventanas = max(escala, float(np.median(sigma)))
        for i, v in enumerate(seleccionar_transitos(ventanas, mx)):
            ej["transitos"][i] = np.clip(v.serie / escala_ventanas, -recorte, recorte)
            ej["mascara_transitos"][i] = 1.0
            ej["paridad_transitos"][i] = v.indice % 2
        ej["consistencia"] = estadisticos_consistencia(ventanas)
        ej["disponible_transitos"] = 1

    # ---- rama 4: escalares medidos en la época del ejemplo ----
    snr = prof / sig_local
    valores = [np.log10(ef.periodo), np.log10(ef.duracion * 24.0), prof * 1e6, snr,
               np.log10(max(len(validas), 1)), centroide[0], centroide[1]]
    for i, v in enumerate(valores):
        if np.isfinite(v):
            ej["escalares"][i], ej["faltantes"][i] = v, 0.0
        else:
            ej["escalares"][i], ej["faltantes"][i] = 0.0, 1.0  # 0 SIEMPRE acompañado de su aviso
    ej["aviso_sin_senal"] = int(snr > 5)
    return ej
