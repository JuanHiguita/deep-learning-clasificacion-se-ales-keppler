"""Métricas del proyecto (plan de ejecución, sección 4.3).

Convenciones:
  p  : probabilidades (n, 3) en el orden fijo [planeta, falso_positivo, sin_senal].
  y  : clase verdadera (n,), enteros 0, 1, 2.
  q  : probabilidad de planeta condicionada a que hay señal, q = p0 / (p0 + p1).
Todas las funciones aceptan pesos por ejemplo, necesarios para evaluar con una
tasa base distinta de la del conjunto de prueba (experimento E9).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def _pesos(pesos: np.ndarray | None, n: int) -> np.ndarray:
    if pesos is None:
        return np.ones(n, dtype=float)
    pesos = np.asarray(pesos, dtype=float)
    if pesos.shape != (n,) or np.any(pesos < 0) or pesos.sum() <= 0:
        raise ValueError("Pesos inválidos")
    return pesos


def _validar(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(int)
    if p.ndim != 2 or p.shape[0] != y.shape[0]:
        raise ValueError("Formas incompatibles entre p e y")
    if not np.all(np.isfinite(p)):
        raise ValueError("Las probabilidades contienen NaN o infinitos")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Las filas de p no suman 1")
    return p, y


def q_planeta(p: np.ndarray) -> np.ndarray:
    """P(planeta | hay señal) = p0 / (p0 + p1)."""
    p = np.asarray(p, dtype=float)
    return p[:, 0] / (p[:, 0] + p[:, 1] + EPS)


def auc(puntaje: np.ndarray, positivo: np.ndarray, pesos: np.ndarray | None = None) -> float:
    """Área bajo la curva ROC con empates y pesos. Devuelve NaN si falta una clase."""
    puntaje = np.asarray(puntaje, dtype=float)
    positivo = np.asarray(positivo).astype(bool)
    w = _pesos(pesos, len(puntaje))
    wp, wn = w[positivo].sum(), w[~positivo].sum()
    if wp <= 0 or wn <= 0:
        return float("nan")
    orden = np.argsort(puntaje, kind="mergesort")
    s, pos, w = puntaje[orden], positivo[orden], w[orden]
    # Para cada valor distinto: peso negativo por debajo + mitad del peso negativo empatado.
    _, inicio, inverso = np.unique(s, return_index=True, return_inverse=True)
    neg_por_grupo = np.bincount(inverso, weights=w * (~pos))
    neg_debajo = np.concatenate([[0.0], np.cumsum(neg_por_grupo)[:-1]])
    aporte = neg_debajo[inverso] + 0.5 * neg_por_grupo[inverso]
    return float(np.sum(w * pos * aporte) / (wp * wn))


def auc_planeta_fp(p: np.ndarray, y: np.ndarray, pesos: np.ndarray | None = None) -> float:
    """Métrica principal: AUC planeta vs falso positivo, solo en ejemplos con y < 2."""
    p, y = _validar(p, y)
    real = y < 2
    w = None if pesos is None else np.asarray(pesos, dtype=float)[real]
    return auc(q_planeta(p[real]), y[real] == 0, w)


def exactitud(p: np.ndarray, y: np.ndarray, pesos: np.ndarray | None = None) -> float:
    p, y = _validar(p, y)
    w = _pesos(pesos, len(y))
    return float(np.sum(w * (p.argmax(axis=1) == y)) / w.sum())


def matriz_confusion(p: np.ndarray, y: np.ndarray, n_clases: int = 3) -> np.ndarray:
    p, y = _validar(p, y)
    m = np.zeros((n_clases, n_clases), dtype=int)
    np.add.at(m, (y, p.argmax(axis=1)), 1)
    return m


def nll(p: np.ndarray, y: np.ndarray, pesos: np.ndarray | None = None) -> float:
    """Log-loss multiclase."""
    p, y = _validar(p, y)
    w = _pesos(pesos, len(y))
    return float(-np.sum(w * np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0))) / w.sum())


def brier(p: np.ndarray, y: np.ndarray, pesos: np.ndarray | None = None) -> float:
    """Brier multiclase: media de la suma de cuadrados sobre las clases."""
    p, y = _validar(p, y)
    w = _pesos(pesos, len(y))
    uno = np.zeros_like(p)
    uno[np.arange(len(y)), y] = 1.0
    return float(np.sum(w * np.sum((p - uno) ** 2, axis=1)) / w.sum())


def _ece(confianza: np.ndarray, acierto: np.ndarray, w: np.ndarray, n_intervalos: int) -> float:
    bordes = np.linspace(0.0, 1.0, n_intervalos + 1)
    idx = np.clip(np.digitize(confianza, bordes[1:-1], right=False), 0, n_intervalos - 1)
    total = w.sum()
    error = 0.0
    for b in range(n_intervalos):
        m = idx == b
        wb = w[m].sum()
        if wb > 0:
            conf_b = np.sum(w[m] * confianza[m]) / wb
            acc_b = np.sum(w[m] * acierto[m]) / wb
            error += (wb / total) * abs(acc_b - conf_b)
    return float(error)


def ece_top(p: np.ndarray, y: np.ndarray, n_intervalos: int = 15, pesos: np.ndarray | None = None) -> float:
    """ECE de la clase más probable, intervalos de igual ancho (Guo et al., 2017)."""
    p, y = _validar(p, y)
    w = _pesos(pesos, len(y))
    return _ece(p.max(axis=1), (p.argmax(axis=1) == y).astype(float), w, n_intervalos)


def ece_binario(q: np.ndarray, positivo: np.ndarray, n_intervalos: int = 15,
                pesos: np.ndarray | None = None) -> float:
    """ECE de una probabilidad binaria: |frecuencia observada − probabilidad media| por intervalo."""
    q = np.asarray(q, dtype=float)
    positivo = np.asarray(positivo).astype(float)
    if q.shape != positivo.shape or np.any((q < 0) | (q > 1)):
        raise ValueError("q inválido")
    w = _pesos(pesos, len(q))
    return _ece(q, positivo, w, n_intervalos)


def pesos_para_tasa_base(positivo: np.ndarray, tasa: float) -> np.ndarray:
    """Pesos que hacen que la fracción ponderada de positivos sea `tasa` (E9)."""
    positivo = np.asarray(positivo).astype(bool)
    if not 0 < tasa < 1:
        raise ValueError("La tasa base debe estar entre 0 y 1")
    n_pos, n_neg = positivo.sum(), (~positivo).sum()
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Hace falta al menos un ejemplo de cada clase")
    return np.where(positivo, tasa / n_pos, (1 - tasa) / n_neg) * len(positivo)


def corregir_prior(q: np.ndarray, tasa_entrenamiento: float, tasa_objetivo: float) -> np.ndarray:
    """Corrección por cambio de tasa base (Saerens et al., 2002), caso binario.

    q' = q·r / (q·r + 1 − q), con r = razón de momios objetivo / razón de momios de entrenamiento.
    """
    q = np.asarray(q, dtype=float)
    for t in (tasa_entrenamiento, tasa_objetivo):
        if not 0 < t < 1:
            raise ValueError("Las tasas deben estar entre 0 y 1")
    r = (tasa_objetivo / (1 - tasa_objetivo)) / (tasa_entrenamiento / (1 - tasa_entrenamiento))
    return q * r / (q * r + 1 - q)
