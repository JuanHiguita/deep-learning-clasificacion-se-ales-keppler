"""Intervalos de confianza por bootstrap agrupado por estrella (plan, sección 4.3).

Se remuestrean ESTRELLAS con reemplazo, no ejemplos: los dos ejemplos de una misma
estrella (tránsito y sin señal) entran o salen juntos.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Intervalo:
    estimacion: float
    ic_inf: float
    ic_sup: float
    n_validos: int  # remuestreos en los que la métrica se pudo calcular

    def incluye(self, valor: float = 0.0) -> bool:
        return self.ic_inf <= valor <= self.ic_sup


def _indices_por_grupo(grupos: np.ndarray) -> list[np.ndarray]:
    grupos = np.asarray(grupos)
    _, inverso = np.unique(grupos, return_inverse=True)
    orden = np.argsort(inverso, kind="mergesort")
    cortes = np.flatnonzero(np.diff(inverso[orden])) + 1
    return np.split(orden, cortes)


def bootstrap_por_estrella(
    grupos: np.ndarray,
    estadistico: Callable[[np.ndarray], float],
    n_remuestreos: int = 2000,
    semilla: int = 20261206,
    percentiles: tuple[float, float] = (2.5, 97.5),
) -> Intervalo:
    """Intervalo por percentiles de `estadistico(indices)`.

    estadistico recibe un arreglo de índices de ejemplos (con repeticiones) y devuelve
    un número. Para una diferencia pareada entre dos modelos, basta con que el
    estadístico calcule métrica_A(indices) − métrica_B(indices): ambos modelos se
    evalúan sobre las mismas estrellas en cada remuestreo.
    """
    por_grupo = _indices_por_grupo(grupos)
    n_grupos = len(por_grupo)
    if n_grupos < 2:
        raise ValueError("Hacen falta al menos 2 estrellas")
    todos = np.arange(len(np.asarray(grupos)))
    estimacion = float(estadistico(todos))
    rng = np.random.default_rng(semilla)
    valores = np.empty(n_remuestreos)
    for b in range(n_remuestreos):
        elegidos = rng.integers(0, n_grupos, n_grupos)
        indices = np.concatenate([por_grupo[g] for g in elegidos])
        valores[b] = estadistico(indices)
    validos = valores[np.isfinite(valores)]
    if len(validos) < 0.9 * n_remuestreos:
        raise ValueError(
            f"Solo {len(validos)} de {n_remuestreos} remuestreos son válidos; el estrato es demasiado pequeño"
        )
    inf, sup = np.percentile(validos, percentiles)
    return Intervalo(estimacion, float(inf), float(sup), int(len(validos)))


def diferencia_media_semillas(
    metrica: Callable[[np.ndarray, np.ndarray], float],
    y: np.ndarray,
    predicciones_a: list[np.ndarray],
    predicciones_b: list[np.ndarray],
) -> Callable[[np.ndarray], float]:
    """Construye el estadístico Δ = media_semillas(métrica A) − media_semillas(métrica B).

    metrica(p, y) → número. predicciones_x: una matriz de probabilidades por semilla,
    todas alineadas con y.
    """
    y = np.asarray(y)

    def estadistico(indices: np.ndarray) -> float:
        a = np.mean([metrica(p[indices], y[indices]) for p in predicciones_a])
        b = np.mean([metrica(p[indices], y[indices]) for p in predicciones_b])
        return float(a - b)

    return estadistico
