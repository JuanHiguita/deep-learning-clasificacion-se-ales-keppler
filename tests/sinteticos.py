"""Curvas de luz sintéticas SOLO para probar código. Nunca producen una cifra del informe."""

import numpy as np

from temporal.curvas import CADENCIA_D, Efemerides


def curva(
    ef: Efemerides,
    profundidad=1000e-6,
    ruido=200e-6,
    semilla=0,
    n_trimestres=17,
    profundidad_impar=None,
    secundario=0.0,
    otro: tuple[Efemerides, float] | None = None,
    borrar: tuple[float, float] | None = None,
):
    """Devuelve una lista de trimestres [(tiempo, flujo)] con tránsitos de caja.

    profundidad_impar: si se da, los tránsitos impares tienen esa profundidad (estrella doble).
    secundario: profundidad de un eclipse en fase 0,5. otro: segundo planeta (efemérides, profundidad).
    borrar: intervalo de tiempo que se elimina (hueco en los datos).
    """
    rng = np.random.default_rng(semilla)
    trimestres = []
    inicio = 131.5
    for q in range(n_trimestres):
        t = inicio + np.arange(int(90 / CADENCIA_D)) * CADENCIA_D
        inicio = t[-1] + 1.5  # hueco entre trimestres
        f = 1.0 + ruido * rng.standard_normal(len(t))
        f *= (1.0 + 0.002 * np.sin(2 * np.pi * (t - t[0]) / 25.0 + q)) * (
            1000.0 + 50.0 * q
        )  # tendencia y escala
        n = np.round((t - ef.t0) / ef.periodo).astype(int)
        en = np.abs(t - ef.t0 - n * ef.periodo) < ef.duracion / 2
        d = np.full(len(t), profundidad)
        if profundidad_impar is not None:
            d = np.where(n % 2 == 1, profundidad_impar, profundidad)
        f = np.where(en, f * (1 - d), f)
        if secundario:
            ph = ((t - ef.t0) % ef.periodo) - ef.periodo / 2
            f = np.where(np.abs(ph) < ef.duracion / 2, f * (1 - secundario), f)
        if otro is not None:
            o, prof_o = otro
            no = np.round((t - o.t0) / o.periodo)
            f = np.where(np.abs(t - o.t0 - no * o.periodo) < o.duracion / 2, f * (1 - prof_o), f)
        if borrar is not None:
            quedan = (t < borrar[0]) | (t > borrar[1])
            t, f = t[quedan], f[quedan]
        trimestres.append((t, f))
    return trimestres
