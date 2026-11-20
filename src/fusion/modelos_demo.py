"""Arquitecturas de la demo (src/demo/demo_cnn.py, commit 6ffa66f), reproducidas capa por capa.

NO son un aporte del curso de Deep Learning: son código anterior a los cursos. Se usan para
  - la rama de imagen v0 (congelada) de la fusión, y
  - la línea base M0 ("la demo vuelta a medir" sobre la base del curso).
Única diferencia con la demo: la capa densa de 32 de cada rama se llama 'embedding' para poder exportarla.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras  # noqa: E402
import numpy as np  # noqa: E402
from keras import layers  # noqa: E402


def _rama_imagen(a):
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(
        a
    )  # CNN 2D SIN pooling, como en la demo
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.Dropout(0.4)(layers.Flatten()(x))
    return x


def _rama_curva(b):
    z = layers.Conv1D(16, 5, padding="same", activation="relu")(b)
    z = layers.MaxPooling1D(2)(z)
    z = layers.Conv1D(32, 5, padding="same", activation="relu")(z)
    return layers.GlobalMaxPooling1D()(z)


def imagen_v0(semilla: int) -> keras.Model:
    """Rama de imagen de la demo sola: (11, 11, 4) → 32 → 3 clases."""
    keras.utils.set_random_seed(semilla)
    a = keras.Input((11, 11, 4), name="img")
    z = layers.Dense(32, activation="relu", name="embedding")(_rama_imagen(a))
    return keras.Model(a, layers.Dense(3, activation="softmax")(z), name="imagen_v0")


def demo_curva(semilla: int) -> keras.Model:
    """Rama 1D de la demo sola: (61, 2) → 32 → 3 clases."""
    keras.utils.set_random_seed(semilla)
    b = keras.Input((61, 2), name="seq")
    z = layers.Dense(32, activation="relu", name="embedding")(_rama_curva(b))
    h = layers.Dropout(0.3)(layers.Dense(32, activation="relu")(z))
    return keras.Model(b, layers.Dense(3, activation="softmax")(h), name="demo_curva")


def demo_completo(semilla: int) -> keras.Model:
    """Modelo combinado de la demo (imagen + curva)."""
    keras.utils.set_random_seed(semilla)
    a = keras.Input((11, 11, 4), name="img")
    b = keras.Input((61, 2), name="seq")
    h = layers.Concatenate()(
        [
            layers.Dense(32, activation="relu")(_rama_imagen(a)),
            layers.Dense(32, activation="relu")(_rama_curva(b)),
        ]
    )
    z = layers.Dense(32, activation="relu", name="embedding")(h)
    return keras.Model(
        [a, b], layers.Dense(3, activation="softmax")(layers.Dropout(0.3)(z)), name="demo_completo"
    )


def aumentar_demo(entradas: dict, rng: np.random.Generator) -> dict:
    """Aumentación de la demo: rotaciones de 90° de la imagen y ruido leve (0,05)."""
    s = dict(entradas)
    if "img" in entradas:
        k = rng.integers(0, 4, len(entradas["img"]))
        im = np.stack(
            [np.rot90(i, kk, axes=(0, 1)) for i, kk in zip(entradas["img"], k, strict=True)]
        ).copy()
        im[..., 0] += rng.normal(0, 0.05, im[..., 0].shape)
        s["img"] = im.astype(np.float32)
    if "seq" in entradas:
        s["seq"] = (entradas["seq"] + rng.normal(0, 0.05, entradas["seq"].shape)).astype(np.float32)
    return s
