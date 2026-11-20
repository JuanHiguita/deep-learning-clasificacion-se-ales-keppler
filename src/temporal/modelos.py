"""Modelos de las ramas del curso (spec 004). Todos terminan en una capa 'embedding' y una cabeza softmax."""
from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras  # noqa: E402
from keras import layers  # noqa: E402


def m4_escalares(semilla: int, d: int = 7, dropout: float = 0.2) -> keras.Model:
    """Rama 4: escalares estandarizados + avisos de faltante → vector de 16."""
    keras.utils.set_random_seed(semilla)
    x = keras.Input((d,), name="escalares")
    f = keras.Input((d,), name="faltantes")
    h = layers.Concatenate()([x, f])
    h = layers.Dense(32, activation="relu")(h)
    h = layers.Dropout(dropout)(h)
    z = layers.Dense(16, activation="relu", name="embedding")(h)
    return keras.Model([x, f], layers.Dense(3, activation="softmax", name="clases")(z), name="m4_escalares")
