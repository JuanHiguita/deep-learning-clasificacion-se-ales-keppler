"""Modelos de las ramas del curso (spec 004). Todos terminan en una capa 'embedding' y una cabeza softmax.

Rama 1: m1_astronet (CNN 1D sobre las vistas plegadas).
Rama 2: m2 = codificador por tránsito con pesos compartidos + agregador (promedio | gru | atencion).
Rama 4: m4_escalares.
Las capas propias usan keras.ops y están registradas, para poder cargar los .keras con safe_mode=True.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras  # noqa: E402
from keras import layers, ops  # noqa: E402

AGREGADORES = ("promedio", "gru", "atencion")


@keras.saving.register_keras_serializable(package="temporal")
class AplicarMascara(layers.Layer):
    """Pone en cero los tránsitos vacíos. Entradas: [x (lote, T, C), mascara (lote, T)]."""

    def call(self, entradas):
        x, m = entradas
        return x * ops.expand_dims(ops.cast(m, x.dtype), -1)


@keras.saving.register_keras_serializable(package="temporal")
class MascaraBooleana(layers.Layer):
    """(lote, T) → booleano. Con `cuadrada`, (lote, 1, T) para enmascarar las claves de la atención."""

    def __init__(self, cuadrada: bool = False, **kw):
        super().__init__(**kw)
        self.cuadrada = cuadrada

    def call(self, m):
        b = ops.greater(m, 0.5)
        return ops.expand_dims(b, 1) if self.cuadrada else b

    def get_config(self):
        return {**super().get_config(), "cuadrada": self.cuadrada}


@keras.saving.register_keras_serializable(package="temporal")
class MediaEnmascarada(layers.Layer):
    """Promedio sobre los tránsitos presentes. Invariante al orden."""

    def call(self, entradas):
        x, m = entradas
        m = ops.expand_dims(ops.cast(m, x.dtype), -1)
        return ops.sum(x * m, axis=1) / ops.maximum(ops.sum(m, axis=1), 1.0)


@keras.saving.register_keras_serializable(package="temporal")
class AtencionAgregadora(layers.Layer):
    """Suma ponderada con pesos aprendidos (attention pooling). Invariante al orden.

    Devuelve [vector (lote, C) ‖ pesos (lote, T)] concatenados; `Recorte` separa cada parte.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.puntaje = layers.Dense(1)

    def build(self, formas):
        self.puntaje.build(formas[0])
        super().build(formas)

    def call(self, entradas):
        x, m = entradas
        m = ops.cast(m, x.dtype)
        s = ops.squeeze(self.puntaje(x), -1) + (1.0 - m) * -1e9
        w = ops.softmax(s, axis=-1) * m
        w = w / ops.maximum(ops.sum(w, axis=-1, keepdims=True), 1e-9)
        return ops.concatenate([ops.sum(x * ops.expand_dims(w, -1), axis=1), w], axis=-1)


@keras.saving.register_keras_serializable(package="temporal")
class Recorte(layers.Layer):
    """x[:, inicio:fin]."""

    def __init__(self, inicio: int, fin: int, **kw):
        super().__init__(**kw)
        self.inicio, self.fin = inicio, fin

    def call(self, x):
        return x[:, self.inicio : self.fin]

    def get_config(self):
        return {**super().get_config(), "inicio": self.inicio, "fin": self.fin}


def _cabeza(h, dim: int, dropout: float):
    h = layers.Dropout(dropout)(h)
    z = layers.Dense(dim, activation="relu", name="embedding")(h)
    return layers.Dense(3, activation="softmax", name="clases")(z)


NORMALIZACIONES = (None, "lote", "capa")


def _normalizar(x, normalizacion: str | None):
    """Capa de normalización opcional: por lote (BatchNorm) o por capa (LayerNorm)."""
    if normalizacion not in NORMALIZACIONES:
        raise ValueError(normalizacion)
    if normalizacion == "lote":
        return layers.BatchNormalization()(x)
    if normalizacion == "capa":
        return layers.LayerNormalization()(x)
    return x


def _con_profundidad(filtros: tuple[int, ...], capas_extra: int) -> tuple[int, ...]:
    """Quita el último bloque (−1), deja igual (0) o añade uno con el doble de filtros (+1)."""
    if capas_extra == -1:
        return filtros[:-1] if len(filtros) > 1 else filtros
    if capas_extra == 1:
        return (*filtros, 2 * filtros[-1])
    if capas_extra != 0:
        raise ValueError(capas_extra)
    return filtros


def _torre(
    x,
    filtros: tuple[int, ...],
    ventana_pool: int,
    pooling: str,
    activacion: str = "relu",
    normalizacion: str | None = None,
):
    for f in filtros:
        x = layers.Conv1D(f, 5, padding="same", activation=activacion)(x)
        if pooling == "paso":  # sin pooling: la segunda convolución reduce con paso 2
            x = layers.Conv1D(f, 5, padding="same", activation=activacion, strides=2)(x)
            x = _normalizar(x, normalizacion)
        else:
            x = layers.Conv1D(f, 5, padding="same", activation=activacion)(x)
            x = _normalizar(x, normalizacion)
            capa = layers.MaxPooling1D if pooling == "max" else layers.AveragePooling1D
            x = capa(ventana_pool, strides=2, padding="same")(x)
    return layers.GlobalMaxPooling1D()(x)


def _escalar(filtros: tuple[int, ...], ancho: float) -> tuple[int, ...]:
    """Número de filtros de cada capa multiplicado por `ancho` (1.0 = arquitectura del protocolo)."""
    return tuple(max(4, int(round(f * ancho))) for f in filtros)


def m1_astronet(
    semilla: int,
    pooling: str = "max",
    dropout: float = 0.3,
    ancho: float = 1.0,
    activacion: str = "relu",
    normalizacion: str | None = None,
    capas_extra: int = 0,
) -> keras.Model:
    """Rama 1: CNN 1D tipo AstroNet reducida sobre la vista global (2001, 1) y la local (201, 4).

    `ancho` multiplica el número de filtros y de neuronas ocultas; el vector de salida no cambia.
    """
    if pooling not in ("max", "promedio", "paso"):
        raise ValueError(pooling)
    keras.utils.set_random_seed(semilla)
    g = keras.Input((2001, 1), name="vista_global")
    loc = keras.Input((201, 4), name="vista_local")
    h = layers.Concatenate()(
        [
            _torre(
                g,
                _con_profundidad(_escalar((16, 32, 64), ancho), capas_extra),
                5,
                pooling,
                activacion,
                normalizacion,
            ),
            _torre(
                loc,
                _con_profundidad(_escalar((16, 32), ancho), capas_extra),
                7,
                pooling,
                activacion,
                normalizacion,
            ),
        ]
    )
    h = layers.Dense(_escalar((64,), ancho)[0], activation=activacion)(h)
    return keras.Model([g, loc], _cabeza(h, 32, dropout), name=f"m1_astronet_{pooling}")


def codificador_por_transito(
    ancho: float = 1.0,
    activacion: str = "relu",
    normalizacion: str | None = None,
    capas_extra: int = 0,
) -> keras.Model:
    """φ: una ventana de tránsito (61, 1) → 32. Se aplica con los mismos pesos a cada tránsito."""
    filtros = _con_profundidad(_escalar((16, 32), ancho), capas_extra)
    if (activacion, normalizacion, capas_extra) != ("relu", None, 0):
        capas = [keras.Input((61, 1))]
        for f in filtros:
            capas.append(layers.Conv1D(f, 5, padding="same", activation=activacion))
            if normalizacion == "lote":
                capas.append(layers.BatchNormalization())
            elif normalizacion == "capa":
                capas.append(layers.LayerNormalization())
        capas += [layers.GlobalMaxPooling1D(), layers.Dense(32, activation=activacion)]
        return keras.Sequential(capas, name="phi")
    return keras.Sequential(
        [
            keras.Input((61, 1)),
            layers.Conv1D(filtros[0], 5, padding="same", activation="relu"),
            layers.Conv1D(filtros[1], 5, padding="same", activation="relu"),
            layers.GlobalMaxPooling1D(),
            layers.Dense(32, activation="relu"),
        ],
        name="phi",
    )


def m2(
    semilla: int,
    agregador: str,
    usar_consistencia: bool = True,
    dropout: float = 0.3,
    ancho: float = 1.0,
    unidades_gru: int = 32,
    cabezas: int = 2,
    activacion: str = "relu",
    normalizacion: str | None = None,
    capas_extra: int = 0,
) -> keras.Model:
    """Rama 2: φ compartido por tránsito + agregador. Solo cambia cómo se combinan los tránsitos.

    Hiperparámetros de arquitectura (los valores por defecto son los del protocolo):
    `ancho` (filtros de φ), `unidades_gru` (solo GRU) y `cabezas` de atención (solo atención).
    """
    if agregador not in AGREGADORES:
        raise ValueError(agregador)
    keras.utils.set_random_seed(semilla)
    tr = keras.Input((32, 61), name="transitos")
    mk = keras.Input((32,), name="mascara_transitos")
    co = keras.Input((4,), name="consistencia")
    h = layers.TimeDistributed(
        codificador_por_transito(ancho, activacion, normalizacion, capas_extra),
        name="phi_por_transito",
    )(layers.Reshape((32, 61, 1))(tr))
    h = AplicarMascara()([h, mk])
    if agregador == "promedio":
        v = MediaEnmascarada(name="agregador")([h, mk])
    elif agregador == "gru":
        v = layers.Bidirectional(layers.GRU(unidades_gru), name="agregador")(
            h, mask=MascaraBooleana()(mk)
        )
    else:
        a = layers.MultiHeadAttention(
            num_heads=cabezas, key_dim=32 // cabezas, name="autoatencion"
        )(h, h, attention_mask=MascaraBooleana(cuadrada=True)(mk))
        h = layers.LayerNormalization()(layers.Add()([h, a]))
        f = layers.Dense(32)(layers.Dense(64, activation=activacion)(h))
        h = AplicarMascara()([layers.LayerNormalization()(layers.Add()([h, f])), mk])
        v = Recorte(0, 32, name="vector_agregado")(AtencionAgregadora(name="agregador")([h, mk]))
    if usar_consistencia:
        v = layers.Concatenate()([v, co])
    entradas = [tr, mk, co] if usar_consistencia else [tr, mk]  # sin entradas sin usar
    return keras.Model(entradas, _cabeza(v, 32, dropout), name=f"m2_{agregador}")


def submodelo_pesos_atencion(modelo: keras.Model) -> keras.Model:
    """Para m2 con atención: devuelve el peso que el agregador dio a cada tránsito, (lote, 32)."""
    capa = next(
        c for c in modelo.layers if isinstance(c, AtencionAgregadora)
    )  # por tipo: el nombre puede variar
    salida = capa.output
    return keras.Model(modelo.inputs, Recorte(32, 64)(salida))


def m4_escalares(
    semilla: int,
    d: int = 7,
    dropout: float = 0.2,
    oculta: int = 32,
    activacion: str = "relu",
    normalizacion: str | None = None,
    capas_extra: int = 0,
) -> keras.Model:
    """Rama 4: escalares estandarizados + avisos de faltante → vector de 16."""
    keras.utils.set_random_seed(semilla)
    x = keras.Input((d,), name="escalares")
    f = keras.Input((d,), name="faltantes")
    if (activacion, normalizacion, capas_extra) == ("relu", None, 0):  # protocolo
        h = layers.Dense(oculta, activation="relu")(layers.Concatenate()([x, f]))
    else:
        h = layers.Concatenate()([x, f])
        for _ in range(1 + capas_extra):  # −1: sin capa oculta; +1: dos capas ocultas
            h = _normalizar(layers.Dense(oculta, activation=activacion)(h), normalizacion)
    return keras.Model([x, f], _cabeza(h, 16, dropout), name="m4_escalares")
