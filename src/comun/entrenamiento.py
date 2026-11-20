"""Motor de experimentos: entrena, evalúa y exporta cualquier rama con el mismo protocolo (spec 003).

Para cada semilla y cada partición de prueba k:

    entrenamiento      = particiones distintas de k y de (k + 1) mod n
    validación interna = partición (k + 1) mod n   → decide la parada temprana
    prueba             = partición k               → solo se usa para predecir al final

El flujo de `entrenar_cv` es:

    por cada semilla
        por cada partición
            1. separar los ejemplos y estandarizar con el entrenamiento
            2. entrenar con parada temprana
            3. predecir prueba y validación interna
            4. (opcional) exportar el codificador y guardar el modelo
        guardar las predicciones de la semilla
    resumir métricas

Cada partición terminada se guarda en `parciales/`, de modo que una corrida interrumpida se reanuda.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from comun import estadistica, metricas  # noqa: E402
from comun.protocolo import Separacion, comprobar_sin_solape, separar  # noqa: E402
from comun.utilidades import escribir_manifiesto  # noqa: E402

Entradas = dict[str, np.ndarray]
N_CLASES = 3


class Incompleto(RuntimeError):
    """Se agotó el presupuesto de tiempo; lo entrenado quedó guardado y la corrida se puede reanudar."""


@dataclass(frozen=True)
class Experimento:
    """Todo lo que distingue a un experimento de otro. El protocolo es el mismo para todos."""

    nombre: str
    constructor: Callable[[int], keras.Model]  # semilla → modelo sin compilar, con capa "embedding"
    claves_entrada: tuple[str, ...]  # nombres de las entradas del modelo = claves del dataset
    clave_disponible: str | None = None  # p. ej. "disponible_transitos"
    aumentar: Callable[[Entradas, np.random.Generator], Entradas] | None = None
    estandarizar: tuple[str, ...] = ()  # claves que se estandarizan con el entrenamiento
    clave_faltantes: str | None = None  # máscara (1 = falta) de las claves estandarizadas


@dataclass(frozen=True)
class _Datos:
    """Dataset de un experimento, ya alineado con la tabla de ejemplos."""

    entradas: Entradas
    ids: np.ndarray
    tabla: pd.DataFrame
    y: np.ndarray
    particion: np.ndarray
    kepid: np.ndarray
    disponible: np.ndarray


@dataclass
class _ResultadoParticion:
    """Lo que deja una partición entrenada (y lo que se guarda para poder reanudar)."""

    p_prueba: np.ndarray
    p_validacion: np.ndarray
    historial: dict
    segundos_por_epoca: float
    parametros: int


# ---------------------------------------------------------------------------------------------
# Preparación de datos
# ---------------------------------------------------------------------------------------------


def alinear(ejemplos: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    """Filas de ejemplos.csv en el orden de `ids`. Falla si un id no existe o está prerregistrado."""
    tabla = ejemplos.set_index("id_ejemplo").loc[[str(i) for i in ids]].reset_index()
    if tabla["prerregistrado"].astype(bool).any():
        raise ValueError("El dataset contiene ejemplos prerregistrados")
    return tabla


def estadisticas_estandarizacion(
    x: np.ndarray, faltantes: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Media y desviación por columna, sin contar los valores faltantes."""
    x = np.asarray(x, dtype=float)
    presente = np.ones_like(x, dtype=bool) if faltantes is None else ~np.asarray(faltantes, bool)
    n = np.maximum(presente.sum(axis=0), 1)
    media = np.where(presente, x, 0.0).sum(axis=0) / n
    varianza = (np.where(presente, x - media, 0.0) ** 2).sum(axis=0) / n
    desviacion = np.sqrt(varianza)
    return media, np.where(desviacion < 1e-6, 1.0, desviacion)


def aplicar_estandarizacion(
    x: np.ndarray, media: np.ndarray, desv: np.ndarray, faltantes: np.ndarray | None = None
) -> np.ndarray:
    """(x − media) / desviación; los faltantes quedan en 0 (su aviso viaja en la máscara)."""
    z = (np.asarray(x, dtype=float) - media) / desv
    if faltantes is not None:
        z = np.where(np.asarray(faltantes, dtype=bool), 0.0, z)
    return z.astype(np.float32)


def pesos_por_clase(y: np.ndarray, n_clases: int = N_CLASES) -> dict[int, float]:
    """Peso inverso a la frecuencia de cada clase en el entrenamiento."""
    return {c: float(len(y) / (n_clases * max(int((y == c).sum()), 1))) for c in range(n_clases)}


def _tomar(entradas: Entradas, claves: Sequence[str], indices: np.ndarray) -> Entradas:
    return {clave: entradas[clave][indices] for clave in claves}


def _estandarizar_con_entrenamiento(
    exp: Experimento, entradas: Entradas, i_ent: np.ndarray
) -> tuple[Entradas, dict]:
    """Estandariza las claves pedidas usando SOLO los ejemplos de entrenamiento."""
    x, estadisticas = dict(entradas), {}
    faltantes = entradas[exp.clave_faltantes] if exp.clave_faltantes else None
    for clave in exp.estandarizar:
        faltantes_ent = None if faltantes is None else faltantes[i_ent]
        media, desv = estadisticas_estandarizacion(entradas[clave][i_ent], faltantes_ent)
        x[clave] = aplicar_estandarizacion(entradas[clave], media, desv, faltantes)
        estadisticas[clave] = (media, desv)
    return x, estadisticas


def _preparar_datos(exp: Experimento, entradas: Entradas, ejemplos: pd.DataFrame) -> _Datos:
    ids = np.array([str(i) for i in entradas["id_ejemplo"]], dtype="U")
    tabla = alinear(ejemplos, ids)
    if exp.clave_disponible is None:
        disponible = np.ones(len(ids), dtype=bool)
    else:
        disponible = entradas[exp.clave_disponible].astype(bool)
    return _Datos(
        entradas=entradas,
        ids=ids,
        tabla=tabla,
        y=tabla["clase"].to_numpy().astype(int),
        particion=tabla["particion"].to_numpy(),
        kepid=tabla["kepid"].to_numpy(),
        disponible=disponible,
    )


# ---------------------------------------------------------------------------------------------
# Entrenamiento de un modelo
# ---------------------------------------------------------------------------------------------


def crear_optimizador(cfg: dict) -> keras.optimizers.Optimizer:
    """Optimizador según `cfg["optimizador"]`. El del protocolo es AdamW.

    SGD usa momento de Nesterov 0,9 y una tasa 10 veces mayor: con la tasa de Adam casi no avanza.
    """
    nombre = cfg.get("optimizador", "adamw")
    tasa, decaimiento = cfg["tasa_aprendizaje"], cfg["weight_decay"]
    if nombre == "adamw":
        return keras.optimizers.AdamW(learning_rate=tasa, weight_decay=decaimiento)
    if nombre == "adam":  # igual que AdamW pero sin decaimiento de pesos
        return keras.optimizers.Adam(learning_rate=tasa)
    if nombre == "sgd":
        return keras.optimizers.SGD(
            learning_rate=10 * tasa, momentum=0.9, nesterov=True, weight_decay=decaimiento
        )
    if nombre == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=tasa, weight_decay=decaimiento)
    raise ValueError(f"Optimizador desconocido: {nombre!r}")


def crear_perdida(cfg: dict):
    """Función de pérdida según `cfg["perdida"]`. La del protocolo es la entropía cruzada.

    "suavizado": entropía cruzada con suavizado de etiquetas 0,1 (castiga la confianza extrema).
    "focal": pérdida focal con γ = 2 (da más peso a los ejemplos difíciles; Lin et al., 2017).
    """
    nombre = cfg.get("perdida", "ce")
    if nombre == "ce":
        return "sparse_categorical_crossentropy"
    if nombre not in ("suavizado", "focal"):
        raise ValueError(f"Pérdida desconocida: {nombre!r}")

    def perdida(y, p):
        en_columnas = keras.ops.one_hot(
            keras.ops.cast(keras.ops.reshape(y, (-1,)), "int32"), N_CLASES
        )
        if nombre == "suavizado":
            return keras.losses.categorical_crossentropy(en_columnas, p, label_smoothing=0.1)
        return keras.losses.categorical_focal_crossentropy(en_columnas, p, gamma=2.0)

    return perdida


def entrenar_una(
    modelo: keras.Model,
    x_ent: Entradas,
    y_ent: np.ndarray,
    x_val: Entradas,
    y_val: np.ndarray,
    cfg: dict,
    aumentar=None,
    rng: np.random.Generator | None = None,
    usar_pesos_de_clase: bool = True,
) -> tuple[keras.Model, dict]:
    """Bucle de épocas con parada temprana por NLL de validación; restaura los mejores pesos."""
    modelo.compile(optimizer=crear_optimizador(cfg), loss=crear_perdida(cfg))
    con_pesos = usar_pesos_de_clase and cfg.get("pesos_de_clase", True)
    pesos = pesos_por_clase(y_ent) if con_pesos else None
    rng = rng or np.random.default_rng(0)

    mejor_perdida, epocas_sin_mejorar, mejores_pesos = np.inf, 0, modelo.get_weights()
    historial = {"perdida_entrenamiento": [], "perdida_validacion": [], "segundos_por_epoca": []}

    for _ in range(cfg["epocas_max"]):
        x_epoca = aumentar(x_ent, rng) if aumentar else x_ent
        inicio = time.perf_counter()
        registro = modelo.fit(
            x_epoca,
            y_ent,
            epochs=1,
            batch_size=cfg["lote"],
            shuffle=True,
            verbose=0,
            class_weight=pesos,
        )
        historial["segundos_por_epoca"].append(time.perf_counter() - inicio)
        perdida_val = float(modelo.evaluate(x_val, y_val, verbose=0, batch_size=256))
        historial["perdida_entrenamiento"].append(float(registro.history["loss"][0]))
        historial["perdida_validacion"].append(perdida_val)

        if perdida_val < mejor_perdida - 1e-4:
            mejor_perdida, epocas_sin_mejorar = perdida_val, 0
            mejores_pesos = modelo.get_weights()
        else:
            epocas_sin_mejorar += 1
            if epocas_sin_mejorar >= cfg["paciencia"]:
                break

    modelo.set_weights(mejores_pesos)
    historial["epoca_elegida"] = int(np.argmin(historial["perdida_validacion"])) + 1
    historial["mejor_perdida_validacion"] = float(mejor_perdida)
    return modelo, historial


def _predecir(
    modelo: keras.Model, exp: Experimento, x: Entradas, indices, disponible, a_priori
) -> np.ndarray:
    """Probabilidades de `indices`. Los ejemplos no disponibles reciben la probabilidad a priori."""
    p = np.tile(a_priori, (len(indices), 1))
    con_dato = disponible[indices]
    if con_dato.any():
        lote = _tomar(x, exp.claves_entrada, indices[con_dato])
        p[con_dato] = modelo.predict(lote, verbose=0, batch_size=256)
    return p


def _entrenar_particion(
    exp: Experimento, datos: _Datos, sep: Separacion, semilla: int, cfg: dict
) -> tuple[_ResultadoParticion, keras.Model, Entradas, dict]:
    """Entrena el modelo de una partición y predice su prueba y su validación interna."""
    k = sep.k
    i_ent = np.flatnonzero(sep.entrenamiento & datos.disponible)
    i_val = np.flatnonzero(sep.validacion & datos.disponible)
    x, estadisticas = _estandarizar_con_entrenamiento(exp, datos.entradas, i_ent)

    semilla_modelo = 1000 * semilla + k
    keras.utils.set_random_seed(semilla_modelo)
    modelo = exp.constructor(semilla_modelo)
    rng = np.random.default_rng([semilla, k])
    modelo, historial = entrenar_una(
        modelo,
        _tomar(x, exp.claves_entrada, i_ent),
        datos.y[i_ent],
        _tomar(x, exp.claves_entrada, i_val),
        datos.y[i_val],
        cfg,
        exp.aumentar,
        rng,
    )

    a_priori = np.bincount(datos.y[i_ent], minlength=N_CLASES) / len(i_ent)
    tiempos = historial["segundos_por_epoca"]
    resultado = _ResultadoParticion(
        p_prueba=_predecir(modelo, exp, x, np.flatnonzero(sep.prueba), datos.disponible, a_priori),
        p_validacion=_predecir(
            modelo, exp, x, np.flatnonzero(sep.validacion), datos.disponible, a_priori
        ),
        historial={"particion": k, **historial},
        segundos_por_epoca=float(np.mean(tiempos[1:] or tiempos)),  # la 1.ª época incluye compilar
        parametros=int(modelo.count_params()),
    )
    return resultado, modelo, x, estadisticas


# ---------------------------------------------------------------------------------------------
# Reanudación: una partición terminada no se repite
# ---------------------------------------------------------------------------------------------


def _guardar_parcial(ruta: Path, r: _ResultadoParticion) -> None:
    ruta.parent.mkdir(exist_ok=True)
    np.savez(
        ruta,
        p_prueba=r.p_prueba,
        p_validacion=r.p_validacion,
        historial=np.array(json.dumps(r.historial)),
        segundos=r.segundos_por_epoca,
        parametros=r.parametros,
    )


def _cargar_parcial(ruta: Path) -> _ResultadoParticion:
    with np.load(ruta, allow_pickle=False) as c:
        return _ResultadoParticion(
            p_prueba=c["p_prueba"],
            p_validacion=c["p_validacion"],
            historial=json.loads(str(c["historial"])),
            segundos_por_epoca=float(c["segundos"]),
            parametros=int(c["parametros"]),
        )


# ---------------------------------------------------------------------------------------------
# Exportación según el contrato §5
# ---------------------------------------------------------------------------------------------


def _exportar_particion(
    modelo,
    x: Entradas,
    exp: Experimento,
    ids: np.ndarray,
    k: int,
    estadisticas: dict,
    destino: Path,
) -> None:
    """Guarda el codificador (hasta la capa 'embedding') y los vectores de TODOS los ejemplos."""
    destino.mkdir(parents=True, exist_ok=True)
    codificador = keras.Model(modelo.inputs, modelo.get_layer("embedding").output)
    codificador.save(destino / f"particion_{k}.keras")

    todos = np.arange(len(ids))
    z = codificador.predict(_tomar(x, exp.claves_entrada, todos), verbose=0, batch_size=256)
    np.savez_compressed(
        destino / f"vectores_particion_{k}.npz", id_ejemplo=ids, z=z.astype(np.float32)
    )
    if estadisticas:
        np.savez(
            destino / f"estandarizacion_particion_{k}.npz",
            **{f"{clave}_media": media for clave, (media, _) in estadisticas.items()},
            **{f"{clave}_desviacion": desv for clave, (_, desv) in estadisticas.items()},
        )


# ---------------------------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------------------------


def _intervalo(grupos, estadistico, cfg) -> dict:
    """Estimación con intervalo por bootstrap de estrellas; `None` si el estrato es muy pequeño."""
    try:
        r = estadistica.bootstrap_por_estrella(
            grupos,
            estadistico,
            cfg["bootstrap_remuestreos"],
            cfg["bootstrap_semilla"],
            tuple(cfg["ic_percentiles"]),
        )
    except ValueError:
        return {"estimacion": None, "ic_inf": None, "ic_sup": None}
    return {
        "estimacion": round(r.estimacion, 4),
        "ic_inf": round(r.ic_inf, 4),
        "ic_sup": round(r.ic_sup, 4),
    }


def _acierto_fp_por_bandera(p: np.ndarray, tabla: pd.DataFrame, y: np.ndarray) -> dict:
    """Falsos positivos bien clasificados, por bandera del catálogo (estrato de reporte)."""
    prediccion = p.argmax(axis=1)
    resultado = {}
    for bandera in ("fp_ss", "fp_co", "fp_nt", "fp_ec"):
        del_tipo = (y == 1) & (tabla[bandera].to_numpy() == 1)
        resultado[bandera] = {
            "n": int(del_tipo.sum()),
            "aciertos": int((prediccion[del_tipo] == 1).sum()),
        }
    return resultado


def _auc_por_tercil_mes(p: np.ndarray, tabla: pd.DataFrame, y: np.ndarray, protocolo: dict) -> dict:
    """AUC planeta vs FP por intensidad de la señal (terciles de MES fijados antes de entrenar)."""
    real = y < 2
    mes = tabla["mes"].to_numpy(dtype=float)
    terciles = protocolo.get("terciles_mes", np.nanquantile(mes[real], [1 / 3, 2 / 3]))
    cortes = [-np.inf, *terciles, np.inf]
    resultado = {}
    for j, nombre in enumerate(("bajo", "medio", "alto")):
        en_tercil = real & (mes > cortes[j]) & (mes <= cortes[j + 1])
        suficiente = en_tercil.sum() >= protocolo["metricas"]["min_n_estrato"]
        if suficiente and len(set(y[en_tercil])) == 2:
            valor = round(metricas.auc_planeta_fp(p[en_tercil], y[en_tercil]), 4)
        else:
            valor = None
        resultado[nombre] = {"n": int(en_tercil.sum()), "estimacion": valor}
    return resultado


def evaluar_predicciones(
    p: np.ndarray, tabla: pd.DataFrame, protocolo: dict, con_intervalos: bool = True
) -> dict:
    """Todas las métricas de un conjunto de predicciones alineado con `tabla` (plan, sección 4.3)."""
    cfg = protocolo["metricas"]
    y, grupos = tabla["clase"].to_numpy(), tabla["kepid"].to_numpy()
    n_intervalos = cfg["ece_intervalos"]
    funciones = {
        "auc_planeta_fp": lambda i: metricas.auc_planeta_fp(p[i], y[i]),
        "nll": lambda i: metricas.nll(p[i], y[i]),
        "brier": lambda i: metricas.brier(p[i], y[i]),
        "ece_top": lambda i: metricas.ece_top(p[i], y[i], n_intervalos),
        "exactitud": lambda i: metricas.exactitud(p[i], y[i]),
    }
    todos = np.arange(len(y))
    globales = {}
    for nombre, funcion in funciones.items():
        if con_intervalos:
            globales[nombre] = _intervalo(grupos, funcion, cfg)
        else:
            globales[nombre] = {"estimacion": round(float(funcion(todos)), 4)}

    real = y < 2
    ece_q = metricas.ece_binario(metricas.q_planeta(p[real]), y[real] == 0, n_intervalos)
    globales["ece_binario_q"] = {"estimacion": round(ece_q, 4)}

    return {
        "n": int(len(y)),
        "matriz_confusion": metricas.matriz_confusion(p, y).tolist(),
        "global": globales,
        # Estratos de reporte: nunca son entradas del modelo.
        "acierto_fp_por_bandera": _acierto_fp_por_bandera(p, tabla, y),
        "auc_por_tercil_mes": _auc_por_tercil_mes(p, tabla, y, protocolo),
    }


def _resumen_de_semilla(p_prueba, p_val, y, resultados: list[_ResultadoParticion]) -> dict:
    return {
        "auc_planeta_fp": round(metricas.auc_planeta_fp(p_prueba, y), 4),
        "nll": round(metricas.nll(p_prueba, y), 4),
        "nll_validacion": round(metricas.nll(p_val, y), 4),
        "epocas": [r.historial["epoca_elegida"] for r in resultados],
        "segundos_por_epoca": round(float(np.mean([r.segundos_por_epoca for r in resultados])), 3),
    }


# ---------------------------------------------------------------------------------------------
# Validación cruzada completa
# ---------------------------------------------------------------------------------------------


@dataclass
class _Corrida:
    """Opciones de una corrida de `entrenar_cv` que comparten todas sus particiones."""

    exp: Experimento
    datos: _Datos
    protocolo: dict
    salida: Path
    exportar: Path | None
    guardar_modelos: bool
    presupuesto_s: float | None
    inicio: float
    duraciones: list[float]

    def comprobar_presupuesto(self, semilla: int, k: int) -> None:
        """Lanza `Incompleto` si la siguiente partición ya no cabe en el tiempo disponible."""
        if self.presupuesto_s is None:
            return
        usado = time.perf_counter() - self.inicio
        margen = 1.3 * max(self.duraciones, default=0.0)
        if usado + margen > self.presupuesto_s:
            raise Incompleto(f"{self.exp.nombre}: pausa antes de semilla {semilla}, partición {k}")


def _particion_entrenada(corrida: _Corrida, sep: Separacion, semilla: int) -> _ResultadoParticion:
    """Resultado de una partición: se lee de `parciales/` si ya existe; si no, se entrena."""
    exp, k = corrida.exp, sep.k
    ruta_parcial = corrida.salida / "parciales" / f"semilla{semilla}_particion{k}.npz"
    if ruta_parcial.is_file():
        return _cargar_parcial(ruta_parcial)

    corrida.comprobar_presupuesto(semilla, k)
    inicio = time.perf_counter()
    cfg = corrida.protocolo["entrenamiento"]
    resultado, modelo, x, estadisticas = _entrenar_particion(exp, corrida.datos, sep, semilla, cfg)

    if corrida.guardar_modelos:
        (corrida.salida / "modelos").mkdir(exist_ok=True)
        modelo.save(corrida.salida / "modelos" / f"particion_{k}_semilla_{semilla}.keras")
    es_la_semilla_congelada = semilla == corrida.protocolo["semilla_codificador_congelado"]
    if corrida.exportar is not None and es_la_semilla_congelada:
        _exportar_particion(modelo, x, exp, corrida.datos.ids, k, estadisticas, corrida.exportar)

    _guardar_parcial(ruta_parcial, resultado)
    corrida.duraciones.append(time.perf_counter() - inicio)
    return resultado


def _entrenar_semilla(corrida: _Corrida, semilla: int) -> tuple[np.ndarray, dict]:
    """Las n particiones de una semilla. Guarda sus predicciones; devuelve (p_prueba, resumen)."""
    datos, n = corrida.datos, len(corrida.datos.y)
    n_particiones = corrida.protocolo["particiones"]["n"]
    p_prueba, p_val = np.full((n, N_CLASES), np.nan), np.full((n, N_CLASES), np.nan)
    modelo_de_validacion = np.full(n, -1, dtype=int)
    resultados = []

    for k in range(n_particiones):
        sep = separar(datos.particion, k, n_particiones)
        comprobar_sin_solape(datos.kepid, sep)
        resultado = _particion_entrenada(corrida, sep, semilla)
        p_prueba[sep.prueba] = resultado.p_prueba
        p_val[sep.validacion] = resultado.p_validacion
        modelo_de_validacion[sep.validacion] = k
        resultados.append(resultado)

    np.savez_compressed(
        corrida.salida / f"pred_semilla{semilla}.npz",
        id_ejemplo=datos.ids,
        particion=datos.particion,
        p_prueba=p_prueba.astype(np.float32),
        p_validacion=p_val.astype(np.float32),
        particion_modelo_validacion=modelo_de_validacion,
    )
    ruta_historial = corrida.salida / f"historial_semilla{semilla}.json"
    with open(ruta_historial, "w", encoding="utf-8", newline="\n") as f:
        json.dump([r.historial for r in resultados], f, indent=1)

    resumen = _resumen_de_semilla(p_prueba, p_val, datos.y, resultados)
    resumen["parametros"] = resultados[-1].parametros
    return p_prueba, resumen


def entrenar_cv(
    exp: Experimento,
    entradas: Entradas,
    ejemplos: pd.DataFrame,
    protocolo: dict,
    semillas: Sequence[int],
    salida: str | Path,
    exportar: str | Path | None = None,
    version: str = "v1",
    guardar_modelos: bool = False,
    con_intervalos: bool = True,
    presupuesto_s: float | None = None,
) -> dict:
    """Validación cruzada completa de un experimento. Escribe predicciones, historiales y métricas.

    entradas: dataset completo (incluye 'id_ejemplo').
    exportar: si se da, la semilla congelada del protocolo se exporta como codificador por
        partición en `<exportar>/<version>/`.
    presupuesto_s: si se da, lanza `Incompleto` antes de empezar una partición que ya no cabe.
    """
    salida = Path(salida) / exp.nombre
    salida.mkdir(parents=True, exist_ok=True)
    destino_exportacion = None if exportar is None else Path(exportar) / version
    datos = _preparar_datos(exp, entradas, ejemplos)
    corrida = _Corrida(
        exp=exp,
        datos=datos,
        protocolo=protocolo,
        salida=salida,
        exportar=destino_exportacion,
        guardar_modelos=guardar_modelos,
        presupuesto_s=presupuesto_s,
        inicio=time.perf_counter(),
        duraciones=[],
    )

    resumen = {
        "experimento": exp.nombre,
        "semillas": list(semillas),
        "n_no_disponibles": int((~datos.disponible).sum()),
        "por_semilla": {},
    }
    p_de_la_primera_semilla = None
    for semilla in semillas:
        p_prueba, resumen_semilla = _entrenar_semilla(corrida, semilla)
        resumen["parametros"] = resumen_semilla.pop("parametros")
        resumen["por_semilla"][str(semilla)] = resumen_semilla
        if p_de_la_primera_semilla is None:
            p_de_la_primera_semilla = p_prueba

    por_semilla = resumen["por_semilla"].values()
    aucs = [s["auc_planeta_fp"] for s in por_semilla]
    resumen["auc_media_semillas"] = round(float(np.mean(aucs)), 4)
    resumen["auc_desviacion_semillas"] = round(float(np.std(aucs)), 4)
    resumen["nll_validacion_media"] = round(
        float(np.mean([s["nll_validacion"] for s in por_semilla])), 4
    )
    resumen["semilla_0"] = evaluar_predicciones(
        p_de_la_primera_semilla, datos.tabla, protocolo, con_intervalos
    )

    with open(salida / "metricas.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=1)
    escribir_manifiesto(
        salida / "manifiesto.json",
        {"experimento": exp.nombre, "semillas": list(semillas)},
        extra={"protocolo": protocolo["entrenamiento"]},
    )
    if destino_exportacion is not None:
        escribir_manifiesto(
            destino_exportacion / "manifiesto.json",
            {
                "experimento": exp.nombre,
                "version": version,
                "semilla": protocolo["semilla_codificador_congelado"],
            },
            extra={
                "entradas_modelo": list(exp.claves_entrada),
                "estandarizadas": list(exp.estandarizar),
            },
        )
    return resumen


def guardar_modelo_de_particion(
    exp: Experimento,
    entradas: Entradas,
    ejemplos: pd.DataFrame,
    protocolo: dict,
    k: int,
    semilla: int,
    destino: str | Path,
) -> dict[str, np.ndarray]:
    """Entrena el modelo de UNA partición y lo guarda completo, listo para volver a usarlo.

    Escribe en `destino`:
        particion_<k>.keras                  modelo completo (entradas → 3 probabilidades)
        estandarizacion_particion_<k>.npz    media y desviación de las entradas estandarizadas
    Devuelve sus predicciones de prueba y de validación interna, para comprobar que el modelo
    guardado es el mismo que produjo los resultados publicados (el entrenamiento es determinista).
    """
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    datos = _preparar_datos(exp, entradas, ejemplos)
    sep = separar(datos.particion, k, protocolo["particiones"]["n"])
    comprobar_sin_solape(datos.kepid, sep)
    resultado, modelo, _, estadisticas = _entrenar_particion(
        exp, datos, sep, semilla, protocolo["entrenamiento"]
    )
    modelo.save(destino / f"particion_{k}.keras")
    if estadisticas:
        np.savez(
            destino / f"estandarizacion_particion_{k}.npz",
            **{f"{clave}_media": media for clave, (media, _) in estadisticas.items()},
            **{f"{clave}_desviacion": desv for clave, (_, desv) in estadisticas.items()},
        )
    return {"p_prueba": resultado.p_prueba, "p_validacion": resultado.p_validacion}


# ---------------------------------------------------------------------------------------------
# Pruebas de cordura (plan, sección 4.5)
# ---------------------------------------------------------------------------------------------


def prueba_etiquetas_barajadas(
    exp: Experimento,
    entradas: Entradas,
    ejemplos: pd.DataFrame,
    protocolo: dict,
    salida: str | Path,
) -> float:
    """Con las etiquetas planeta/FP permutadas entre estrellas, el AUC debe quedar cerca de 0,5."""
    rng = np.random.default_rng(99)
    barajado = ejemplos.copy()
    es_koi = barajado["clase"] < 2
    barajado.loc[es_koi, "clase"] = rng.permutation(barajado.loc[es_koi, "clase"].to_numpy())
    exp_barajado = Experimento(
        exp.nombre + "_barajado",
        exp.constructor,
        exp.claves_entrada,
        exp.clave_disponible,
        exp.aumentar,
        exp.estandarizar,
        exp.clave_faltantes,
    )
    resumen = entrenar_cv(
        exp_barajado, entradas, barajado, protocolo, [0], salida, con_intervalos=False
    )
    return resumen["auc_media_semillas"]


def prueba_sobreajuste_lote(
    exp: Experimento, entradas: Entradas, ejemplos: pd.DataFrame, n: int = 32, epocas: int = 300
) -> float:
    """El modelo debe poder memorizar un lote pequeño. Devuelve la exactitud de entrenamiento."""
    datos = _preparar_datos(exp, entradas, ejemplos)
    lote = np.flatnonzero(datos.disponible)[:n]
    x, _ = _estandarizar_con_entrenamiento(exp, entradas, lote)
    x_lote, y_lote = _tomar(x, exp.claves_entrada, lote), datos.y[lote]

    keras.utils.set_random_seed(0)
    modelo = exp.constructor(0)
    modelo.compile(optimizer=keras.optimizers.Adam(3e-3), loss="sparse_categorical_crossentropy")
    modelo.fit(x_lote, y_lote, epochs=epocas, batch_size=n, verbose=0)
    # predict() no aplica dropout: mide si los pesos memorizaron el lote
    return float((modelo.predict(x_lote, verbose=0).argmax(axis=1) == y_lote).mean())
