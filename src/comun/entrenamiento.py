"""Motor de experimentos: entrena, evalúa y exporta cualquier rama con el mismo protocolo (spec 003).

Para cada semilla y cada partición de prueba k:
  entrenamiento = particiones distintas de k y de (k+1) mod n; validación interna = (k+1) mod n; prueba = k.
La validación interna decide la parada temprana. La prueba solo se usa para predecir al final.
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
from comun.protocolo import comprobar_sin_solape, separar  # noqa: E402
from comun.utilidades import escribir_manifiesto  # noqa: E402

Entradas = dict[str, np.ndarray]


class Incompleto(RuntimeError):
    """Se agotó el presupuesto de tiempo; lo ya entrenado quedó guardado y la corrida se puede reanudar."""


@dataclass(frozen=True)
class Experimento:
    nombre: str
    constructor: Callable[[int], keras.Model]      # semilla → modelo sin compilar, con una capa "embedding"
    claves_entrada: tuple[str, ...]                # nombres de las entradas del modelo = claves del dataset
    clave_disponible: str | None = None            # p. ej. "disponible_transitos"
    aumentar: Callable[[Entradas, np.random.Generator], Entradas] | None = None
    estandarizar: tuple[str, ...] = ()             # claves que se estandarizan con el entrenamiento de cada partición
    clave_faltantes: str | None = None             # máscara (1 = falta) que acompaña a las claves estandarizadas


def alinear(ejemplos: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    """Filas de ejemplos.csv en el orden de `ids`. Falla si un id no existe o está prerregistrado."""
    tabla = ejemplos.set_index("id_ejemplo").loc[[str(i) for i in ids]].reset_index()
    if tabla["prerregistrado"].astype(bool).any():
        raise ValueError("El dataset contiene ejemplos prerregistrados")
    return tabla


def estadisticas_estandarizacion(x: np.ndarray, faltantes: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Media y desviación por columna, sin contar los valores faltantes."""
    x = np.asarray(x, dtype=float)
    presente = np.ones_like(x, dtype=bool) if faltantes is None else ~np.asarray(faltantes, dtype=bool)
    n = np.maximum(presente.sum(axis=0), 1)
    media = np.where(presente, x, 0.0).sum(axis=0) / n
    var = (np.where(presente, x - media, 0.0) ** 2).sum(axis=0) / n
    desv = np.sqrt(var)
    return media, np.where(desv < 1e-6, 1.0, desv)


def aplicar_estandarizacion(x: np.ndarray, media: np.ndarray, desv: np.ndarray,
                            faltantes: np.ndarray | None = None) -> np.ndarray:
    """(x − media) / desviación; los faltantes quedan en 0 (su aviso viaja en la máscara)."""
    z = (np.asarray(x, dtype=float) - media) / desv
    if faltantes is not None:
        z = np.where(np.asarray(faltantes, dtype=bool), 0.0, z)
    return z.astype(np.float32)


def pesos_por_clase(y: np.ndarray, n_clases: int = 3) -> dict[int, float]:
    return {c: float(len(y) / (n_clases * max(int((y == c).sum()), 1))) for c in range(n_clases)}


def _tomar(entradas: Entradas, claves: Sequence[str], idx: np.ndarray) -> Entradas:
    return {c: entradas[c][idx] for c in claves}


def entrenar_una(modelo: keras.Model, x_ent: Entradas, y_ent: np.ndarray, x_val: Entradas, y_val: np.ndarray,
                 cfg: dict, aumentar=None, rng: np.random.Generator | None = None,
                 usar_pesos_de_clase: bool = True) -> tuple[keras.Model, dict]:
    """Bucle de épocas con parada temprana por NLL de validación; restaura los mejores pesos."""
    modelo.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=cfg["tasa_aprendizaje"], weight_decay=cfg["weight_decay"]),
        loss="sparse_categorical_crossentropy")
    pesos = pesos_por_clase(y_ent) if usar_pesos_de_clase else None
    rng = rng or np.random.default_rng(0)
    mejor, espera, mejores_pesos = np.inf, 0, modelo.get_weights()
    hist = {"perdida_entrenamiento": [], "perdida_validacion": [], "segundos_por_epoca": []}
    for _ in range(cfg["epocas_max"]):
        x = aumentar(x_ent, rng) if aumentar else x_ent
        t0 = time.perf_counter()
        h = modelo.fit(x, y_ent, epochs=1, batch_size=cfg["lote"], shuffle=True, verbose=0, class_weight=pesos)
        hist["segundos_por_epoca"].append(time.perf_counter() - t0)
        perdida_val = float(modelo.evaluate(x_val, y_val, verbose=0, batch_size=256))
        hist["perdida_entrenamiento"].append(float(h.history["loss"][0]))
        hist["perdida_validacion"].append(perdida_val)
        if perdida_val < mejor - 1e-4:
            mejor, espera, mejores_pesos = perdida_val, 0, modelo.get_weights()
        else:
            espera += 1
            if espera >= cfg["paciencia"]:
                break
    modelo.set_weights(mejores_pesos)
    hist["epoca_elegida"] = int(np.argmin(hist["perdida_validacion"])) + 1
    hist["mejor_perdida_validacion"] = float(mejor)
    return modelo, hist


def _intervalo(grupos, fn, cfg) -> dict:
    try:
        r = estadistica.bootstrap_por_estrella(grupos, fn, cfg["bootstrap_remuestreos"], cfg["bootstrap_semilla"],
                                               tuple(cfg["ic_percentiles"]))
        return {"estimacion": round(r.estimacion, 4), "ic_inf": round(r.ic_inf, 4), "ic_sup": round(r.ic_sup, 4)}
    except ValueError:
        return {"estimacion": None, "ic_inf": None, "ic_sup": None}


def evaluar_predicciones(p: np.ndarray, tabla: pd.DataFrame, protocolo: dict, con_intervalos: bool = True) -> dict:
    """Todas las métricas de un conjunto de predicciones alineado con `tabla` (plan, sección 4.3)."""
    cfg = protocolo["metricas"]
    y, grupos = tabla["clase"].to_numpy(), tabla["kepid"].to_numpy()
    nb = cfg["ece_intervalos"]
    funciones = {
        "auc_planeta_fp": lambda i: metricas.auc_planeta_fp(p[i], y[i]),
        "nll": lambda i: metricas.nll(p[i], y[i]),
        "brier": lambda i: metricas.brier(p[i], y[i]),
        "ece_top": lambda i: metricas.ece_top(p[i], y[i], nb),
        "exactitud": lambda i: metricas.exactitud(p[i], y[i]),
    }
    todos = np.arange(len(y))
    R: dict = {"n": int(len(y)), "matriz_confusion": metricas.matriz_confusion(p, y).tolist(), "global": {}}
    for nombre, fn in funciones.items():
        R["global"][nombre] = _intervalo(grupos, fn, cfg) if con_intervalos else {"estimacion": round(float(fn(todos)), 4)}
    real = y < 2
    R["global"]["ece_binario_q"] = {"estimacion": round(metricas.ece_binario(metricas.q_planeta(p[real]), y[real] == 0, nb), 4)}
    # Estratos de reporte (nunca entradas del modelo).
    pred = p.argmax(axis=1)
    R["acierto_fp_por_bandera"] = {}
    for bandera in ("fp_ss", "fp_co", "fp_nt", "fp_ec"):
        m = (y == 1) & (tabla[bandera].to_numpy() == 1)
        R["acierto_fp_por_bandera"][bandera] = {"n": int(m.sum()), "aciertos": int((pred[m] == 1).sum())}
    R["auc_por_tercil_mes"] = {}
    mes = tabla["mes"].to_numpy(dtype=float)
    cortes = [-np.inf, *protocolo.get("terciles_mes", np.nanquantile(mes[real], [1 / 3, 2 / 3])), np.inf]
    for j, nombre in enumerate(("bajo", "medio", "alto")):
        m = real & (mes > cortes[j]) & (mes <= cortes[j + 1])
        if m.sum() < cfg["min_n_estrato"] or len(set(y[m])) < 2:
            R["auc_por_tercil_mes"][nombre] = {"n": int(m.sum()), "estimacion": None}
        else:
            R["auc_por_tercil_mes"][nombre] = {"n": int(m.sum()), "estimacion": round(metricas.auc_planeta_fp(p[m], y[m]), 4)}
    return R


def entrenar_cv(exp: Experimento, entradas: Entradas, ejemplos: pd.DataFrame, protocolo: dict,
                semillas: Sequence[int], salida: str | Path, exportar: str | Path | None = None,
                version: str = "v1", guardar_modelos: bool = False, con_intervalos: bool = True,
                presupuesto_s: float | None = None) -> dict:
    """Validación cruzada completa de un experimento. Escribe predicciones, historiales y métricas.

    entradas: dataset completo (incluye 'id_ejemplo'). Si `exportar` se da, la semilla congelada del
    protocolo se exporta como codificador por partición según el contrato §5.
    Cada partición terminada se guarda en `parciales/`; si se vuelve a llamar, no se repite. Con
    `presupuesto_s`, se lanza `Incompleto` antes de empezar una partición que ya no cabe en el tiempo.
    """
    inicio, duraciones = time.perf_counter(), []
    salida = Path(salida) / exp.nombre
    salida.mkdir(parents=True, exist_ok=True)
    ids = np.array([str(i) for i in entradas["id_ejemplo"]], dtype="U")
    tabla = alinear(ejemplos, ids)
    y = tabla["clase"].to_numpy().astype(int)
    particion, kepid = tabla["particion"].to_numpy(), tabla["kepid"].to_numpy()
    n_part, cfg = protocolo["particiones"]["n"], protocolo["entrenamiento"]
    disponible = np.ones(len(y), bool) if exp.clave_disponible is None else entradas[exp.clave_disponible].astype(bool)
    semilla_congelada = protocolo["semilla_codificador_congelado"]
    resumen = {"experimento": exp.nombre, "semillas": list(semillas), "n_no_disponibles": int((~disponible).sum()),
               "por_semilla": {}}
    preds = []
    for semilla in semillas:
        p_prueba = np.full((len(y), 3), np.nan)
        p_val = np.full((len(y), 3), np.nan)
        modelo_val = np.full(len(y), -1, dtype=int)
        historiales, segundos, n_param = [], [], None
        for k in range(n_part):
            sep = separar(particion, k, n_part)
            comprobar_sin_solape(kepid, sep)
            cache = salida / "parciales" / f"semilla{semilla}_particion{k}.npz"
            if cache.is_file():
                with np.load(cache, allow_pickle=False) as c:
                    p_prueba[sep.prueba], p_val[sep.validacion] = c["p_prueba"], c["p_validacion"]
                    historiales.append(json.loads(str(c["historial"])))
                    segundos.append(float(c["segundos"]))
                    n_param = int(c["parametros"])
                modelo_val[sep.validacion] = k
                continue
            if presupuesto_s is not None:
                usado = time.perf_counter() - inicio
                if usado + 1.3 * (max(duraciones) if duraciones else 0.0) > presupuesto_s:
                    raise Incompleto(f"{exp.nombre}: pausa antes de semilla {semilla}, partición {k}")
            t_particion = time.perf_counter()
            i_ent, i_val = np.flatnonzero(sep.entrenamiento & disponible), np.flatnonzero(sep.validacion & disponible)
            i_pru = np.flatnonzero(sep.prueba)
            x = dict(entradas)
            estadisticas = {}
            for clave in exp.estandarizar:
                falt = entradas[exp.clave_faltantes] if exp.clave_faltantes else None
                media, desv = estadisticas_estandarizacion(entradas[clave][i_ent], None if falt is None else falt[i_ent])
                x[clave] = aplicar_estandarizacion(entradas[clave], media, desv, falt)
                estadisticas[clave] = (media, desv)
            keras.utils.set_random_seed(1000 * semilla + k)
            modelo = exp.constructor(1000 * semilla + k)
            n_param = int(modelo.count_params())
            rng = np.random.default_rng([semilla, k])
            modelo, hist = entrenar_una(modelo, _tomar(x, exp.claves_entrada, i_ent), y[i_ent],
                                        _tomar(x, exp.claves_entrada, i_val), y[i_val], cfg, exp.aumentar, rng)
            historiales.append({"particion": k, **hist})
            segundos.append(float(np.mean(hist["segundos_por_epoca"][1:] or hist["segundos_por_epoca"])))
            a_priori = np.bincount(y[i_ent], minlength=3) / len(i_ent)
            for destino, idx in ((p_prueba, i_pru), (p_val, np.flatnonzero(sep.validacion))):
                destino[idx] = a_priori                      # ejemplos no disponibles: probabilidad a priori
                ok = idx[disponible[idx]]
                if len(ok):
                    destino[ok] = modelo.predict(_tomar(x, exp.claves_entrada, ok), verbose=0, batch_size=256)
            modelo_val[sep.validacion] = k
            if guardar_modelos:
                (salida / "modelos").mkdir(exist_ok=True)
                modelo.save(salida / "modelos" / f"particion_{k}_semilla_{semilla}.keras")
            if exportar is not None and semilla == semilla_congelada:
                _exportar_particion(modelo, x, exp, ids, k, estadisticas, Path(exportar) / version)
            cache.parent.mkdir(exist_ok=True)
            np.savez(cache, p_prueba=p_prueba[sep.prueba], p_validacion=p_val[sep.validacion],
                     historial=np.array(json.dumps(historiales[-1])), segundos=segundos[-1], parametros=n_param)
            duraciones.append(time.perf_counter() - t_particion)
        np.savez_compressed(salida / f"pred_semilla{semilla}.npz", id_ejemplo=ids, particion=particion,
                            p_prueba=p_prueba.astype(np.float32), p_validacion=p_val.astype(np.float32),
                            particion_modelo_validacion=modelo_val)
        with open(salida / f"historial_semilla{semilla}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(historiales, f, indent=1)
        preds.append(p_prueba)
        resumen["por_semilla"][str(semilla)] = {
            "auc_planeta_fp": round(metricas.auc_planeta_fp(p_prueba, y), 4), "nll": round(metricas.nll(p_prueba, y), 4),
            "nll_validacion": round(metricas.nll(p_val, y), 4),
            "epocas": [h["epoca_elegida"] for h in historiales], "segundos_por_epoca": round(float(np.mean(segundos)), 3)}
        resumen["parametros"] = n_param
    aucs = [v["auc_planeta_fp"] for v in resumen["por_semilla"].values()]
    resumen["auc_media_semillas"] = round(float(np.mean(aucs)), 4)
    resumen["auc_desviacion_semillas"] = round(float(np.std(aucs)), 4)
    resumen["nll_validacion_media"] = round(float(np.mean([v["nll_validacion"] for v in resumen["por_semilla"].values()])), 4)
    resumen["semilla_0"] = evaluar_predicciones(preds[0], tabla, protocolo, con_intervalos)
    with open(salida / "metricas.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=1)
    escribir_manifiesto(salida / "manifiesto.json", {"experimento": exp.nombre, "semillas": list(semillas)},
                        extra={"protocolo": protocolo["entrenamiento"]})
    if exportar is not None:
        escribir_manifiesto(Path(exportar) / version / "manifiesto.json",
                            {"experimento": exp.nombre, "version": version, "semilla": semilla_congelada},
                            extra={"entradas_modelo": list(exp.claves_entrada), "estandarizadas": list(exp.estandarizar)})
    return resumen


def _exportar_particion(modelo, x: Entradas, exp: Experimento, ids: np.ndarray, k: int, estadisticas: dict,
                        destino: Path) -> None:
    """Contrato §5: codificador hasta la capa 'embedding' y vectores de TODOS los ejemplos."""
    destino.mkdir(parents=True, exist_ok=True)
    codificador = keras.Model(modelo.inputs, modelo.get_layer("embedding").output)
    codificador.save(destino / f"particion_{k}.keras")
    z = codificador.predict(_tomar(x, exp.claves_entrada, np.arange(len(ids))), verbose=0, batch_size=256)
    np.savez_compressed(destino / f"vectores_particion_{k}.npz", id_ejemplo=ids, z=z.astype(np.float32))
    if estadisticas:
        np.savez(destino / f"estandarizacion_particion_{k}.npz",
                 **{f"{c}_media": m for c, (m, _) in estadisticas.items()},
                 **{f"{c}_desviacion": d for c, (_, d) in estadisticas.items()})


def prueba_etiquetas_barajadas(exp: Experimento, entradas: Entradas, ejemplos: pd.DataFrame, protocolo: dict,
                               salida: str | Path) -> float:
    """Cordura: con las etiquetas de planeta/FP permutadas entre estrellas, el AUC debe quedar cerca de 0,5."""
    rng = np.random.default_rng(99)
    barajado = ejemplos.copy()
    es_koi = barajado["clase"] < 2
    barajado.loc[es_koi, "clase"] = rng.permutation(barajado.loc[es_koi, "clase"].to_numpy())
    r = entrenar_cv(Experimento(exp.nombre + "_barajado", exp.constructor, exp.claves_entrada, exp.clave_disponible,
                                exp.aumentar, exp.estandarizar, exp.clave_faltantes),
                    entradas, barajado, protocolo, [0], salida, con_intervalos=False)
    return r["auc_media_semillas"]


def prueba_sobreajuste_lote(exp: Experimento, entradas: Entradas, ejemplos: pd.DataFrame, n: int = 32,
                            epocas: int = 300) -> float:
    """Cordura: el modelo debe poder memorizar un lote pequeño. Devuelve la exactitud de entrenamiento."""
    ids = np.array([str(i) for i in entradas["id_ejemplo"]])
    tabla = alinear(ejemplos, ids)
    disponible = np.ones(len(ids), bool) if exp.clave_disponible is None else entradas[exp.clave_disponible].astype(bool)
    idx = np.flatnonzero(disponible)[:n]
    y = tabla["clase"].to_numpy()[idx]
    x = dict(entradas)
    for clave in exp.estandarizar:
        falt = entradas[exp.clave_faltantes] if exp.clave_faltantes else None
        media, desv = estadisticas_estandarizacion(entradas[clave][idx], None if falt is None else falt[idx])
        x[clave] = aplicar_estandarizacion(entradas[clave], media, desv, falt)
    keras.utils.set_random_seed(0)
    modelo = exp.constructor(0)
    modelo.compile(optimizer=keras.optimizers.Adam(3e-3), loss="sparse_categorical_crossentropy")
    xb = _tomar(x, exp.claves_entrada, idx)
    modelo.fit(xb, y, epochs=epocas, batch_size=n, verbose=0)
    # training=False: sin dropout, mide si los pesos memorizaron el lote
    return float((modelo.predict(xb, verbose=0).argmax(axis=1) == y).mean())
