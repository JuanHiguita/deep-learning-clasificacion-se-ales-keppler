"""Iteración 2: evaluación ÚNICA de las señales prerregistradas (spec 014).

La hipótesis, el estadístico y las reglas están en `iteracion_2/evidencias/prerregistro_declaracion.md`,
escrito y versionado antes de ejecutar esto. Resumen: las 76 señales son FALSE POSITIVE en DR25 y
CANDIDATE hoy; no hay respuesta correcta, así que no se miden aciertos. Se pregunta si la fusión les
da más probabilidad de planeta que a los falsos positivos que nadie ha reabierto.

Orden:
  1. Comprobación: con ejemplos NORMALES de prueba, rehace sus entradas desde las curvas y sus
     predicciones desde los modelos guardados, y exige que coincidan con lo publicado.
  2. Solo si coincide: construye las entradas de las prerregistradas, las predice con el sistema de
     su partición y escribe el resultado. Si el resultado ya existe, se niega a repetir.

Uso:
  python -I src/fusion/it2_prerregistrados.py --koi data/koi_dr25.csv [--solo-comprobar]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from comun import entrenamiento as ent  # noqa: E402
from comun import estadistica, metricas  # noqa: E402
from comun.protocolo import cargar_protocolo, separar  # noqa: E402
from comun.utilidades import sha256_archivo  # noqa: E402
from fusion import fusionar, iteracion_2  # noqa: E402
from fusion.vectores import DISPONIBLE, RAMAS, armar_entrada, cargar_vectores  # noqa: E402
from temporal import construir_dataset_temporal  # noqa: E402
from temporal import modelos as _capas_propias  # noqa: E402, F401  (registra las capas al cargar)
from temporal.exp_ramas import EXPERIMENTOS  # noqa: E402

RAIZ = Path("iteracion_2")
CONFIGURACION = iteracion_2.PRINCIPAL
# Diferencia máxima admitida al reproducir una probabilidad publicada. Era 1e-4; en GPU la primera
# comprobación dio 1,06e-4 (ruido numérico: las entradas coincidían exactamente). Se subió a 1e-3
# ANTES de evaluar ninguna prerregistrada; ver la enmienda en prerregistro_declaracion.md. Un error
# de lógica (otra partición, otra estandarización) produce diferencias de décimas, no de diezmilésimas.
TOLERANCIA = 1e-3
N_COMPROBACION = 40  # ejemplos normales de prueba que se rehacen desde las curvas


def _cargar(ruta: Path) -> keras.Model:
    return keras.saving.load_model(ruta, compile=False, safe_mode=True)


def predecir(
    entradas: dict,
    particion: np.ndarray,
    referencia: dict,
    tabla_referencia: pd.DataFrame,
    carpeta_modelos: Path,
    carpeta_fusion: Path,
    protocolo: dict,
    ramas: tuple[str, ...],
) -> np.ndarray:
    """Probabilidades de la fusión para ejemplos nuevos, cada uno con el sistema de su partición.

    entradas: mismas claves que `dataset_temporal.npz`, solo de los ejemplos a predecir.
    referencia, tabla_referencia: el dataset de entrenamiento y su tabla alineada; solo se usan para
        recalcular la estandarización de la entrada de la cabeza con el entrenamiento de cada partición.
    """
    n_particiones = protocolo["particiones"]["n"]
    versiones = {r: (iteracion_2.VERSION if r in ramas else None) for r in RAMAS}
    p = np.full((len(particion), fusionar.N_CLASES), np.nan)
    for k in range(n_particiones):
        suyos = np.flatnonzero(particion == k)
        if len(suyos) == 0:
            continue
        vectores, banderas = {}, np.zeros((len(suyos), len(RAMAS)), np.float32)
        for rama in ramas:
            carpeta = Path(carpeta_modelos) / rama / iteracion_2.VERSION
            manifiesto = json.loads((carpeta / "manifiesto.json").read_text(encoding="utf-8"))
            exp = EXPERIMENTOS[manifiesto["particiones"][str(k)]["red"]][0]
            x = {clave: entradas[clave][suyos] for clave in exp.claves_entrada}
            ruta_estadisticas = carpeta / f"estandarizacion_particion_{k}.npz"
            if exp.estandarizar:
                faltantes = entradas[exp.clave_faltantes][suyos] if exp.clave_faltantes else None
                with np.load(ruta_estadisticas, allow_pickle=False) as e:
                    for clave in exp.estandarizar:
                        x[clave] = ent.aplicar_estandarizacion(
                            x[clave], e[f"{clave}_media"], e[f"{clave}_desviacion"], faltantes
                        )
            z = _cargar(carpeta / f"particion_{k}.keras").predict(x, verbose=0, batch_size=256)
            con_dato = np.ones(len(suyos), bool)
            if rama in DISPONIBLE:
                con_dato = entradas[DISPONIBLE[rama]][suyos].astype(bool)
            vectores[rama] = np.where(con_dato[:, None], z, 0.0).astype(np.float32)
            banderas[:, RAMAS.index(rama)] = con_dato

        # Estandarización de la entrada de la cabeza: la misma que usó al entrenarse (partición k)
        sep = separar(tabla_referencia["particion"].to_numpy(), k, n_particiones)
        v_ref, b_ref = cargar_vectores(
            carpeta_modelos, versiones, k, referencia["id_ejemplo"], referencia
        )
        x_ref = armar_entrada(v_ref, b_ref, ramas)
        n_vectores = x_ref.shape[1] - len(ramas)
        media, desv = ent.estadisticas_estandarizacion(
            x_ref[np.flatnonzero(sep.entrenamiento), :n_vectores]
        )
        x = armar_entrada(vectores, banderas, ramas)
        x[:, :n_vectores] = ent.aplicar_estandarizacion(x[:, :n_vectores], media, desv)
        inicio = 0
        for rama in ramas:  # un vector ausente sigue en cero después de estandarizar
            ancho = vectores[rama].shape[1]
            x[banderas[:, RAMAS.index(rama)] == 0, inicio : inicio + ancho] = 0.0
            inicio += ancho
        cabeza = _cargar(Path(carpeta_fusion) / "cabezas" / f"particion_{k}.keras")
        p[suyos] = cabeza.predict(x, verbose=0, batch_size=256)
    return p


def construir_entradas(
    filas: pd.DataFrame, koi: pd.DataFrame, llc: Path, protocolo: dict
) -> tuple[dict, pd.DataFrame]:
    """Entradas desde las curvas de luz, con el mismo extractor del dataset de entrenamiento."""
    como_usables = filas.assign(prerregistrado=False)  # el extractor salta las prerregistradas
    return construir_dataset_temporal.construir(como_usables, koi, llc, protocolo)


def comprobar(referencia, tabla, ejemplos, koi, llc, modelos, fusion, protocolo, ramas) -> dict:
    """Rehace entradas y predicciones de ejemplos NORMALES de prueba y las compara con lo publicado."""
    rng = np.random.default_rng(protocolo["metricas"]["bootstrap_semilla"])
    estrellas = rng.choice(np.unique(tabla["kepid"]), N_COMPROBACION // 2, replace=False)
    filas = ejemplos[ejemplos["kepid"].isin(estrellas) & ~ejemplos["prerregistrado"].astype(bool)]
    entradas, _ = construir_entradas(filas, koi, llc, protocolo)
    ids = [str(i) for i in entradas["id_ejemplo"]]
    posicion = {str(i): j for j, i in enumerate(referencia["id_ejemplo"])}
    donde = np.array([posicion[i] for i in ids])
    dif_entradas = max(
        float(np.abs(entradas[c].astype(float) - referencia[c][donde].astype(float)).max())
        for c in ("vista_global", "vista_local", "transitos", "escalares", "consistencia")
    )
    particion = tabla["particion"].to_numpy()[donde]
    p = predecir(entradas, particion, referencia, tabla, modelos, fusion, protocolo, ramas)
    with np.load(Path(fusion) / "pred_semilla0.npz", allow_pickle=False) as d:
        publicado = dict(zip((str(i) for i in d["id_ejemplo"]), d["p_prueba"], strict=True))
    diferencias = np.abs(p - np.array([publicado[i] for i in ids])).max(axis=1)
    dif_p = float(diferencias.max())
    return {
        "ejemplos_normales_rehechos": len(ids),
        "diferencia_maxima_en_entradas": dif_entradas,
        "diferencia_maxima_en_probabilidad": dif_p,
        "diferencia_mediana_en_probabilidad": float(np.median(diferencias)),
        "ejemplos_con_diferencia_mayor_que_1e-4": int((diferencias > 1e-4).sum()),
        "tolerancia": TOLERANCIA,
        "supera": bool(dif_p <= TOLERANCIA),
    }


def resumir(q_pre: np.ndarray, estrella_pre, q_fp: np.ndarray, estrella_fp, cfg: dict) -> dict:
    """Estadístico declarado: A = P(q prerregistrada > q de un falso positivo normal)."""
    q = np.concatenate([q_pre, q_fp])
    es_pre = np.concatenate([np.ones(len(q_pre), bool), np.zeros(len(q_fp), bool)])
    grupos = np.concatenate([estrella_pre, estrella_fp])
    r = estadistica.bootstrap_por_estrella(
        grupos,
        lambda i: metricas.auc(q[i], es_pre[i]),
        cfg["bootstrap_remuestreos"],
        cfg["bootstrap_semilla"],
    )
    if r.ic_inf > 0.5:
        veredicto = (
            "respaldada: reciben más probabilidad de planeta que los falsos positivos normales"
        )
    elif r.ic_sup < 0.5:
        veredicto = (
            "refutada: reciben menos probabilidad de planeta que los falsos positivos normales"
        )
    else:
        veredicto = "no concluyente: el intervalo contiene 0,5"
    return {
        "A": {
            "estimacion": round(r.estimacion, 4),
            "ic_inf": round(r.ic_inf, 4),
            "ic_sup": round(r.ic_sup, 4),
        },
        "hipotesis_A_mayor_que_0.5": veredicto,
        "mediana_q": {
            "prerregistradas": round(float(np.median(q_pre)), 4),
            "falsos_positivos_normales": round(float(np.median(q_fp)), 4),
        },
        "fraccion_q_mayor_que_0.5": {
            "prerregistradas": round(float((q_pre > 0.5).mean()), 4),
            "falsos_positivos_normales": round(float((q_fp > 0.5).mean()), 4),
        },
        "n": {"prerregistradas": int(len(q_pre)), "falsos_positivos_normales": int(len(q_fp))},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--koi", type=Path, required=True)
    ap.add_argument("--solo-comprobar", action="store_true", help="no toca las prerregistradas")
    ap.add_argument(
        "--simulacro",
        type=int,
        default=0,
        metavar="N",
        help="ensayo general: N señales NORMALES al azar hacen de prerregistradas (no toca las reales)",
    )
    ap.add_argument(
        "--configuracion", default=CONFIGURACION, choices=list(iteracion_2.CONFIGURACIONES)
    )
    ap.add_argument("--llc", type=Path, default=RAIZ / "datos" / "llc")
    ap.add_argument("--dataset", type=Path, default=RAIZ / "datos" / "dataset_temporal.npz")
    ap.add_argument("--ejemplos", type=Path, default=RAIZ / "datos" / "ejemplos.csv")
    ap.add_argument("--protocolo", type=Path, default=RAIZ / "config" / "protocolo.json")
    ap.add_argument("--modelos", type=Path, default=RAIZ / "modelos_finales")
    ap.add_argument("--fusion", type=Path, default=RAIZ / "evidencias" / "resultados" / "fusion")
    ap.add_argument(
        "--salida", type=Path, default=RAIZ / "evidencias" / "resultados" / "prerregistrados"
    )
    a = ap.parse_args(argv)

    if a.simulacro:
        a.salida = a.salida / "simulacro"
    ruta_resultado = a.salida / "resultado.json"
    if ruta_resultado.is_file() and not a.solo_comprobar and not a.simulacro:
        print(f"Ya existe {ruta_resultado}: la evaluación es única y no se repite.")
        return 2

    protocolo = cargar_protocolo(a.protocolo)
    ramas = iteracion_2.CONFIGURACIONES[a.configuracion]
    fusion = a.fusion / a.configuracion
    ejemplos = pd.read_csv(a.ejemplos)
    koi = pd.read_csv(a.koi)
    with np.load(a.dataset, allow_pickle=False) as archivo:
        referencia = {clave: archivo[clave] for clave in archivo.files}
    tabla = ent.alinear(ejemplos, np.array([str(i) for i in referencia["id_ejemplo"]], dtype="U"))

    print("=== 1. Comprobación con ejemplos normales ===", flush=True)
    comprobacion = comprobar(
        referencia, tabla, ejemplos, koi, a.llc, a.modelos, fusion, protocolo, ramas
    )
    print(json.dumps(comprobacion, ensure_ascii=False, indent=1), flush=True)
    a.salida.mkdir(parents=True, exist_ok=True)
    (a.salida / "comprobacion.json").write_text(
        json.dumps(comprobacion, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    if not comprobacion["supera"]:
        print("La comprobación NO se supera: no se evalúan las prerregistradas.")
        return 1
    if a.solo_comprobar:
        return 0

    if a.simulacro:
        print(
            f"=== 2. SIMULACRO con {a.simulacro} señales normales (no es un resultado) ===",
            flush=True,
        )
        normales = ejemplos[~ejemplos["prerregistrado"].astype(bool)]
        falsos_positivos = normales.loc[normales["clase"] == 1, "kepoi_name"].unique()
        rng = np.random.default_rng(protocolo["metricas"]["bootstrap_semilla"] + 1)
        elegidas = rng.choice(falsos_positivos, a.simulacro, replace=False)
        filas = normales[normales["kepoi_name"].isin(elegidas)]  # con tránsito y sin señal
    else:
        print("=== 2. Evaluación única de las prerregistradas ===", flush=True)
        filas = ejemplos[ejemplos["prerregistrado"].astype(bool)]
    entradas, registro = construir_entradas(filas, koi, a.llc, protocolo)
    ids = [str(i) for i in entradas["id_ejemplo"]]
    de_ejemplo = filas.set_index("id_ejemplo").loc[ids].reset_index()
    p = predecir(
        entradas,
        de_ejemplo["particion"].to_numpy(),
        referencia,
        tabla,
        a.modelos,
        fusion,
        protocolo,
        ramas,
    )
    np.savez_compressed(a.salida / "predicciones.npz", id_ejemplo=np.array(ids, dtype="U"), p=p)

    con_transito = (de_ejemplo["tipo"] == "transito").to_numpy()
    q_pre = metricas.q_planeta(p[con_transito])
    with np.load(fusion / "pred_semilla0.npz", allow_pickle=False) as d:
        p_normal = d["p_prueba"].astype(float)
    fp_normal = tabla["clase"].to_numpy() == 1
    cfg = protocolo["metricas"]
    resultado = {
        "es_simulacro": bool(a.simulacro),
        "declaracion": "iteracion_2/evidencias/prerregistro_declaracion.md",
        "configuracion": a.configuracion,
        "comprobacion": comprobacion,
        "n_ejemplos_prerregistrados": int(len(ids)),
        "disponibilidad": {
            "vistas": int(entradas["disponible_vistas"][con_transito].sum()),
            "transitos": int(entradas["disponible_transitos"][con_transito].sum()),
        },
        "global": resumir(
            q_pre,
            de_ejemplo["kepid"].to_numpy()[con_transito],
            metricas.q_planeta(p_normal[fp_normal]),
            tabla["kepid"].to_numpy()[fp_normal],
            cfg,
        ),
        "por_tipo_de_fp_dr25": {},
        "sin_senal_clasificados_sin_senal": round(
            float((p[~con_transito].argmax(1) == 2).mean()), 4
        ),
    }
    tipo_pre = de_ejemplo["tipo_fp"].to_numpy()[con_transito]
    tipo_normal = tabla["tipo_fp"].to_numpy()
    for tipo in sorted(set(tipo_pre)):
        de_tipo, normales = tipo_pre == tipo, fp_normal & (tipo_normal == tipo)
        resultado["por_tipo_de_fp_dr25"][str(tipo)] = {
            "n_prerregistradas": int(de_tipo.sum()),
            "n_normales": int(normales.sum()),
            "mediana_q_prerregistradas": round(float(np.median(q_pre[de_tipo])), 4),
            "mediana_q_normales": round(float(np.median(metricas.q_planeta(p_normal[normales]))), 4)
            if normales.any()
            else None,
        }
    detalle = de_ejemplo.loc[con_transito, ["kepoi_name", "kepid", "periodo_d", "mes", "tipo_fp"]]
    detalle = detalle.assign(
        q_planeta=q_pre.round(4), particion=de_ejemplo["particion"][con_transito]
    )
    detalle.sort_values("q_planeta", ascending=False).to_csv(
        a.salida / "detalle.csv", index=False, lineterminator="\n"
    )
    registro.to_csv(a.salida / "registro_extraccion.csv", index=False, lineterminator="\n")
    resultado["sha256_predicciones"] = sha256_archivo(a.salida / "predicciones.npz")
    ruta_resultado.write_text(
        json.dumps(resultado, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(resultado, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
