"""Incertidumbre del sistema (spec 006): MC Dropout frente a ensamble de semillas (E10) y casos al azar (E11).

La incertidumbre solo cubre la cabeza de fusión: los codificadores están congelados.

Uso:
  python -I src/fusion/incertidumbre.py --modelos modelos --dataset data/dataset_temporal.npz --ejemplos compartido/ejemplos.csv \
      --fusion resultados/dl/fusion/e6a_cuatro_ramas --salida resultados/dl/incertidumbre
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from comun import entrenamiento as ent  # noqa: E402
from comun import estadistica, metricas  # noqa: E402
from comun.protocolo import cargar_protocolo, separar  # noqa: E402
from comun.utilidades import escribir_manifiesto  # noqa: E402
from fusion.fusionar import entrada_de_particion  # noqa: E402
from fusion.vectores import RAMAS  # noqa: E402


def entropia(p: np.ndarray) -> np.ndarray:
    return -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=-1)


def informacion_mutua(p_miembros: np.ndarray) -> np.ndarray:
    """Desacuerdo entre miembros: entropía de la media − media de las entropías. (m, n, c) → (n,)."""
    return np.maximum(entropia(p_miembros.mean(axis=0)) - entropia(p_miembros).mean(axis=0), 0.0)


def riesgo_cobertura(confianza: np.ndarray, acierto: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Se retienen primero los casos más confiados. Devuelve cobertura, riesgo (error) y su área (AURC)."""
    orden = np.argsort(-confianza, kind="mergesort")
    errores = np.cumsum(1.0 - acierto[orden])
    n = np.arange(1, len(orden) + 1)
    riesgo = errores / n
    return n / len(orden), riesgo, float(riesgo.mean())


def auroc_deteccion_errores(incertidumbre: np.ndarray, acierto: np.ndarray) -> float:
    return metricas.auc(incertidumbre, acierto == 0)


def predecir_mc_dropout(cabeza: keras.Model, x: np.ndarray, pasadas: int, semilla: int) -> np.ndarray:
    keras.utils.set_random_seed(semilla)
    return np.stack([np.asarray(cabeza(x, training=True)) for _ in range(pasadas)])


def resumen(p: np.ndarray, y: np.ndarray, grupos, cfg, incert: np.ndarray | None = None) -> dict:
    acierto = (p.argmax(1) == y).astype(float)
    u = entropia(p) if incert is None else incert
    _, _, aurc = riesgo_cobertura(-entropia(p), acierto)
    ic = estadistica.bootstrap_por_estrella(grupos, lambda i: metricas.nll(p[i], y[i]), cfg["bootstrap_remuestreos"], cfg["bootstrap_semilla"])
    return {"nll": round(metricas.nll(p, y), 4), "nll_ic": [round(ic.ic_inf, 4), round(ic.ic_sup, 4)],
            "brier": round(metricas.brier(p, y), 4), "ece_top": round(metricas.ece_top(p, y, cfg["ece_intervalos"]), 4),
            "exactitud": round(float(acierto.mean()), 4), "aurc": round(aurc, 4),
            "auroc_deteccion_errores": round(auroc_deteccion_errores(u, acierto), 4),
            "entropia_media_aciertos": round(float(entropia(p)[acierto == 1].mean()), 4),
            "entropia_media_errores": round(float(entropia(p)[acierto == 0].mean()), 4)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modelos", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--ejemplos", type=Path, required=True)
    ap.add_argument("--fusion", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--protocolo", type=Path, default=Path("config/protocolo.json"))
    a = ap.parse_args(argv)
    protocolo = cargar_protocolo(a.protocolo)
    cfg, n_part = protocolo["metricas"], protocolo["particiones"]["n"]
    with np.load(a.dataset, allow_pickle=False) as d:
        dataset = {c: d[c] for c in d.files}
    tabla = ent.alinear(pd.read_csv(a.ejemplos), dataset["id_ejemplo"])
    y, grupos, part = tabla["clase"].to_numpy().astype(int), tabla["kepid"].to_numpy(), tabla["particion"].to_numpy()

    semillas = protocolo["semillas_cabeza_fusion"]
    miembros = np.stack([np.load(a.fusion / f"pred_semilla{s}.npz")["p_prueba"].astype(float) for s in semillas])
    p_det, p_ens = miembros[0], miembros.mean(axis=0)
    pasadas = protocolo["calibracion"]["mc_dropout_pasadas"]
    mc = np.zeros((pasadas, len(y), 3))
    for k in range(n_part):
        sep = separar(part, k, n_part)
        x = entrada_de_particion(a.modelos, k, dataset["id_ejemplo"], dataset, RAMAS, np.flatnonzero(sep.entrenamiento))
        cabeza = keras.saving.load_model(a.fusion / "cabezas" / f"particion_{k}.keras", safe_mode=True)
        assert np.abs(cabeza.predict(x[sep.prueba], verbose=0) - p_det[sep.prueba]).max() < 1e-3
        mc[:, sep.prueba] = predecir_mc_dropout(cabeza, x[sep.prueba], pasadas, 100 + k)
    p_mc = mc.mean(axis=0)

    R = {"determinista_semilla_0": resumen(p_det, y, grupos, cfg),
         "mc_dropout": resumen(p_mc, y, grupos, cfg), "ensamble_5_semillas": resumen(p_ens, y, grupos, cfg),
         "auroc_errores_con_informacion_mutua": {
             "mc_dropout": round(auroc_deteccion_errores(informacion_mutua(mc), (p_mc.argmax(1) == y).astype(float)), 4),
             "ensamble_5_semillas": round(auroc_deteccion_errores(informacion_mutua(miembros), (p_ens.argmax(1) == y).astype(float)), 4)},
         "nota": "La incertidumbre solo cubre la cabeza de fusión: los codificadores están congelados."}
    dif = estadistica.bootstrap_por_estrella(grupos, lambda i: metricas.nll(p_ens[i], y[i]) - metricas.nll(p_mc[i], y[i]),
                                             cfg["bootstrap_remuestreos"], cfg["bootstrap_semilla"])
    R["diferencia_nll_ensamble_menos_mc"] = {"estimacion": round(dif.estimacion, 4), "ic_inf": round(dif.ic_inf, 4), "ic_sup": round(dif.ic_sup, 4)}

    # ---------- E11: casos al azar con semilla fija ----------
    rng = np.random.default_rng(protocolo["muestras_al_azar"]["semilla"])
    acierto = p_ens.argmax(1) == y
    elegidos = np.concatenate([rng.choice(np.flatnonzero(~acierto), min(10, int((~acierto).sum())), replace=False),
                               rng.choice(np.flatnonzero(acierto), 10, replace=False)])
    fijo = np.flatnonzero(dataset["id_ejemplo"].astype(str) == "K00022.01|transito")
    casos = pd.DataFrame({"id_ejemplo": dataset["id_ejemplo"].astype(str), "clase_real": y, "prediccion": p_ens.argmax(1),
                          "p_planeta": p_ens[:, 0].round(3), "p_fp": p_ens[:, 1].round(3), "p_sin_senal": p_ens[:, 2].round(3),
                          "entropia": entropia(p_ens).round(3), "informacion_mutua": informacion_mutua(miembros).round(4)})
    casos = casos.iloc[np.r_[elegidos, fijo]].assign(
        seleccion=["error al azar"] * int(min(10, (~acierto).sum())) + ["acierto al azar"] * 10 + ["caso fijo de la demo"] * len(fijo))

    a.salida.mkdir(parents=True, exist_ok=True)
    casos.to_csv(a.salida / "casos_e11.csv", index=False, lineterminator="\n")
    np.savez_compressed(a.salida / "predicciones.npz", id_ejemplo=np.array([str(i) for i in dataset["id_ejemplo"]], dtype="U"),
                        p_determinista=p_det.astype(np.float32), p_mc_dropout=p_mc.astype(np.float32), p_ensamble=p_ens.astype(np.float32),
                        im_mc_dropout=informacion_mutua(mc).astype(np.float32), im_ensamble=informacion_mutua(miembros).astype(np.float32))
    (a.salida / "metricas.json").write_text(json.dumps(R, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    escribir_manifiesto(a.salida / "manifiesto.json", vars(a))
    for k in ("determinista_semilla_0", "mc_dropout", "ensamble_5_semillas"):
        print(k, R[k])
    print(R["auroc_errores_con_informacion_mutua"], R["diferencia_nll_ensamble_menos_mc"])
    print(casos[casos.seleccion == "caso fijo de la demo"].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
