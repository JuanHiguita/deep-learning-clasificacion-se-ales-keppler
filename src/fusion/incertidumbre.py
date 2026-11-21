"""Incertidumbre del sistema (spec 006): MC Dropout frente a ensamble (E10) y casos al azar (E11).

Se comparan tres formas de predecir con la cabeza de fusión:
    determinista   una cabeza (semilla 0), dropout apagado
    MC Dropout     la misma cabeza, dropout ENCENDIDO, promedio de varias pasadas
    ensamble       promedio de las cabezas de 5 semillas

La incertidumbre solo cubre la cabeza de fusión: los codificadores están congelados.

Uso:
  python -I src/fusion/incertidumbre.py --modelos modelos --dataset data/dataset_temporal.npz \
      --ejemplos compartido/ejemplos.csv --fusion resultados/dl/fusion/e6a_cuatro_ramas \
      --salida resultados/dl/incertidumbre
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
from comun.metricas import entropia, informacion_mutua, riesgo_cobertura  # noqa: E402
from comun.protocolo import cargar_protocolo, separar  # noqa: E402
from comun.utilidades import escribir_manifiesto  # noqa: E402
from fusion.fusionar import entrada_de_particion  # noqa: E402
from fusion.vectores import RAMAS  # noqa: E402

NOTA = "La incertidumbre solo cubre la cabeza de fusión: los codificadores están congelados."
CASO_FIJO = "K00022.01|transito"  # el caso que la demo mostraba; se reporta siempre
N_CASOS = 10  # errores y aciertos al azar que se muestran en E11
N_CLASES = 3
SEMILLA_MC = 100  # semilla de las pasadas de MC Dropout (se le suma la partición)


# ---------------------------------------------------------------------------------------------
# Medidas de incertidumbre
# ---------------------------------------------------------------------------------------------


def auroc_deteccion_errores(incertidumbre: np.ndarray, acierto: np.ndarray) -> float:
    """¿La incertidumbre señala los errores? 0,5 = no los distingue; 1 = los ordena perfecto."""
    return metricas.auc(incertidumbre, acierto == 0)


def predecir_mc_dropout(
    cabeza: keras.Model, x: np.ndarray, pasadas: int, semilla: int
) -> np.ndarray:
    """`pasadas` predicciones con el dropout encendido. Devuelve (pasadas, n, clases)."""
    keras.utils.set_random_seed(semilla)
    return np.stack([np.asarray(cabeza(x, training=True)) for _ in range(pasadas)])


def _aciertos(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (p.argmax(axis=1) == y).astype(float)


def resumen(p: np.ndarray, y: np.ndarray, grupos, cfg, incert: np.ndarray | None = None) -> dict:
    """Calidad de las probabilidades y utilidad de su incertidumbre para un método."""
    acierto = _aciertos(p, y)
    incertidumbre = entropia(p) if incert is None else incert
    _, _, aurc = riesgo_cobertura(-entropia(p), acierto)
    ic_nll = estadistica.bootstrap_por_estrella(
        grupos,
        lambda i: metricas.nll(p[i], y[i]),
        cfg["bootstrap_remuestreos"],
        cfg["bootstrap_semilla"],
    )
    return {
        "nll": round(metricas.nll(p, y), 4),
        "nll_ic": [round(ic_nll.ic_inf, 4), round(ic_nll.ic_sup, 4)],
        "brier": round(metricas.brier(p, y), 4),
        "ece_top": round(metricas.ece_top(p, y, cfg["ece_intervalos"]), 4),
        "exactitud": round(float(acierto.mean()), 4),
        "aurc": round(aurc, 4),
        "auroc_deteccion_errores": round(auroc_deteccion_errores(incertidumbre, acierto), 4),
        "entropia_media_aciertos": round(float(entropia(p)[acierto == 1].mean()), 4),
        "entropia_media_errores": round(float(entropia(p)[acierto == 0].mean()), 4),
    }


# ---------------------------------------------------------------------------------------------
# E10: MC Dropout frente a ensamble
# ---------------------------------------------------------------------------------------------


def cargar_miembros(fusion: Path, semillas) -> np.ndarray:
    """Predicciones de prueba de la cabeza de cada semilla: (semillas, n, clases)."""
    return np.stack(
        [np.load(fusion / f"pred_semilla{s}.npz")["p_prueba"].astype(float) for s in semillas]
    )


def pasadas_mc_dropout(
    modelos: Path,
    fusion: Path,
    dataset: dict,
    particion,
    p_determinista,
    protocolo: dict,
    versiones: dict | None = None,
) -> np.ndarray:
    """Pasadas de MC Dropout de la cabeza de la semilla 0 sobre su prueba: (pasadas, n, clases)."""
    n_part = protocolo["particiones"]["n"]
    pasadas = protocolo["calibracion"]["mc_dropout_pasadas"]
    mc = np.zeros((pasadas, len(particion), N_CLASES))
    for k in range(n_part):
        sep = separar(particion, k, n_part)
        i_ent = np.flatnonzero(sep.entrenamiento)
        x = entrada_de_particion(
            modelos, k, dataset["id_ejemplo"], dataset, RAMAS, i_ent, versiones
        )
        ruta = fusion / "cabezas" / f"particion_{k}.keras"
        # La capa Dropout toma su semilla al crearse (es decir, al cargar el modelo): hay que
        # fijar la semilla ANTES de cargar para que las pasadas se puedan regenerar.
        keras.utils.set_random_seed(SEMILLA_MC + k)
        cabeza = keras.saving.load_model(ruta, safe_mode=True)
        # la cabeza guardada debe reproducir las predicciones publicadas
        diferencia = np.abs(cabeza.predict(x[sep.prueba], verbose=0) - p_determinista[sep.prueba])
        if diferencia.max() >= 1e-3:
            raise RuntimeError(f"La cabeza de la partición {k} no reproduce sus predicciones")
        mc[:, sep.prueba] = predecir_mc_dropout(cabeza, x[sep.prueba], pasadas, SEMILLA_MC + k)
    return mc


def experimento_e10(miembros: np.ndarray, mc: np.ndarray, y, grupos, cfg: dict) -> dict:
    """Compara determinista, MC Dropout y ensamble; diferencia pareada de NLL con intervalo."""
    p_det, p_mc, p_ens = miembros[0], mc.mean(axis=0), miembros.mean(axis=0)
    diferencia = estadistica.bootstrap_por_estrella(
        grupos,
        lambda i: metricas.nll(p_ens[i], y[i]) - metricas.nll(p_mc[i], y[i]),
        cfg["bootstrap_remuestreos"],
        cfg["bootstrap_semilla"],
    )
    auroc_im_mc = auroc_deteccion_errores(informacion_mutua(mc), _aciertos(p_mc, y))
    auroc_im_ens = auroc_deteccion_errores(informacion_mutua(miembros), _aciertos(p_ens, y))
    return {
        "determinista_semilla_0": resumen(p_det, y, grupos, cfg),
        "mc_dropout": resumen(p_mc, y, grupos, cfg),
        "ensamble_5_semillas": resumen(p_ens, y, grupos, cfg),
        "auroc_errores_con_informacion_mutua": {
            "mc_dropout": round(auroc_im_mc, 4),
            "ensamble_5_semillas": round(auroc_im_ens, 4),
        },
        "nota": NOTA,
        "diferencia_nll_ensamble_menos_mc": {
            "estimacion": round(diferencia.estimacion, 4),
            "ic_inf": round(diferencia.ic_inf, 4),
            "ic_sup": round(diferencia.ic_sup, 4),
        },
    }


# ---------------------------------------------------------------------------------------------
# E11: casos al azar con semilla fija
# ---------------------------------------------------------------------------------------------


def casos_al_azar(ids: np.ndarray, y, miembros: np.ndarray, semilla: int) -> pd.DataFrame:
    """Errores y aciertos del ensamble elegidos al azar (semilla fija), más el caso de la demo.

    Los casos no se escogen a mano: así los errores quedan tan visibles como los aciertos.
    """
    p_ens = miembros.mean(axis=0)
    acierto = p_ens.argmax(axis=1) == y
    rng = np.random.default_rng(semilla)
    n_errores = min(N_CASOS, int((~acierto).sum()))
    errores = rng.choice(np.flatnonzero(~acierto), n_errores, replace=False)
    aciertos = rng.choice(np.flatnonzero(acierto), N_CASOS, replace=False)
    fijo = np.flatnonzero(ids == CASO_FIJO)

    casos = pd.DataFrame(
        {
            "id_ejemplo": ids,
            "clase_real": y,
            "prediccion": p_ens.argmax(axis=1),
            "p_planeta": p_ens[:, 0].round(3),
            "p_fp": p_ens[:, 1].round(3),
            "p_sin_senal": p_ens[:, 2].round(3),
            "entropia": entropia(p_ens).round(3),
            "informacion_mutua": informacion_mutua(miembros).round(4),
        }
    )
    etiquetas = (
        ["error al azar"] * n_errores
        + ["acierto al azar"] * N_CASOS
        + ["caso fijo de la demo"] * len(fijo)
    )
    return casos.iloc[np.r_[errores, aciertos, fijo]].assign(seleccion=etiquetas)


# ---------------------------------------------------------------------------------------------
# Programa
# ---------------------------------------------------------------------------------------------


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
    with np.load(a.dataset, allow_pickle=False) as d:
        dataset = {clave: d[clave] for clave in d.files}
    ids = np.array([str(i) for i in dataset["id_ejemplo"]], dtype="U")
    tabla = ent.alinear(pd.read_csv(a.ejemplos), dataset["id_ejemplo"])
    y = tabla["clase"].to_numpy().astype(int)
    grupos, particion = tabla["kepid"].to_numpy(), tabla["particion"].to_numpy()

    miembros = cargar_miembros(a.fusion, protocolo["semillas_cabeza_fusion"])
    mc = pasadas_mc_dropout(a.modelos, a.fusion, dataset, particion, miembros[0], protocolo)
    e10 = experimento_e10(miembros, mc, y, grupos, protocolo["metricas"])
    casos = casos_al_azar(ids, y, miembros, protocolo["muestras_al_azar"]["semilla"])

    a.salida.mkdir(parents=True, exist_ok=True)
    casos.to_csv(a.salida / "casos_e11.csv", index=False, lineterminator="\n")
    np.savez_compressed(
        a.salida / "predicciones.npz",
        id_ejemplo=ids,
        p_determinista=miembros[0].astype(np.float32),
        p_mc_dropout=mc.mean(axis=0).astype(np.float32),
        p_ensamble=miembros.mean(axis=0).astype(np.float32),
        im_mc_dropout=informacion_mutua(mc).astype(np.float32),
        im_ensamble=informacion_mutua(miembros).astype(np.float32),
    )
    (a.salida / "metricas.json").write_text(
        json.dumps(e10, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    escribir_manifiesto(a.salida / "manifiesto.json", vars(a))

    for metodo in ("determinista_semilla_0", "mc_dropout", "ensamble_5_semillas"):
        print(metodo, e10[metodo])
    print(e10["auroc_errores_con_informacion_mutua"], e10["diferencia_nll_ensamble_menos_mc"])
    print(casos[casos.seleccion == "caso fijo de la demo"].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
