"""Fusión de las cuatro ramas y ablación (spec 005): E6a, E6c, E7 y diagnóstico del apilado.

La cabeza del contrato §6 (Densa 64 → Dropout 0,3 → Densa 3) se entrena sobre vectores CONGELADOS,
con las mismas particiones y validación interna que las ramas. La rama de imagen es la v0 de la demo.

Uso:
  python -I src/fusion/fusionar.py --modelos modelos --dataset data/dataset_temporal.npz --ejemplos compartido/ejemplos.csv \
      --resultados resultados/dl [--presupuesto 150]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from keras import layers  # noqa: E402

from comun import entrenamiento as ent  # noqa: E402
from comun import metricas  # noqa: E402
from comun.protocolo import cargar_protocolo, comprobar_sin_solape, separar  # noqa: E402
from comun.utilidades import escribir_manifiesto, sha256_archivo  # noqa: E402
from fusion.vectores import RAMAS, armar_entrada, cargar_vectores  # noqa: E402

VERSIONES = {"vistas": "v1", "transitos": "v1", "imagen": "v0", "escalares": "v1"}
CONFIGURACIONES = {"e6a_cuatro_ramas": RAMAS,
                   **{f"e7_sin_{r}": tuple(x for x in RAMAS if x != r) for r in RAMAS},
                   **{f"e7_solo_{r}": (r,) for r in RAMAS},
                   # Remedio del apilado (spec 005, RF-8): la cabeza se entrena SOLO con vectores que los
                   # codificadores no vieron (la validación interna), separando un 25 % de estrellas para parar.
                   "e6a_ajuste_cruzado": RAMAS}
EXPERIMENTO_DE_RAMA = {"vistas": "m1_astronet", "imagen": "imagen_v0", "escalares": "m4_escalares"}


def cabeza(dim_entrada: int, semilla: int, dropout: float = 0.3) -> keras.Model:
    keras.utils.set_random_seed(semilla)
    x = keras.Input((dim_entrada,), name="z")
    h = layers.Dropout(dropout, name="dropout_fusion")(layers.Dense(64, activation="relu")(x))
    return keras.Model(x, layers.Dense(3, activation="softmax")(h), name="cabeza_fusion")


def entrada_de_particion(modelos: Path, k: int, ids, dataset: dict, ramas, i_ent) -> np.ndarray:
    """Entrada de la cabeza para la partición k, estandarizada con el entrenamiento (las banderas no)."""
    v, b = cargar_vectores(modelos, VERSIONES, k, ids, dataset)
    x = armar_entrada(v, b, ramas)
    n_vec = x.shape[1] - len(ramas)
    media, desv = ent.estadisticas_estandarizacion(x[i_ent, :n_vec])
    x[:, :n_vec] = ent.aplicar_estandarizacion(x[:, :n_vec], media, desv)
    # un vector ausente debe seguir en cero después de estandarizar
    for j, r in enumerate(ramas):
        ini = sum(v[q].shape[1] for q in ramas[:j])
        x[b[:, RAMAS.index(r)] == 0, ini:ini + v[r].shape[1]] = 0.0
    return x


def entrenar_fusion(nombre: str, ramas, modelos: Path, dataset: dict, tabla: pd.DataFrame, protocolo: dict,
                    salida: Path, limite: float | None = None, inicio: float = 0.0) -> dict | None:
    """Entrena la cabeza para una configuración de ramas, con 5 semillas. Reanudable por semilla."""
    carpeta = salida / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    ids, y = dataset["id_ejemplo"], tabla["clase"].to_numpy().astype(int)
    part, kepid, n = tabla["particion"].to_numpy(), tabla["kepid"].to_numpy(), protocolo["particiones"]["n"]
    cfg = protocolo["entrenamiento"]
    X = {}
    for k in range(n):
        sep = separar(part, k, n)
        comprobar_sin_solape(kepid, sep)
        X[k] = (sep, entrada_de_particion(modelos, k, ids, dataset, ramas, np.flatnonzero(sep.entrenamiento)))
    diag = []
    for semilla in protocolo["semillas_cabeza_fusion"]:
        ruta = carpeta / f"pred_semilla{semilla}.npz"
        if ruta.is_file():
            continue
        if limite is not None and time.perf_counter() - inicio > limite:
            return None
        p_prueba, p_val = np.zeros((len(y), 3)), np.zeros((len(y), 3))
        modelo_val = np.full(len(y), -1)
        for k in range(n):
            sep, x = X[k]
            m = cabeza(x.shape[1], 1000 * semilla + k, cfg["dropout_fusion"])
            if nombre.endswith("ajuste_cruzado"):
                estrellas = np.unique(kepid[sep.validacion])
                parar = set(estrellas[np.arange(len(estrellas)) % 4 == 0].tolist())
                i_par = sep.validacion & np.isin(kepid, list(parar))
                i_ent = sep.validacion & ~i_par
            else:
                i_ent, i_par = sep.entrenamiento, sep.validacion
            m, hist = ent.entrenar_una(m, x[i_ent], y[i_ent], x[i_par], y[i_par],
                                       cfg, None, np.random.default_rng([semilla, k]))
            p_prueba[sep.prueba] = m.predict(x[sep.prueba], verbose=0)
            p_val[sep.validacion] = m.predict(x[sep.validacion], verbose=0)
            modelo_val[sep.validacion] = k
            if semilla == 0:
                (carpeta / "cabezas").mkdir(exist_ok=True)
                m.save(carpeta / "cabezas" / f"particion_{k}.keras")
                diag.append({"particion": k, "epocas": hist["epoca_elegida"],
                             "nll_entrenamiento": round(metricas.nll(m.predict(x[i_ent], verbose=0), y[i_ent]), 4),
                             "nll_validacion": round(metricas.nll(m.predict(x[i_par], verbose=0), y[i_par]), 4)})
        np.savez_compressed(ruta, id_ejemplo=np.array([str(i) for i in ids], dtype="U"), particion=part,
                            p_prueba=p_prueba.astype(np.float32), p_validacion=p_val.astype(np.float32),
                            particion_modelo_validacion=modelo_val)
        if semilla == 0:
            (carpeta / "diagnostico_apilado.json").write_text(json.dumps(diag, indent=1) + "\n", encoding="utf-8")
    preds = [np.load(carpeta / f"pred_semilla{s}.npz")["p_prueba"] for s in protocolo["semillas_cabeza_fusion"]]
    aucs = [metricas.auc_planeta_fp(p, y) for p in preds]
    R = {"configuracion": nombre, "ramas": list(ramas), "dim_entrada": int(X[0][1].shape[1]),
         "auc_por_semilla": [round(a, 4) for a in aucs], "auc_media_semillas": round(float(np.mean(aucs)), 4),
         "auc_desviacion_semillas": round(float(np.std(aucs)), 4),
         "semilla_0": ent.evaluar_predicciones(preds[0], tabla, protocolo, con_intervalos=(nombre == "e6a_cuatro_ramas"))}
    ruta_diag = carpeta / "diagnostico_apilado.json"
    if ruta_diag.is_file():
        d = json.loads(ruta_diag.read_text(encoding="utf-8"))
        ent_, val_ = np.mean([x["nll_entrenamiento"] for x in d]), np.mean([x["nll_validacion"] for x in d])
        R["apilado"] = {"nll_entrenamiento": round(float(ent_), 4), "nll_validacion": round(float(val_), 4),
                        "razon": round(float(val_ / ent_), 2), "alerta": bool(val_ > 2 * ent_)}
    (carpeta / "metricas.json").write_text(json.dumps(R, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return R


def fusion_tardia(resultados: Path, ids: np.ndarray, rama2: str) -> np.ndarray:
    """E6c: promedio de las probabilidades de las cuatro ramas entrenadas por separado (semilla 0).

    Un ejemplo sin imagen usa la probabilidad media de entrenamiento de esa rama (1/3 por clase aquí no
    aplica: se promedia solo sobre las ramas que sí tienen predicción para ese ejemplo).
    """
    ids = np.array([str(i) for i in ids])
    suma, cuenta = np.zeros((len(ids), 3)), np.zeros(len(ids))
    for nombre in (EXPERIMENTO_DE_RAMA["vistas"], rama2, EXPERIMENTO_DE_RAMA["imagen"], EXPERIMENTO_DE_RAMA["escalares"]):
        with np.load(resultados / nombre / "pred_semilla0.npz", allow_pickle=False) as d:
            pos = {str(i): j for j, i in enumerate(d["id_ejemplo"])}
            p = d["p_prueba"]
        for i, ident in enumerate(ids):
            if ident in pos:
                suma[i] += p[pos[ident]]
                cuenta[i] += 1
    return suma / cuenta[:, None]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modelos", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--ejemplos", type=Path, required=True)
    ap.add_argument("--resultados", type=Path, required=True)
    ap.add_argument("--presupuesto", type=float, default=None)
    ap.add_argument("--protocolo", type=Path, default=Path("config/protocolo.json"))
    a = ap.parse_args(argv)
    inicio = time.perf_counter()
    protocolo = cargar_protocolo(a.protocolo)
    protocolo["terciles_mes"] = json.loads(a.ejemplos.with_name("manifiesto_base.json").read_text(encoding="utf-8"))["resumen"]["terciles_mes"]
    with np.load(a.dataset, allow_pickle=False) as d:
        dataset = {c: d[c] for c in d.files}
    tabla = ent.alinear(pd.read_csv(a.ejemplos), dataset["id_ejemplo"])
    salida = a.resultados / "fusion"
    for nombre, ramas in CONFIGURACIONES.items():
        if (salida / nombre / "metricas.json").is_file():
            continue
        r = entrenar_fusion(nombre, ramas, a.modelos, dataset, tabla, protocolo, salida, a.presupuesto, inicio)
        if r is None:
            print("INCOMPLETO: pausa en", nombre)
            return 3
        print(nombre, r["auc_por_semilla"], r.get("apilado", ""))
    rama2 = json.loads((a.resultados / "seleccion_rama2.json").read_text(encoding="utf-8"))["elegido"]
    p = fusion_tardia(a.resultados, dataset["id_ejemplo"], rama2)
    (salida / "e6c_fusion_tardia").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(salida / "e6c_fusion_tardia" / "pred_semilla0.npz", id_ejemplo=np.array([str(i) for i in dataset["id_ejemplo"]], dtype="U"),
                        particion=tabla["particion"].to_numpy(), p_prueba=p.astype(np.float32))
    R = {"configuracion": "e6c_fusion_tardia", "semilla_0": ent.evaluar_predicciones(p, tabla, protocolo, con_intervalos=False)}
    (salida / "e6c_fusion_tardia" / "metricas.json").write_text(json.dumps(R, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("e6c_fusion_tardia", R["semilla_0"]["global"]["auc_planeta_fp"])
    vec = {r: sha256_archivo(a.modelos / r / VERSIONES[r] / "vectores_particion_0.npz") for r in RAMAS}
    escribir_manifiesto(salida / "manifiesto.json", vars(a), extra={
        "versiones": VERSIONES, "rama_2": rama2, "sha256_vectores_particion_0": vec,
        "imagen": "v0 · código de la demo (base común) · no es resultado de ningún curso"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
