"""Fusión de las cuatro ramas y ablación (spec 005): E6a, E6c, E7 y diagnóstico del apilado.

La cabeza del contrato §6 (Densa 64 → Dropout 0,3 → Densa 3) se entrena sobre vectores CONGELADOS,
con las mismas particiones y validación interna que las ramas. La rama de imagen es la v0 de la
demo (base común): no es un resultado de este curso.

Configuraciones:
    e6a_cuatro_ramas     las cuatro ramas (resultado principal)
    e7_sin_<rama>        ablación: se quita una rama
    e7_solo_<rama>       una sola rama, con la misma cabeza
    e6a_ajuste_cruzado   remedio del apilado: la cabeza solo ve vectores que los codificadores no
                         usaron para entrenar
    e6c_fusion_tardia    promedio de las probabilidades de las ramas (sin cabeza)

Uso:
  python -I src/fusion/fusionar.py --modelos modelos --dataset data/dataset_temporal.npz \
      --ejemplos compartido/ejemplos.csv --resultados resultados/dl [--presupuesto 150]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from keras import layers  # noqa: E402

from comun import entrenamiento as ent  # noqa: E402
from comun import metricas  # noqa: E402
from comun.protocolo import Separacion, cargar_protocolo, comprobar_sin_solape, separar  # noqa: E402
from comun.utilidades import escribir_manifiesto, sha256_archivo  # noqa: E402
from fusion.vectores import RAMAS, armar_entrada, cargar_vectores  # noqa: E402

VERSIONES = {"vistas": "v1", "transitos": "v1", "imagen": "v0", "escalares": "v1"}
CONFIGURACION_PRINCIPAL = "e6a_cuatro_ramas"
CONFIGURACIONES = {
    CONFIGURACION_PRINCIPAL: RAMAS,
    **{f"e7_sin_{rama}": tuple(r for r in RAMAS if r != rama) for rama in RAMAS},
    **{f"e7_solo_{rama}": (rama,) for rama in RAMAS},
    # Remedio del apilado (spec 005, RF-8): la cabeza se entrena SOLO con vectores que los
    # codificadores no vieron (la validación interna), separando un 25 % de estrellas para parar.
    "e6a_ajuste_cruzado": RAMAS,
}
EXPERIMENTO_DE_RAMA = {"vistas": "m1_astronet", "imagen": "imagen_v0", "escalares": "m4_escalares"}
N_CLASES = 3


@dataclass(frozen=True)
class _Particion:
    """Separación de la partición k y la entrada de la cabeza ya estandarizada."""

    sep: Separacion
    x: np.ndarray


def cabeza(dim_entrada: int, semilla: int, dropout: float = 0.3) -> keras.Model:
    """Cabeza de fusión del contrato §6: Densa(64, ReLU) → Dropout → Densa(3, softmax)."""
    keras.utils.set_random_seed(semilla)
    x = keras.Input((dim_entrada,), name="z")
    # El orden de creación de las capas fija qué números aleatorios recibe cada una: el Dropout
    # se crea antes que la capa densa para reproducir exactamente los resultados publicados.
    dropout_fusion = layers.Dropout(dropout, name="dropout_fusion")
    oculta = dropout_fusion(layers.Dense(64, activation="relu")(x))
    return keras.Model(
        x, layers.Dense(N_CLASES, activation="softmax")(oculta), name="cabeza_fusion"
    )


def entrada_de_particion(modelos: Path, k: int, ids, dataset: dict, ramas, i_ent) -> np.ndarray:
    """Entrada de la cabeza para la partición k: vectores estandarizados + banderas de presencia.

    Los vectores se estandarizan con el entrenamiento; las banderas (1 = la rama tiene dato) no.
    """
    vectores, banderas = cargar_vectores(modelos, VERSIONES, k, ids, dataset)
    x = armar_entrada(vectores, banderas, ramas)
    n_vectores = x.shape[1] - len(ramas)
    media, desv = ent.estadisticas_estandarizacion(x[i_ent, :n_vectores])
    x[:, :n_vectores] = ent.aplicar_estandarizacion(x[:, :n_vectores], media, desv)

    # un vector ausente debe seguir en cero después de estandarizar
    inicio = 0
    for rama in ramas:
        ancho = vectores[rama].shape[1]
        sin_dato = banderas[:, RAMAS.index(rama)] == 0
        x[sin_dato, inicio : inicio + ancho] = 0.0
        inicio += ancho
    return x


def _indices_de_entrenamiento(
    nombre: str, sep: Separacion, kepid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Máscaras (entrenar, parar) de la cabeza.

    Normal: entrena con el entrenamiento de las ramas y para con la validación interna.
    Ajuste cruzado: entrena con el 75 % de las estrellas de la validación interna y para con el
    25 % restante, de modo que la cabeza nunca ve vectores que los codificadores memorizaron.
    """
    if not nombre.endswith("ajuste_cruzado"):
        return sep.entrenamiento, sep.validacion
    estrellas = np.unique(kepid[sep.validacion])
    para_parar = estrellas[np.arange(len(estrellas)) % 4 == 0]
    i_parar = sep.validacion & np.isin(kepid, para_parar)
    return sep.validacion & ~i_parar, i_parar


def _entrenar_semilla(
    nombre: str,
    semilla: int,
    particiones: dict[int, _Particion],
    y: np.ndarray,
    kepid: np.ndarray,
    cfg: dict,
    carpeta: Path,
) -> dict[str, np.ndarray]:
    """Entrena una cabeza por partición con una semilla; devuelve sus predicciones."""
    n = len(y)
    p_prueba, p_val = np.zeros((n, N_CLASES)), np.zeros((n, N_CLASES))
    modelo_de_validacion = np.full(n, -1)
    diagnostico = []

    for k, particion in particiones.items():
        sep, x = particion.sep, particion.x
        i_ent, i_parar = _indices_de_entrenamiento(nombre, sep, kepid)
        modelo = cabeza(x.shape[1], 1000 * semilla + k, cfg["dropout_fusion"])
        modelo, historial = ent.entrenar_una(
            modelo,
            x[i_ent],
            y[i_ent],
            x[i_parar],
            y[i_parar],
            cfg,
            None,
            np.random.default_rng([semilla, k]),
        )
        p_prueba[sep.prueba] = modelo.predict(x[sep.prueba], verbose=0)
        p_val[sep.validacion] = modelo.predict(x[sep.validacion], verbose=0)
        modelo_de_validacion[sep.validacion] = k

        if semilla == 0:  # la semilla 0 deja las cabezas y el diagnóstico del apilado
            (carpeta / "cabezas").mkdir(exist_ok=True)
            modelo.save(carpeta / "cabezas" / f"particion_{k}.keras")
            nll_ent = metricas.nll(modelo.predict(x[i_ent], verbose=0), y[i_ent])
            nll_parar = metricas.nll(modelo.predict(x[i_parar], verbose=0), y[i_parar])
            diagnostico.append(
                {
                    "particion": k,
                    "epocas": historial["epoca_elegida"],
                    "nll_entrenamiento": round(nll_ent, 4),
                    "nll_validacion": round(nll_parar, 4),
                }
            )

    if semilla == 0:
        texto = json.dumps(diagnostico, indent=1) + "\n"
        (carpeta / "diagnostico_apilado.json").write_text(texto, encoding="utf-8")
    return {
        "p_prueba": p_prueba.astype(np.float32),
        "p_validacion": p_val.astype(np.float32),
        "particion_modelo_validacion": modelo_de_validacion,
    }


def _resumen_del_apilado(carpeta: Path) -> dict | None:
    """¿La cabeza ve vectores "demasiado buenos" en entrenamiento? Compara NLL de entrenar y parar.

    Los codificadores ya vieron sus ejemplos de entrenamiento, así que esos vectores separan las
    clases mejor que los de ejemplos nuevos. Si la NLL de validación duplica la de entrenamiento,
    se marca la alerta de sobreconfianza por apilado.
    """
    ruta = carpeta / "diagnostico_apilado.json"
    if not ruta.is_file():
        return None
    filas = json.loads(ruta.read_text(encoding="utf-8"))
    nll_ent = np.mean([fila["nll_entrenamiento"] for fila in filas])
    nll_val = np.mean([fila["nll_validacion"] for fila in filas])
    return {
        "nll_entrenamiento": round(float(nll_ent), 4),
        "nll_validacion": round(float(nll_val), 4),
        "razon": round(float(nll_val / nll_ent), 2),
        "alerta": bool(nll_val > 2 * nll_ent),
    }


def entrenar_fusion(
    nombre: str,
    ramas,
    modelos: Path,
    dataset: dict,
    tabla: pd.DataFrame,
    protocolo: dict,
    salida: Path,
    limite: float | None = None,
    inicio: float = 0.0,
) -> dict | None:
    """Entrena la cabeza para una configuración de ramas, con 5 semillas. Reanudable por semilla.

    Devuelve `None` si se agotó el presupuesto de tiempo (lo terminado queda guardado).
    """
    carpeta = salida / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    ids = dataset["id_ejemplo"]
    y = tabla["clase"].to_numpy().astype(int)
    particion = tabla["particion"].to_numpy()
    kepid = tabla["kepid"].to_numpy()
    n_particiones = protocolo["particiones"]["n"]
    semillas = protocolo["semillas_cabeza_fusion"]

    particiones = {}
    for k in range(n_particiones):
        sep = separar(particion, k, n_particiones)
        comprobar_sin_solape(kepid, sep)
        i_ent = np.flatnonzero(sep.entrenamiento)
        x = entrada_de_particion(modelos, k, ids, dataset, ramas, i_ent)
        particiones[k] = _Particion(sep, x)

    for semilla in semillas:
        ruta = carpeta / f"pred_semilla{semilla}.npz"
        if ruta.is_file():
            continue
        if limite is not None and time.perf_counter() - inicio > limite:
            return None
        predicciones = _entrenar_semilla(
            nombre, semilla, particiones, y, kepid, protocolo["entrenamiento"], carpeta
        )
        np.savez_compressed(
            ruta,
            id_ejemplo=np.array([str(i) for i in ids], dtype="U"),
            particion=particion,
            **predicciones,
        )

    p_por_semilla = [np.load(carpeta / f"pred_semilla{s}.npz")["p_prueba"] for s in semillas]
    aucs = [metricas.auc_planeta_fp(p, y) for p in p_por_semilla]
    resumen = {
        "configuracion": nombre,
        "ramas": list(ramas),
        "dim_entrada": int(particiones[0].x.shape[1]),
        "auc_por_semilla": [round(a, 4) for a in aucs],
        "auc_media_semillas": round(float(np.mean(aucs)), 4),
        "auc_desviacion_semillas": round(float(np.std(aucs)), 4),
        "semilla_0": ent.evaluar_predicciones(
            p_por_semilla[0],
            tabla,
            protocolo,
            con_intervalos=(nombre == CONFIGURACION_PRINCIPAL),
        ),
    }
    apilado = _resumen_del_apilado(carpeta)
    if apilado is not None:
        resumen["apilado"] = apilado
    _escribir_json(carpeta / "metricas.json", resumen)
    return resumen


def fusion_tardia(resultados: Path, ids: np.ndarray, rama2: str) -> np.ndarray:
    """E6c: promedio de las probabilidades de las cuatro ramas entrenadas por separado (semilla 0).

    Para cada ejemplo se promedia solo sobre las ramas que tienen predicción para él.
    """
    ids = np.array([str(i) for i in ids])
    suma, cuenta = np.zeros((len(ids), N_CLASES)), np.zeros(len(ids))
    experimentos = (
        EXPERIMENTO_DE_RAMA["vistas"],
        rama2,
        EXPERIMENTO_DE_RAMA["imagen"],
        EXPERIMENTO_DE_RAMA["escalares"],
    )
    for experimento in experimentos:
        ruta = resultados / experimento / "pred_semilla0.npz"
        with np.load(ruta, allow_pickle=False) as d:
            posicion = {str(ident): j for j, ident in enumerate(d["id_ejemplo"])}
            p = d["p_prueba"]
        for i, ident in enumerate(ids):
            if ident in posicion:
                suma[i] += p[posicion[ident]]
                cuenta[i] += 1
    return suma / cuenta[:, None]


def _escribir_json(ruta: Path, contenido) -> None:
    ruta.write_text(json.dumps(contenido, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _guardar_fusion_tardia(salida: Path, p, dataset: dict, tabla, protocolo: dict) -> dict:
    carpeta = salida / "e6c_fusion_tardia"
    carpeta.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        carpeta / "pred_semilla0.npz",
        id_ejemplo=np.array([str(i) for i in dataset["id_ejemplo"]], dtype="U"),
        particion=tabla["particion"].to_numpy(),
        p_prueba=p.astype(np.float32),
    )
    resumen = {
        "configuracion": "e6c_fusion_tardia",
        "semilla_0": ent.evaluar_predicciones(p, tabla, protocolo, con_intervalos=False),
    }
    _escribir_json(carpeta / "metricas.json", resumen)
    return resumen


def _argumentos(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modelos", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--ejemplos", type=Path, required=True)
    ap.add_argument("--resultados", type=Path, required=True)
    ap.add_argument("--presupuesto", type=float, default=None, help="segundos; pausa y reanuda")
    ap.add_argument("--protocolo", type=Path, default=Path("config/protocolo.json"))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = _argumentos(argv)
    inicio = time.perf_counter()
    protocolo = cargar_protocolo(a.protocolo)
    manifiesto_base = json.loads(
        a.ejemplos.with_name("manifiesto_base.json").read_text(encoding="utf-8")
    )
    protocolo["terciles_mes"] = manifiesto_base["resumen"]["terciles_mes"]
    with np.load(a.dataset, allow_pickle=False) as d:
        dataset = {clave: d[clave] for clave in d.files}
    tabla = ent.alinear(pd.read_csv(a.ejemplos), dataset["id_ejemplo"])
    salida = a.resultados / "fusion"

    # 1. Cabeza de fusión para cada configuración de ramas
    for nombre, ramas in CONFIGURACIONES.items():
        if (salida / nombre / "metricas.json").is_file():
            continue
        resumen = entrenar_fusion(
            nombre, ramas, a.modelos, dataset, tabla, protocolo, salida, a.presupuesto, inicio
        )
        if resumen is None:
            print("INCOMPLETO: pausa en", nombre)
            return 3
        print(nombre, resumen["auc_por_semilla"], resumen.get("apilado", ""))

    # 2. Fusión tardía: promedio de probabilidades, sin cabeza
    seleccion = json.loads((a.resultados / "seleccion_rama2.json").read_text(encoding="utf-8"))
    rama2 = seleccion["elegido"]
    p_tardia = fusion_tardia(a.resultados, dataset["id_ejemplo"], rama2)
    resumen = _guardar_fusion_tardia(salida, p_tardia, dataset, tabla, protocolo)
    print("e6c_fusion_tardia", resumen["semilla_0"]["global"]["auc_planeta_fp"])

    # 3. Manifiesto: con qué vectores se fusionó
    huellas = {
        rama: sha256_archivo(a.modelos / rama / VERSIONES[rama] / "vectores_particion_0.npz")
        for rama in RAMAS
    }
    escribir_manifiesto(
        salida / "manifiesto.json",
        vars(a),
        extra={
            "versiones": VERSIONES,
            "rama_2": rama2,
            "sha256_vectores_particion_0": huellas,
            "imagen": "v0 · código de la demo (base común) · no es resultado de ningún curso",
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
