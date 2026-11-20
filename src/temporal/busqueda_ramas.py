"""Búsqueda de hiperparámetros de las ramas 1, 2 y 4 (spec 009). Reanudable.

Cada arquitectura recibe el MISMO número de configuraciones (comparación justa de la pregunta 1).
Los espacios están declarados abajo y no se cambian después de ver resultados.

Uso:
  python -I src/temporal/busqueda_ramas.py --dataset data/dataset_temporal.npz \
      --ejemplos compartido/ejemplos.csv --salida resultados/dl/busqueda \
      --arquitecturas m2_atencion [--n 8] [--referencia resultados/dl] [--presupuesto 150]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from functools import partial
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from comun import busqueda  # noqa: E402
from comun import entrenamiento as ent  # noqa: E402
from comun.protocolo import cargar_protocolo  # noqa: E402
from comun.utilidades import sha256_archivo  # noqa: E402
from temporal import modelos  # noqa: E402
from temporal.exp_ramas import EXPERIMENTOS  # noqa: E402

N_CONFIGURACIONES = 8  # la del protocolo y 7 sorteadas, igual para todas las arquitecturas

_ENTRENAMIENTO = {
    "tasa_aprendizaje": ("log", 3e-4, 3e-3),
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    "lote": [16, 32, 64],
}
_FIJO = {"tasa_aprendizaje": 1e-3, "weight_decay": 1e-4, "lote": 32}

# arquitectura → (espacio, configuración fija del protocolo)
ESPACIOS = {
    "m4_escalares": (
        {**_ENTRENAMIENTO, "dropout": [0.1, 0.2, 0.4], "oculta": [16, 32, 64]},
        {**_FIJO, "dropout": 0.2, "oculta": 32},
    ),
    # Rama 1: sin ancho 2 ni lote 16, porque una sola partición no cabría en el tiempo por corrida.
    "m1_astronet": (
        {
            **_ENTRENAMIENTO,
            "lote": [32, 64],
            "dropout": [0.1, 0.3, 0.5],
            "ancho": [0.5, 1.0],
            "pooling": ["max", "promedio"],
        },
        {**_FIJO, "dropout": 0.3, "ancho": 1.0, "pooling": "max"},
    ),
    "m2_promedio": (
        {**_ENTRENAMIENTO, "dropout": [0.1, 0.3, 0.5], "ancho": [0.5, 1.0, 2.0]},
        {**_FIJO, "dropout": 0.3, "ancho": 1.0},
    ),
    "m2_gru": (
        {
            **_ENTRENAMIENTO,
            "dropout": [0.1, 0.3, 0.5],
            "ancho": [0.5, 1.0, 2.0],
            "unidades_gru": [16, 32, 64],
        },
        {**_FIJO, "dropout": 0.3, "ancho": 1.0, "unidades_gru": 32},
    ),
    "m2_atencion": (
        {
            **_ENTRENAMIENTO,
            "dropout": [0.1, 0.3, 0.5],
            "ancho": [0.5, 1.0, 2.0],
            "cabezas": [1, 2, 4],
        },
        {**_FIJO, "dropout": 0.3, "ancho": 1.0, "cabezas": 2},
    ),
}
_CONSTRUCTOR = {
    "m4_escalares": modelos.m4_escalares,
    "m1_astronet": modelos.m1_astronet,
    "m2_promedio": partial(modelos.m2, agregador="promedio"),
    "m2_gru": partial(modelos.m2, agregador="gru"),
    "m2_atencion": partial(modelos.m2, agregador="atencion"),
}


def configuraciones_de(arquitectura: str, n: int = N_CONFIGURACIONES) -> list[dict]:
    espacio, fija = ESPACIOS[arquitectura]
    return busqueda.muestrear(espacio, fija, n, list(ESPACIOS).index(arquitectura))


def experimento_de(arquitectura: str, indice: int, configuracion: dict) -> ent.Experimento:
    """El experimento del protocolo, con otro constructor y otro nombre (cfgNN)."""
    base = EXPERIMENTOS[arquitectura][0]
    constructor = partial(_CONSTRUCTOR[arquitectura], **busqueda.de_arquitectura(configuracion))
    return ent.Experimento(
        busqueda.nombre_de(indice),
        constructor,
        base.claves_entrada,
        base.clave_disponible,
        base.aumentar,
        base.estandarizar,
        base.clave_faltantes,
    )


def _copiar_protocolo_fijo(origen: Path, destino: Path) -> None:
    """Usa como configuración 0 la corrida ya publicada del protocolo fijo (semilla 0)."""
    if (destino / "metricas.json").is_file():
        return
    destino.mkdir(parents=True, exist_ok=True)
    for archivo in ("pred_semilla0.npz", "historial_semilla0.json", "metricas.json"):
        shutil.copyfile(origen / archivo, destino / archivo)


def ejecutar(
    arquitectura: str,
    dataset: dict,
    ejemplos: pd.DataFrame,
    protocolo: dict,
    salida: Path,
    n: int = N_CONFIGURACIONES,
    presupuesto_s: float | None = None,
    referencia: Path | None = None,
) -> dict:
    """Entrena las configuraciones que falten (semilla 0) y resume. Lo ya entrenado no se repite.

    referencia: carpeta de resultados del protocolo fijo (`resultados/dl`). La configuración 0 ES
        ese experimento (mismo código, misma semilla), así que se copia en vez de reentrenarla.
    Lanza `ent.Incompleto` si se agota el presupuesto; volver a llamar continúa donde quedó.
    """
    carpeta = Path(salida) / arquitectura
    carpeta.mkdir(parents=True, exist_ok=True)
    configuraciones = configuraciones_de(arquitectura, n)
    (carpeta / "configuraciones.json").write_text(
        json.dumps(configuraciones, indent=1) + "\n", encoding="utf-8"
    )
    if referencia is not None:
        _copiar_protocolo_fijo(Path(referencia) / arquitectura, carpeta / busqueda.nombre_de(0))
    inicio = time.perf_counter()
    for indice, configuracion in enumerate(configuraciones):
        if (carpeta / busqueda.nombre_de(indice) / "metricas.json").is_file():
            continue
        restante = None if presupuesto_s is None else presupuesto_s - (time.perf_counter() - inicio)
        if restante is not None and restante <= 0:
            raise ent.Incompleto(f"{arquitectura}: pausa antes de {busqueda.nombre_de(indice)}")
        ent.entrenar_cv(
            experimento_de(arquitectura, indice, configuracion),
            dataset,
            ejemplos,
            busqueda.protocolo_con(protocolo, configuracion),
            [0],
            carpeta,
            con_intervalos=False,
            presupuesto_s=restante,
        )
    return busqueda.resumir(carpeta, ejemplos, protocolo)


def guardar_mejores(
    arquitectura: str,
    dataset: dict,
    ejemplos: pd.DataFrame,
    protocolo: dict,
    salida: Path,
    destino: Path,
) -> dict:
    """Guarda el modelo completo que eligió cada partición, con lo necesario para volver a usarlo.

    Se reentrena cada (partición, configuración elegida) y se comprueba que sus predicciones son
    las mismas que las de la búsqueda: así los pesos guardados son los del resultado reportado.
    Reanudable: una partición ya guardada no se repite.
    """
    carpeta = Path(salida) / arquitectura
    destino = Path(destino) / arquitectura
    destino.mkdir(parents=True, exist_ok=True)
    resumen = json.loads((carpeta / "resumen.json").read_text(encoding="utf-8"))
    configuraciones = json.loads((carpeta / "configuraciones.json").read_text(encoding="utf-8"))
    ruta_manifiesto = destino / "manifiesto.json"
    manifiesto = {"arquitectura": arquitectura, "semilla": 0, "particiones": {}}
    if ruta_manifiesto.is_file():
        manifiesto = json.loads(ruta_manifiesto.read_text(encoding="utf-8"))

    for k, nombre in enumerate(resumen["elegida_por_particion"]):
        if str(k) in manifiesto["particiones"]:
            continue
        indice = int(nombre.removeprefix("cfg"))
        configuracion = configuraciones[indice]
        predicciones = ent.guardar_modelo_de_particion(
            experimento_de(arquitectura, indice, configuracion),
            dataset,
            ejemplos,
            busqueda.protocolo_con(protocolo, configuracion),
            k,
            0,
            destino,
        )
        with np.load(carpeta / nombre / "pred_semilla0.npz", allow_pickle=False) as publicado:
            de_prueba = publicado["particion"] == k
            diferencia = np.abs(publicado["p_prueba"][de_prueba] - predicciones["p_prueba"]).max()
        if diferencia > 1e-4:
            raise RuntimeError(
                f"{arquitectura}, partición {k}: el modelo guardado no reproduce la búsqueda "
                f"(diferencia {diferencia:.2g})"
            )
        manifiesto["particiones"][str(k)] = {
            "archivo": f"particion_{k}.keras",
            "configuracion": nombre,
            "hiperparametros": configuracion,
            "diferencia_maxima_con_la_busqueda": float(diferencia),
            "sha256": sha256_archivo(destino / f"particion_{k}.keras"),
        }
        ruta_manifiesto.write_text(
            json.dumps(manifiesto, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
    return manifiesto


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--ejemplos", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--arquitecturas", nargs="+", choices=list(ESPACIOS), required=True)
    ap.add_argument("--n", type=int, default=N_CONFIGURACIONES)
    ap.add_argument("--presupuesto", type=float, default=None, help="segundos; pausa y reanuda")
    ap.add_argument("--referencia", type=Path, default=None, help="resultados del protocolo fijo")
    ap.add_argument(
        "--guardar-mejores",
        type=Path,
        default=None,
        help="carpeta donde guardar el modelo elegido por cada partición (p. ej. modelos/mejores)",
    )
    ap.add_argument("--protocolo", type=Path, default=Path("config/protocolo.json"))
    a = ap.parse_args(argv)

    protocolo = cargar_protocolo(a.protocolo)
    with np.load(a.dataset, allow_pickle=False) as archivo:
        dataset = {clave: archivo[clave] for clave in archivo.files}
    ejemplos = pd.read_csv(a.ejemplos)
    inicio = time.perf_counter()
    for arquitectura in a.arquitecturas:
        restante = None if a.presupuesto is None else a.presupuesto - (time.perf_counter() - inicio)
        try:
            resumen = ejecutar(
                arquitectura, dataset, ejemplos, protocolo, a.salida, a.n, restante, a.referencia
            )
        except ent.Incompleto as pausa:
            print("INCOMPLETO:", pausa)
            return 3
        print(arquitectura, json.dumps(resumen["auc_prueba"]), resumen["elegida_por_particion"])
        if a.guardar_mejores is not None:
            guardar_mejores(arquitectura, dataset, ejemplos, protocolo, a.salida, a.guardar_mejores)
            print(arquitectura, "modelos guardados en", a.guardar_mejores / arquitectura)
    return 0


if __name__ == "__main__":
    sys.exit(main())
