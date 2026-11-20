"""Búsqueda de hiperparámetros de la cabeza de fusión (spec 009). Reanudable por configuración.

Los codificadores no se tocan: se usan los vectores congelados ya exportados. Solo cambia la
cabeza (neuronas ocultas, dropout) y cómo se entrena (tasa de aprendizaje, decaimiento, lote).

Uso:
  python -I src/fusion/busqueda_cabeza.py --modelos modelos --dataset data/dataset_temporal.npz \
      --ejemplos compartido/ejemplos.csv --resultados resultados/dl [--n 8]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from comun import busqueda  # noqa: E402
from comun import entrenamiento as ent  # noqa: E402
from comun.protocolo import cargar_protocolo  # noqa: E402
from comun.utilidades import sha256_archivo  # noqa: E402
from fusion import fusionar  # noqa: E402
from fusion.vectores import RAMAS  # noqa: E402

N_CONFIGURACIONES = 8
INDICE_EN_EL_SORTEO = 100  # distinto de los índices de las ramas: otro sorteo
ESPACIO = {
    "tasa_aprendizaje": ("log", 3e-4, 3e-3),
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    "lote": [16, 32, 64],
    "dropout": [0.1, 0.3, 0.5],
    "oculta": [32, 64, 128],
}
FIJA = {"tasa_aprendizaje": 1e-3, "weight_decay": 1e-4, "lote": 32, "dropout": 0.3, "oculta": 64}
CONFIGURACION = fusionar.CONFIGURACION_PRINCIPAL


def configuraciones(n: int = N_CONFIGURACIONES) -> list[dict]:
    return busqueda.muestrear(ESPACIO, FIJA, n, INDICE_EN_EL_SORTEO)


def _copiar_protocolo_fijo(origen: Path, destino: Path) -> None:
    """La configuración 0 es la fusión publicada (semilla 0): se copia en vez de reentrenarla."""
    destino.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(origen / "pred_semilla0.npz", destino / "pred_semilla0.npz")
    diagnostico = json.loads((origen / "diagnostico_apilado.json").read_text(encoding="utf-8"))
    epocas = [fila["epocas"] for fila in diagnostico]
    _escribir_metricas(destino, epocas, parametros=None)


def _escribir_metricas(carpeta: Path, epocas: list[int], parametros: int | None) -> None:
    """Mismo formato mínimo que `entrenar_cv`, que es el que lee `busqueda.resumir`."""
    contenido = {"parametros": parametros, "por_semilla": {"0": {"epocas": epocas}}}
    (carpeta / "metricas.json").write_text(json.dumps(contenido, indent=1) + "\n", encoding="utf-8")


def ejecutar(
    modelos: Path,
    dataset: dict,
    ejemplos: pd.DataFrame,
    protocolo: dict,
    resultados: Path,
    n: int = N_CONFIGURACIONES,
) -> dict:
    """Entrena la cabeza con cada configuración que falte (semilla 0) y resume."""
    carpeta = Path(resultados) / "busqueda" / "cabeza_fusion"
    carpeta.mkdir(parents=True, exist_ok=True)
    lista = configuraciones(n)
    (carpeta / "configuraciones.json").write_text(
        json.dumps(lista, indent=1) + "\n", encoding="utf-8"
    )

    tabla = ent.alinear(ejemplos, dataset["id_ejemplo"])
    y, kepid = tabla["clase"].to_numpy().astype(int), tabla["kepid"].to_numpy()
    particiones = None  # se preparan solo si hay algo que entrenar

    for indice, configuracion in enumerate(lista):
        destino = carpeta / busqueda.nombre_de(indice)
        if (destino / "metricas.json").is_file():
            continue
        if indice == 0:
            _copiar_protocolo_fijo(Path(resultados) / "fusion" / CONFIGURACION, destino)
            continue
        if particiones is None:
            particiones = fusionar.preparar_particiones(
                RAMAS, modelos, dataset, tabla, protocolo["particiones"]["n"]
            )
        destino.mkdir(parents=True, exist_ok=True)
        cfg = busqueda.protocolo_con(
            protocolo,
            configuracion,
            extra={
                "dropout_fusion": configuracion["dropout"],
                "oculta_fusion": configuracion["oculta"],
            },
        )["entrenamiento"]
        predicciones = fusionar.entrenar_semilla(
            CONFIGURACION, 0, particiones, y, kepid, cfg, destino
        )
        np.savez_compressed(
            destino / "pred_semilla0.npz",
            id_ejemplo=np.array([str(i) for i in dataset["id_ejemplo"]], dtype="U"),
            particion=tabla["particion"].to_numpy(),
            **predicciones,
        )
        diagnostico = json.loads((destino / "diagnostico_apilado.json").read_text(encoding="utf-8"))
        dim_entrada = particiones[0].x.shape[1]
        oculta = configuracion["oculta"]
        parametros = (dim_entrada + 1) * oculta + (oculta + 1) * fusionar.N_CLASES
        _escribir_metricas(destino, [fila["epocas"] for fila in diagnostico], parametros)
    return busqueda.resumir(carpeta, ejemplos, protocolo)


def guardar_mejores(resultados: Path, destino: Path) -> dict:
    """Copia a `destino/cabeza_fusion` la cabeza que eligió cada partición, con su manifiesto.

    Las cabezas ya se guardaron al entrenar (semilla 0), así que no hace falta reentrenar.
    """
    carpeta = Path(resultados) / "busqueda" / "cabeza_fusion"
    destino = Path(destino) / "cabeza_fusion"
    destino.mkdir(parents=True, exist_ok=True)
    resumen = json.loads((carpeta / "resumen.json").read_text(encoding="utf-8"))
    lista = json.loads((carpeta / "configuraciones.json").read_text(encoding="utf-8"))
    manifiesto = {"arquitectura": "cabeza_fusion", "semilla": 0, "particiones": {}}
    for k, nombre in enumerate(resumen["elegida_por_particion"]):
        if nombre == busqueda.nombre_de(0):  # la del protocolo vive con la fusión publicada
            origen = Path(resultados) / "fusion" / CONFIGURACION / "cabezas"
        else:
            origen = carpeta / nombre / "cabezas"
        archivo = f"particion_{k}.keras"
        shutil.copyfile(origen / archivo, destino / archivo)
        manifiesto["particiones"][str(k)] = {
            "archivo": archivo,
            "configuracion": nombre,
            "hiperparametros": lista[int(nombre.removeprefix("cfg"))],
            "sha256": sha256_archivo(destino / archivo),
        }
    (destino / "manifiesto.json").write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return manifiesto


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modelos", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--ejemplos", type=Path, required=True)
    ap.add_argument("--resultados", type=Path, required=True)
    ap.add_argument("--n", type=int, default=N_CONFIGURACIONES)
    ap.add_argument("--guardar-mejores", type=Path, default=None, help="p. ej. modelos/mejores")
    ap.add_argument("--protocolo", type=Path, default=Path("config/protocolo.json"))
    a = ap.parse_args(argv)

    protocolo = cargar_protocolo(a.protocolo)
    with np.load(a.dataset, allow_pickle=False) as archivo:
        dataset = {clave: archivo[clave] for clave in archivo.files}
    resumen = ejecutar(a.modelos, dataset, pd.read_csv(a.ejemplos), protocolo, a.resultados, a.n)
    print("cabeza_fusion", json.dumps(resumen["auc_prueba"]), resumen["elegida_por_particion"])
    if a.guardar_mejores is not None:
        guardar_mejores(a.resultados, a.guardar_mejores)
        print("cabezas guardadas en", a.guardar_mejores / "cabeza_fusion")
    return 0


if __name__ == "__main__":
    sys.exit(main())
