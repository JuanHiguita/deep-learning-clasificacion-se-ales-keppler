"""Prueba de extremo a extremo con datos reales (spec 011).

Para una estrella por partición, elegida al azar con semilla fija, recorre TODO el camino:

    archivos FITS de Kepler ─► extractor ─► entradas ─► 4 codificadores ─► cabeza ─► temperatura

y comprueba, en cada paso, que se obtiene lo mismo que quedó publicado:

    1. las entradas reconstruidas desde los FITS = las de data/dataset_temporal.npz
    2. el vector de cada rama                  = el guardado en modelos/<rama>/<versión>/
    3. la probabilidad calibrada               = la de resultados/dl/calibracion/pred_calibrada.npz

La imagen de píxeles (rama 3) se toma de data/dataset_demo.npz, que produce el script de la demo
sin modificar: ese paso no se vuelve a derivar aquí.

Uso:  python scripts/prueba_e2e.py [--estrellas-por-particion 1]
Termina con código 0 si todo coincide y 1 si algo difiere.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from comun.protocolo import cargar_protocolo  # noqa: E402
from fusion import fusionar  # noqa: E402
from fusion.sistema import Sistema  # noqa: E402
from fusion.vectores import RAMAS  # noqa: E402
from temporal import construir_dataset_temporal  # noqa: E402

SEMILLA = 7  # la de `muestras_al_azar` del protocolo
TOLERANCIA = 1e-4
CLASES = ("planeta", "falso positivo", "sin señal")


def _cargar(ruta: Path) -> dict:
    with np.load(ruta, allow_pickle=False) as archivo:
        return {clave: archivo[clave] for clave in archivo.files}


def _filas(ids_de_referencia: np.ndarray, ids: np.ndarray) -> np.ndarray:
    posicion = {str(ident): i for i, ident in enumerate(ids_de_referencia)}
    return np.array([posicion[str(ident)] for ident in ids])


def _imagenes(ids: np.ndarray, demo: dict) -> dict:
    """Imagen de cada ejemplo (ceros y bandera 0 si la estrella no tiene imagen)."""
    posicion = {str(ident): i for i, ident in enumerate(demo["id_ejemplo"])}
    img = np.zeros((len(ids), *demo["img"].shape[1:]), np.float32)
    disponible = np.zeros(len(ids), np.float32)
    for i, ident in enumerate(ids):
        if str(ident) in posicion:
            img[i], disponible[i] = demo["img"][posicion[str(ident)]], 1.0
    return {"img": img, "disponible_imagen": disponible}


def probar(estrellas_por_particion: int = 1) -> dict:
    protocolo = cargar_protocolo(RAIZ / "config" / "protocolo.json")
    ejemplos = pd.read_csv(RAIZ / "compartido" / "ejemplos.csv")
    koi = pd.read_csv(RAIZ / "data" / "koi_dr25.csv")
    publicado = _cargar(RAIZ / "data" / "dataset_temporal.npz")
    demo = _cargar(RAIZ / "data" / "dataset_demo.npz")
    calibrada = _cargar(RAIZ / "resultados" / "dl" / "calibracion" / "pred_calibrada.npz")
    sistema = Sistema(RAIZ / "modelos")

    # Estrellas al azar con semilla fija, entre las que tienen todas las ramas
    rng = np.random.default_rng(SEMILLA)
    con_imagen = {str(i).split("|")[0] for i in demo["id_ejemplo"]}
    disponibles = ejemplos[
        ejemplos.kepoi_name.isin(con_imagen) & ~ejemplos.prerregistrado.astype(bool)
    ]
    informe = {"semilla": SEMILLA, "tolerancia": TOLERANCIA, "estrellas": [], "correcto": True}

    for k in range(protocolo["particiones"]["n"]):
        candidatas = np.sort(disponibles[disponibles.particion == k].kepid.unique())
        for kepid in rng.choice(candidatas, estrellas_por_particion, replace=False):
            de_la_estrella = ejemplos[ejemplos.kepid == kepid]

            # 1. De los FITS a las entradas
            entradas, _ = construir_dataset_temporal.construir(
                de_la_estrella, koi, RAIZ / "data" / "llc", protocolo
            )
            ids = np.array([str(i) for i in entradas["id_ejemplo"]])
            filas = _filas(publicado["id_ejemplo"], ids)
            dif_entradas = max(
                float(np.abs(entradas[c].astype(float) - publicado[c][filas].astype(float)).max())
                for c in ("vista_global", "vista_local", "transitos", "escalares", "consistencia")
            )

            # 2 y 3. Del sistema: vectores de cada rama y probabilidad calibrada
            entradas.update(_imagenes(ids, demo))
            respuesta = sistema.responder(entradas, particion=k, tasa_base=0.05)
            dif_vectores = {}
            for rama in RAMAS:
                carpeta = RAIZ / "modelos" / rama / fusionar.VERSIONES[rama]
                guardado = _cargar(carpeta / f"vectores_particion_{k}.npz")
                con_dato = respuesta.ramas_con_dato[:, RAMAS.index(rama)] > 0
                if not con_dato.any():
                    continue
                z = guardado["z"][_filas(guardado["id_ejemplo"], ids[con_dato])]
                dif_vectores[rama] = float(np.abs(respuesta.vectores[rama][con_dato] - z).max())
            p_publicada = calibrada["p_despues"][_filas(calibrada["id_ejemplo"], ids)]
            dif_p = float(np.abs(respuesta.p - p_publicada).max())

            # Estrella "nueva": promedio de los cinco sistemas (no se compara: solo debe funcionar)
            como_nueva = sistema.responder(entradas, particion=None)

            diferencias = [dif_entradas, dif_p, *dif_vectores.values()]
            correcto = max(diferencias) < TOLERANCIA
            informe["correcto"] &= correcto
            tabla = de_la_estrella.set_index("id_ejemplo").loc[ids]
            informe["estrellas"].append(
                {
                    "kepid": int(kepid),
                    "particion": k,
                    "correcto": bool(correcto),
                    "diferencia_entradas": dif_entradas,
                    "diferencia_vectores": dif_vectores,
                    "diferencia_probabilidad": dif_p,
                    "ejemplos": [
                        {
                            "id_ejemplo": ident,
                            "clase_real": CLASES[int(tabla.loc[ident, "clase"])],
                            "p_planeta": round(float(respuesta.p[i, 0]), 4),
                            "p_falso_positivo": round(float(respuesta.p[i, 1]), 4),
                            "p_sin_senal": round(float(respuesta.p[i, 2]), 4),
                            "q_planeta_con_tasa_base_0.05": round(
                                float(respuesta.q_con_tasa_base[i]), 4
                            ),
                            "entropia": round(float(respuesta.entropia[i]), 4),
                            "p_planeta_como_estrella_nueva": round(float(como_nueva.p[i, 0]), 4),
                        }
                        for i, ident in enumerate(ids)
                    ],
                }
            )
            estado = "OK " if correcto else "FALLA"
            print(
                f"{estado} kepid {kepid} (partición {k}): entradas {dif_entradas:.1e} · "
                f"vectores {max(dif_vectores.values()):.1e} · probabilidad {dif_p:.1e}"
            )
    return informe


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--estrellas-por-particion", type=int, default=1)
    ap.add_argument(
        "--salida", type=Path, default=RAIZ / "resultados" / "dl" / "e2e" / "informe.json"
    )
    a = ap.parse_args(argv)
    informe = probar(a.estrellas_por_particion)
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    a.salida.write_text(json.dumps(informe, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("PRUEBA E2E:", "correcta" if informe["correcto"] else "CON DIFERENCIAS", "·", a.salida)
    return 0 if informe["correcto"] else 1


if __name__ == "__main__":
    sys.exit(main())
