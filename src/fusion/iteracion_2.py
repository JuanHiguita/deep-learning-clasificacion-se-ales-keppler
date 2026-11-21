"""Iteración 2 (spec 014): de la búsqueda a la respuesta conjunta, sin la rama de imagen.

Pasos (cada uno se salta si ya está hecho):
  1. elegir    por rama y por partición k, la red y la configuración con menor NLL en la
               validación interna de k (la prueba no decide). La rama 2 elige entre sus tres
               agregadores; las ramas 1 y 4 tienen una sola red.
  2. exportar  el codificador elegido (hasta la capa 'embedding') y sus vectores, con el formato
               del contrato §5, en `iteracion_2/modelos_finales/<rama>/v1/`.
  3. fusionar  la cabeza del contrato §6 sobre los vectores congelados, 5 semillas. La rama de
               imagen queda declarada como ausente (bandera en 0): cuando el curso de Visión la
               entregue, basta con poner su versión; nada más cambia.
  4. comparar  diferencias pareadas (bootstrap por estrella) contra los modelos tabulares
               entrenados con las MISMAS señales. Criterio declarado antes de entrenar: la fusión
               debe superar de forma distinguible al tabular SIN centroide (S3-B).

Uso (desde la raíz del repositorio):
  python -I src/fusion/iteracion_2.py --koi data/koi_dr25.csv [--ramas vistas transitos escalares]
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
from comun import estadistica, lineas_base_tabulares, metricas  # noqa: E402
from comun.protocolo import cargar_protocolo, separar  # noqa: E402
from comun.utilidades import sha256_archivo  # noqa: E402
from fusion import fusionar  # noqa: E402
from temporal import modelos as _capas_propias  # noqa: E402, F401  (registra las capas al cargar)
from temporal.exp_ramas import EXPERIMENTOS  # noqa: E402

RAIZ = Path("iteracion_2")
REDES_DE_RAMA = {
    "vistas": ("m1_astronet",),
    "transitos": ("m2_promedio", "m2_gru", "m2_atencion"),
    "escalares": ("m4_escalares",),
}
VERSION = "v1"
# Qué ramas entran en cada configuración de fusión. Las que terminan en `ajuste_cruzado` entrenan
# la cabeza solo con vectores que los codificadores no vieron (remedio del apilado, spec 005).
CONFIGURACIONES = {
    "tres_ramas": ("vistas", "transitos", "escalares"),
    "tres_ramas_ajuste_cruzado": ("vistas", "transitos", "escalares"),
    "solo_curvas": ("vistas", "transitos"),  # nada del catálogo: solo lo que la red lee en la luz
    "solo_vistas": ("vistas",),
    "solo_transitos": ("transitos",),
    "solo_escalares": ("escalares",),
}
PRINCIPAL = "tres_ramas"


def _escribir_json(ruta: Path, contenido) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(contenido, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# 1. Elegir
# ---------------------------------------------------------------------------------------------


def elegir(busqueda: Path, rama: str, n_particiones: int) -> list[dict]:
    """Para cada partición, la (red, configuración) con menor NLL en SU validación interna."""
    candidatas = []
    for red in REDES_DE_RAMA[rama]:
        ruta = Path(busqueda) / red / "configuraciones.csv"
        if not (Path(busqueda) / red / "resumen.json").is_file():
            raise FileNotFoundError(f"La búsqueda de {red} no ha terminado ({ruta.parent})")
        tabla = pd.read_csv(ruta)
        tabla.insert(0, "red", red)
        candidatas.append(tabla)
    candidatas = pd.concat(candidatas, ignore_index=True)
    elegidas = []
    for k in range(n_particiones):
        fila = candidatas.loc[candidatas[f"nll_val_k{k}"].idxmin()]  # empate: la primera
        elegidas.append(
            {
                "particion": k,
                "red": fila["red"],
                "configuracion": fila["configuracion"],
                "nll_validacion": float(fila[f"nll_val_k{k}"]),
                "parametros": int(fila["parametros"]),
            }
        )
    return elegidas


# ---------------------------------------------------------------------------------------------
# 2. Exportar
# ---------------------------------------------------------------------------------------------


def exportar(
    rama: str,
    elegidas: list[dict],
    busqueda: Path,
    dataset: dict,
    ejemplos: pd.DataFrame,
    protocolo: dict,
    destino: Path,
    referencia: Path | None = None,
) -> dict:
    """Codificador y vectores de cada partición, con el modelo que guardó la búsqueda."""
    destino = Path(destino) / rama / VERSION
    ruta_manifiesto = destino / "manifiesto.json"
    if ruta_manifiesto.is_file():
        return json.loads(ruta_manifiesto.read_text(encoding="utf-8"))
    n_particiones = protocolo["particiones"]["n"]
    semilla = protocolo["semilla_codificador_congelado"]
    manifiesto = {"rama": rama, "version": VERSION, "semilla": semilla, "particiones": {}}
    for eleccion in elegidas:
        k, red = eleccion["particion"], eleccion["red"]
        exp = EXPERIMENTOS[red][0]
        datos = ent._preparar_datos(exp, dataset, ejemplos)
        sep = separar(datos.particion, k, n_particiones)
        i_ent = np.flatnonzero(sep.entrenamiento & datos.disponible)
        x, estadisticas = ent._estandarizar_con_entrenamiento(exp, datos.entradas, i_ent)
        origen = (
            Path(busqueda)
            / red
            / eleccion["configuracion"]
            / "modelos"
            / f"particion_{k}_semilla_{semilla}.keras"
        )
        if not origen.is_file() and referencia is not None:
            # La configuración 0 es la corrida del protocolo fijo: su modelo está en esa carpeta.
            origen = Path(referencia) / red / "modelos" / f"particion_{k}_semilla_{semilla}.keras"
        modelo = keras.saving.load_model(origen, compile=False, safe_mode=True)
        ent._exportar_particion(modelo, x, exp, datos.ids, k, estadisticas, destino)
        manifiesto["particiones"][str(k)] = {
            **eleccion,
            "modelo_de_origen": origen.as_posix(),
            "sha256_modelo_de_origen": sha256_archivo(origen),
            "sha256_codificador": sha256_archivo(destino / f"particion_{k}.keras"),
        }
        print(f"  {rama} · partición {k}: {red} {eleccion['configuracion']}", flush=True)
    _escribir_json(ruta_manifiesto, manifiesto)
    return manifiesto


# ---------------------------------------------------------------------------------------------
# 4. Comparar con los modelos tabulares (mismas señales, mismas particiones)
# ---------------------------------------------------------------------------------------------


def _q_por_koi(carpeta: Path, semillas: list[int], kois: pd.Series) -> np.ndarray:
    """P(planeta | planeta o falso positivo) media de las semillas, alineada con `kois`."""
    q = []
    for semilla in semillas:
        with np.load(carpeta / f"pred_semilla{semilla}.npz", allow_pickle=False) as d:
            ids = [str(i) for i in d["id_ejemplo"]]
            por_id = dict(zip(ids, metricas.q_planeta(d["p_prueba"]), strict=True))
        q.append([por_id.get(f"{koi}|transito", np.nan) for koi in kois])
    return np.mean(q, axis=0)


def comparar(
    salida: Path, configuraciones: list[str], ejemplos: pd.DataFrame, koi: Path, protocolo: dict
) -> dict:
    """AUC planeta vs. falso positivo por KOI y diferencias pareadas fusión − tabular."""
    dr25 = pd.read_csv(koi)
    base = ejemplos[(ejemplos["tipo"] == "transito") & ~ejemplos["prerregistrado"].astype(bool)]
    sel = (
        base[["kepoi_name", "kepid", "clase", "particion"]]
        .merge(dr25.drop(columns=["kepid"]), on="kepoi_name", how="left", validate="one_to_one")
        .reset_index(drop=True)
    )
    puntajes = {
        "tabular_sin_centroide": lineas_base_tabulares.predecir_oof(
            sel, lineas_base_tabulares.S3B, protocolo
        ),
        "tabular_con_centroide": lineas_base_tabulares.predecir_oof(
            sel, lineas_base_tabulares.S3C, protocolo
        ),
    }
    for nombre in configuraciones:
        puntajes[nombre] = _q_por_koi(
            Path(salida) / nombre, protocolo["semillas_cabeza_fusion"], sel["kepoi_name"]
        )
    # Solo los KOIs donde todos los modelos tienen predicción (la fusión no ve los que no
    # entraron al dataset por falta de datos).
    comun = np.all([np.isfinite(v) for v in puntajes.values()], axis=0)
    positivo = (sel["clase"].to_numpy() == 0)[comun]
    grupos = sel["kepid"].to_numpy()[comun]
    puntajes = {nombre: v[comun] for nombre, v in puntajes.items()}
    cfg = protocolo["metricas"]

    def intervalo(estadistico) -> dict:
        r = estadistica.bootstrap_por_estrella(
            grupos, estadistico, cfg["bootstrap_remuestreos"], cfg["bootstrap_semilla"]
        )
        return {
            "estimacion": round(r.estimacion, 4),
            "ic_inf": round(r.ic_inf, 4),
            "ic_sup": round(r.ic_sup, 4),
        }

    def auc(nombre):
        return lambda i: metricas.auc(puntajes[nombre][i], positivo[i])

    resultado = {
        "n_kois": int(comun.sum()),
        "n_kois_sin_prediccion_de_la_fusion": int((~comun).sum()),
        "auc": {nombre: intervalo(auc(nombre)) for nombre in puntajes},
        "diferencias": {},
    }
    for nombre in configuraciones:
        for tabular in ("tabular_sin_centroide", "tabular_con_centroide"):
            resultado["diferencias"][f"{nombre} - {tabular}"] = intervalo(
                lambda i, a=nombre, b=tabular: auc(a)(i) - auc(b)(i)
            )
    if PRINCIPAL in configuraciones:
        decisiva = resultado["diferencias"][f"{PRINCIPAL} - tabular_sin_centroide"]
        resultado["criterio_declarado"] = {
            "regla": "la fusión supera de forma distinguible al tabular sin centroide",
            "diferencia": decisiva,
            "se_cumple": bool(decisiva["ic_inf"] > 0),
        }
    _escribir_json(Path(salida) / "comparacion_con_tabulares.json", resultado)
    return resultado


# ---------------------------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--koi", type=Path, required=True, help="tabla de KOIs DR25 (para los tabulares)"
    )
    ap.add_argument("--ramas", nargs="+", choices=list(REDES_DE_RAMA), default=list(REDES_DE_RAMA))
    ap.add_argument("--dataset", type=Path, default=RAIZ / "datos" / "dataset_temporal.npz")
    ap.add_argument("--ejemplos", type=Path, default=RAIZ / "datos" / "ejemplos.csv")
    ap.add_argument("--protocolo", type=Path, default=RAIZ / "config" / "protocolo.json")
    ap.add_argument(
        "--busqueda", type=Path, default=RAIZ / "evidencias" / "resultados" / "busqueda"
    )
    ap.add_argument(
        "--referencia",
        type=Path,
        default=RAIZ / "evidencias" / "resultados",
        help="resultados del protocolo fijo (modelos de la configuración 0)",
    )
    ap.add_argument("--modelos", type=Path, default=RAIZ / "modelos_finales")
    ap.add_argument("--salida", type=Path, default=RAIZ / "evidencias" / "resultados" / "fusion")
    a = ap.parse_args(argv)

    protocolo = cargar_protocolo(a.protocolo)
    with np.load(a.dataset, allow_pickle=False) as archivo:
        dataset = {clave: archivo[clave] for clave in archivo.files}
    ejemplos = pd.read_csv(a.ejemplos)
    n_particiones = protocolo["particiones"]["n"]

    for rama in a.ramas:
        print(f"=== Elegir y exportar: {rama} ===", flush=True)
        elegidas = elegir(a.busqueda, rama, n_particiones)
        exportar(rama, elegidas, a.busqueda, dataset, ejemplos, protocolo, a.modelos, a.referencia)

    # La rama de imagen (y cualquier otra no pedida) queda como ausente, no se inventa.
    versiones = {r: (VERSION if r in a.ramas else None) for r in fusionar.RAMAS}
    tabla = ent.alinear(ejemplos, np.array([str(i) for i in dataset["id_ejemplo"]], dtype="U"))
    hechas = []
    for nombre, ramas in CONFIGURACIONES.items():
        if not set(ramas) <= set(a.ramas):
            continue
        print(f"=== Fusión: {nombre} {ramas} ===", flush=True)
        resumen = fusionar.entrenar_fusion(
            nombre, ramas, a.modelos, dataset, tabla, protocolo, a.salida, versiones=versiones
        )
        print("  AUC por semilla:", resumen["auc_por_semilla"], flush=True)
        hechas.append(nombre)

    print("=== Comparación con los modelos tabulares ===", flush=True)
    resultado = comparar(a.salida, hechas, ejemplos, a.koi, protocolo)
    print(json.dumps(resultado, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
