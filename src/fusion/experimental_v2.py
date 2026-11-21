"""Copia experimental del sistema con la rama 1 mejorada (spec 013). Reanudable.

Solo cambia la rama 1 (tanh en vez de ReLU, la variante con menor NLL de validación del estudio
de diseño). Todo se escribe en `modelos/vistas/v2/` y `resultados/dl/experimental_v2/`: el
resultado principal del proyecto no se toca.

Pasos:
    1. exportar el codificador v2 y comprobar que reproduce la variante ya entrenada
    2. fusión de 4 ramas (5 semillas de cabeza)
    3. temperatura por partición
    4. MC Dropout frente a ensamble
    5. comparación pareada v2 − v1

Uso:
  python -I src/fusion/experimental_v2.py --modelos modelos --dataset data/dataset_temporal.npz \
      --ejemplos compartido/ejemplos.csv --resultados resultados/dl [--presupuesto 150]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from comun import entrenamiento as ent  # noqa: E402
from comun import estadistica, metricas  # noqa: E402
from comun.protocolo import cargar_protocolo  # noqa: E402
from comun.utilidades import escribir_manifiesto  # noqa: E402
from fusion import calibracion, fusionar, incertidumbre  # noqa: E402
from fusion.vectores import RAMAS  # noqa: E402
from temporal import estudio_diseno  # noqa: E402

VARIANTE = "act_tanh"
VERSIONES = {**fusionar.VERSIONES, "vistas": "v2"}
CARPETA = "experimental_v2"
PRINCIPAL = fusionar.CONFIGURACION_PRINCIPAL


def exportar_rama_1(
    dataset: dict, ejemplos: pd.DataFrame, protocolo: dict, modelos: Path, resultados: Path
) -> None:
    """Entrena la variante (semilla 0) exportando el codificador y comprueba que es la misma."""
    destino = modelos / "vistas" / "v2"
    if (destino / "manifiesto.json").is_file():
        return
    variante = estudio_diseno.VARIANTES["m1_astronet"][VARIANTE]
    experimento = estudio_diseno.experimento_de("m1_astronet", "rama1_v2", variante)
    salida = resultados / CARPETA
    ent.entrenar_cv(
        experimento,
        dataset,
        ejemplos,
        protocolo,
        [0],
        salida,
        exportar=modelos / "vistas",
        version="v2",
        con_intervalos=False,
    )
    ya_entrenada = resultados / "diseno" / "m1_astronet" / VARIANTE / "pred_semilla0.npz"
    with (
        np.load(ya_entrenada, allow_pickle=False) as a,
        np.load(salida / "rama1_v2" / "pred_semilla0.npz", allow_pickle=False) as b,
    ):
        diferencia = float(np.abs(a["p_prueba"] - b["p_prueba"]).max())
    if diferencia > 1e-4:
        raise RuntimeError(
            f"El codificador v2 no reproduce la variante entrenada ({diferencia:.2g})"
        )


def _predicciones(carpeta: Path, semillas) -> list[np.ndarray]:
    return [np.load(carpeta / f"pred_semilla{s}.npz")["p_prueba"].astype(float) for s in semillas]


def comparar(resultados: Path, tabla: pd.DataFrame, protocolo: dict) -> pd.DataFrame:
    """v2 − v1 con las mismas estrellas remuestreadas: media de las 5 semillas de cabeza."""
    cfg, semillas = protocolo["metricas"], protocolo["semillas_cabeza_fusion"]
    y, grupos = tabla["clase"].to_numpy().astype(int), tabla["kepid"].to_numpy()
    v1 = _predicciones(resultados / "fusion" / PRINCIPAL, semillas)
    v2 = _predicciones(resultados / CARPETA / "fusion" / PRINCIPAL, semillas)
    funciones = {
        "auc_planeta_fp": metricas.auc_planeta_fp,
        "nll": metricas.nll,
        "ece_top": lambda p, yy: metricas.ece_top(p, yy, cfg["ece_intervalos"]),
        "exactitud": metricas.exactitud,
    }

    def intervalo(estadistico) -> tuple[float, float, float]:
        r = estadistica.bootstrap_por_estrella(
            grupos, estadistico, cfg["bootstrap_remuestreos"], cfg["bootstrap_semilla"]
        )
        return round(r.estimacion, 4), round(r.ic_inf, 4), round(r.ic_sup, 4)

    filas = []
    for metrica, f in funciones.items():

        def media(lista, f=f):
            return lambda i: float(np.mean([f(p[i], y[i]) for p in lista]))

        principal, experimental = intervalo(media(v1)), intervalo(media(v2))
        diferencia = intervalo(estadistica.diferencia_media_semillas(f, y, v2, v1))
        filas.append(
            {
                "metrica": metrica,
                "principal_v1": principal[0],
                "experimental_v2": experimental[0],
                "v2_ic_inf": experimental[1],
                "v2_ic_sup": experimental[2],
                "diferencia_v2_menos_v1": diferencia[0],
                "ic_inf": diferencia[1],
                "ic_sup": diferencia[2],
                "distinguible": not (diferencia[1] <= 0 <= diferencia[2]),
            }
        )
    resultado = pd.DataFrame(filas)
    resultado.to_csv(resultados / CARPETA / "comparacion.csv", index=False, lineterminator="\n")
    return resultado


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modelos", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--ejemplos", type=Path, required=True)
    ap.add_argument("--resultados", type=Path, required=True)
    ap.add_argument("--presupuesto", type=float, default=None, help="segundos para la fusión")
    ap.add_argument("--protocolo", type=Path, default=Path("config/protocolo.json"))
    a = ap.parse_args(argv)

    inicio = time.perf_counter()
    protocolo = cargar_protocolo(a.protocolo)
    manifiesto_base = json.loads(
        a.ejemplos.with_name("manifiesto_base.json").read_text(encoding="utf-8")
    )
    protocolo["terciles_mes"] = manifiesto_base["resumen"]["terciles_mes"]
    with np.load(a.dataset, allow_pickle=False) as archivo:
        dataset = {clave: archivo[clave] for clave in archivo.files}
    ejemplos = pd.read_csv(a.ejemplos)
    tabla = ent.alinear(ejemplos, dataset["id_ejemplo"])
    y, grupos = tabla["clase"].to_numpy().astype(int), tabla["kepid"].to_numpy()
    base = a.resultados / CARPETA

    # 1. Codificador v2
    exportar_rama_1(dataset, ejemplos, protocolo, a.modelos, a.resultados)

    # 2. Fusión con la rama 1 nueva y, como referencia, la rama 1 nueva sola con la misma cabeza
    for nombre in (PRINCIPAL, "e7_solo_vistas"):
        if (base / "fusion" / nombre / "metricas.json").is_file():
            continue
        resumen = fusionar.entrenar_fusion(
            nombre,
            fusionar.CONFIGURACIONES[nombre],
            a.modelos,
            dataset,
            tabla,
            protocolo,
            base / "fusion",
            a.presupuesto,
            inicio,
            versiones=VERSIONES,
        )
        if resumen is None:
            print("INCOMPLETO: pausa en", nombre)
            return 3
    fusion = base / "fusion" / PRINCIPAL

    # 3. Temperatura por partición (solo con validación interna)
    pred = dict(np.load(fusion / "pred_semilla0.npz", allow_pickle=False))
    p_antes = pred["p_prueba"].astype(float)
    p_despues, temperaturas = calibracion.calibrar_por_particion(
        pred, y, protocolo["particiones"]["n"]
    )
    e8 = calibracion.experimento_e8(
        p_antes, p_despues, temperaturas, y, grupos, protocolo["metricas"]
    )
    (base / "calibracion").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        base / "calibracion" / "pred_calibrada.npz",
        id_ejemplo=pred["id_ejemplo"],
        particion=pred["particion"],
        p_antes=p_antes.astype(np.float32),
        p_despues=p_despues.astype(np.float32),
    )
    (base / "calibracion" / "metricas.json").write_text(
        json.dumps({"E8_temperatura": e8}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    # 4. Incertidumbre
    miembros = incertidumbre.cargar_miembros(fusion, protocolo["semillas_cabeza_fusion"])
    mc = incertidumbre.pasadas_mc_dropout(
        a.modelos, fusion, dataset, tabla["particion"].to_numpy(), miembros[0], protocolo, VERSIONES
    )
    e10 = incertidumbre.experimento_e10(miembros, mc, y, grupos, protocolo["metricas"])
    (base / "incertidumbre").mkdir(parents=True, exist_ok=True)
    (base / "incertidumbre" / "metricas.json").write_text(
        json.dumps(e10, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    # 5. Comparación con el sistema principal
    comparacion = comparar(a.resultados, tabla, protocolo)
    escribir_manifiesto(
        base / "manifiesto.json",
        vars(a),
        extra={"variante_rama_1": VARIANTE, "versiones": VERSIONES, "ramas": list(RAMAS)},
    )
    print(comparacion.to_string(index=False))
    print("Temperaturas v2:", temperaturas)
    for metrica in ("ece_top", "nll"):
        print(metrica, "calibrado v2:", e8["despues"][metrica])
    return 0


if __name__ == "__main__":
    sys.exit(main())
