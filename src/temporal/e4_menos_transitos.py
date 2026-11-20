"""E4 del plan: ¿qué pasa con cada agregador cuando hay menos tránsitos? (spec 010)

Se toman los modelos de la rama 2 SIN estadísticos de consistencia (E2b), ya entrenados, y en la
prueba se les dejan solo N tránsitos de cada señal, elegidos al azar con semilla fija. Así lo
único que cambia es cuántos tránsitos ve el agregador.

Por qué sin consistencia: esos 4 estadísticos se calculan con TODOS los tránsitos, y dejarlos
filtraría la información que se quiere quitar.

Límites: simula pocos tránsitos con señales de periodo corto (cada tránsito sigue teniendo buena
señal); los modelos se entrenaron con entre 4 y 32 tránsitos, así que N = 2 es extrapolar.

Uso:
  python -I src/temporal/e4_menos_transitos.py --dataset data/dataset_temporal.npz \
      --ejemplos compartido/ejemplos.csv --diseno resultados/dl/diseno --salida resultados/dl/e4
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
from scipy.stats import rankdata  # noqa: E402

from comun import entrenamiento as ent  # noqa: E402
from comun import estadistica, metricas  # noqa: E402
from comun.protocolo import cargar_protocolo  # noqa: E402
from comun.utilidades import escribir_manifiesto  # noqa: E402
from temporal import modelos as _capas_propias  # noqa: E402, F401  (registra las capas)

AGREGADORES = ("promedio", "gru", "atencion")
NUMEROS_DE_TRANSITOS = (32, 16, 8, 4, 2)
REPETICIONES = 5  # sorteos distintos de qué tránsitos se conservan
SEMILLA = 7


def conservar(mascara: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Máscara con, como mucho, `n` de los tránsitos válidos de cada señal (elegidos al azar)."""
    nueva = np.zeros_like(mascara)
    for i, fila in enumerate(mascara):
        validos = np.flatnonzero(fila > 0)
        elegidos = validos if len(validos) <= n else rng.choice(validos, n, replace=False)
        nueva[i, elegidos] = 1.0
    return nueva


def _auc_por_rangos(puntaje: np.ndarray, positivo: np.ndarray) -> float:
    """AUC (Mann-Whitney con empates a rango medio). Igual que `metricas.auc`, pero más rápido."""
    n_pos = int(positivo.sum())
    n_neg = len(positivo) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rangos = rankdata(puntaje)
    return float((rangos[positivo].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def cargar_modelos(carpeta: Path, particiones) -> dict[int, keras.Model]:
    """El modelo de cada partición (semilla 0) de una variante del estudio de diseño."""
    return {
        int(k): keras.saving.load_model(
            carpeta / "modelos" / f"particion_{k}_semilla_0.keras", safe_mode=True
        )
        for k in particiones
    }


def predecir(
    modelos_por_particion: dict, dataset: dict, particion: np.ndarray, mascara: np.ndarray
) -> np.ndarray:
    """Probabilidades de prueba: cada partición la predice su modelo, con la máscara dada."""
    p = np.zeros((len(particion), 3))
    transitos = dataset["transitos"] * mascara[:, :, None]  # un tránsito quitado vale 0
    for k, modelo in modelos_por_particion.items():
        de_prueba = particion == k
        lote = {"transitos": transitos[de_prueba], "mascara_transitos": mascara[de_prueba]}
        p[de_prueba] = modelo.predict(lote, verbose=0, batch_size=256)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--ejemplos", type=Path, required=True)
    ap.add_argument("--diseno", type=Path, required=True, help="carpeta del estudio de diseño")
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--protocolo", type=Path, default=Path("config/protocolo.json"))
    a = ap.parse_args(argv)

    cfg = cargar_protocolo(a.protocolo)["metricas"]
    with np.load(a.dataset, allow_pickle=False) as archivo:
        dataset = {clave: archivo[clave] for clave in archivo.files}
    tabla = ent.alinear(pd.read_csv(a.ejemplos), dataset["id_ejemplo"])

    # Solo señales reales que tienen tránsitos: aquí no hay probabilidad "por defecto"
    usar = dataset["disponible_transitos"].astype(bool) & (tabla["clase"].to_numpy() < 2)
    datos = {c: dataset[c][usar] for c in ("transitos", "mascara_transitos")}
    y = tabla["clase"].to_numpy()[usar].astype(int)
    grupos, particion = tabla["kepid"].to_numpy()[usar], tabla["particion"].to_numpy()[usar]

    # Paso 1 (lento, se guarda): q = P(planeta | hay señal) de cada modelo, N y repetición
    a.salida.mkdir(parents=True, exist_ok=True)
    ruta_q = a.salida / "q_planeta.npz"
    if ruta_q.is_file():
        with np.load(ruta_q, allow_pickle=False) as archivo:
            guardado = {clave: archivo[clave] for clave in archivo.files}
    else:
        guardado = {}
        mascaras = {}
        for n in NUMEROS_DE_TRANSITOS:
            rng = np.random.default_rng([SEMILLA, n])
            mascaras[n] = [
                conservar(datos["mascara_transitos"], n, rng) for _ in range(REPETICIONES)
            ]
        for agregador in AGREGADORES:
            carpeta = a.diseno / f"m2_{agregador}" / "sin_consistencia"
            de_cada_particion = cargar_modelos(carpeta, np.unique(particion))
            for n in NUMEROS_DE_TRANSITOS:
                q = [
                    metricas.q_planeta(predecir(de_cada_particion, datos, particion, m))
                    for m in mascaras[n]
                ]
                guardado[f"{agregador}_{n}"] = np.stack(q).astype(np.float32)
        np.savez_compressed(ruta_q, **guardado)
    es_planeta = y == 0
    # el AUC por rangos debe coincidir con el oficial del proyecto
    q_de_prueba = guardado[f"atencion_{NUMEROS_DE_TRANSITOS[0]}"][0]
    oficial = metricas.auc(q_de_prueba, es_planeta)
    if abs(_auc_por_rangos(q_de_prueba, es_planeta) - oficial) > 1e-9:
        raise RuntimeError("El AUC por rangos no coincide con comun.metricas.auc")

    def intervalo(estadistico) -> dict:
        r = estadistica.bootstrap_por_estrella(
            grupos, estadistico, cfg["bootstrap_remuestreos"], cfg["bootstrap_semilla"]
        )
        return {
            "estimacion": round(r.estimacion, 4),
            "ic_inf": round(r.ic_inf, 4),
            "ic_sup": round(r.ic_sup, 4),
        }

    def auc_medio(q_por_repeticion):
        """AUC promediado sobre las repeticiones, en los índices remuestreados."""
        return lambda i: float(
            np.mean([_auc_por_rangos(q[i], es_planeta[i]) for q in q_por_repeticion])
        )

    filas = []
    for n in NUMEROS_DE_TRANSITOS:
        for agregador in AGREGADORES:
            fila = intervalo(auc_medio(guardado[f"{agregador}_{n}"]))
            filas.append({"tipo": "modelo", "transitos": n, "modelo": agregador, **fila})
        for otro in ("promedio", "gru"):
            con_atencion, con_otro = guardado[f"atencion_{n}"], guardado[f"{otro}_{n}"]

            def diferencia(i, a_=con_atencion, b_=con_otro):
                return auc_medio(a_)(i) - auc_medio(b_)(i)

            fila = intervalo(diferencia)
            fila["distinguible"] = not (fila["ic_inf"] <= 0 <= fila["ic_sup"])
            filas.append(
                {"tipo": "diferencia", "transitos": n, "modelo": f"atencion − {otro}", **fila}
            )
    resultado = pd.DataFrame(filas)

    resultado.to_csv(a.salida / "e4_menos_transitos.csv", index=False, lineterminator="\n")
    resumen = {
        "n_senales": int(len(y)),
        "repeticiones": REPETICIONES,
        "nota": "Modelos sin estadísticos de consistencia (E2b), semilla 0. N = 2 es extrapolar.",
    }
    (a.salida / "resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    escribir_manifiesto(a.salida / "manifiesto.json", vars(a))
    print(resultado.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
