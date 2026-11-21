"""Iteración 2: prueba corta antes de la búsqueda larga. Tarda alrededor de un minuto.

Comprueba, con el motor real (Keras sobre PyTorch) y datos aleatorios con la forma de los reales:
  1. que todas las redes del espacio ampliado se construyen, entrenan, guardan y vuelven a cargar;
  2. cuánto acelera plegar los tránsitos en el lote (rama 2) frente a `TimeDistributed`.
Escribe `evidencias/prueba_de_humo.json`. Si algo falla, termina con error y la búsqueda no arranca.

Uso: python -I iteracion_2/prueba_de_humo.py [--salida iteracion_2/evidencias/prueba_de_humo.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402
import numpy as np  # noqa: E402

from comun import busqueda  # noqa: E402
from comun import entrenamiento as ent  # noqa: E402
from temporal import busqueda_it2, busqueda_ramas, modelos  # noqa: E402

N_EJEMPLOS, LOTE = 1024, 128


def _datos(modelo: keras.Model, rng: np.random.Generator):
    x = [rng.normal(size=(N_EJEMPLOS, *e.shape[1:])).astype("float32") for e in modelo.inputs]
    for i, entrada in enumerate(modelo.inputs):
        if entrada.name in ("mascara_transitos", "faltantes"):
            x[i] = (x[i] > 0).astype("float32")
    return x, rng.integers(0, 3, N_EJEMPLOS)


def _segundos_por_epoca(modelo: keras.Model, x, y, perdida: str = "ce", epocas: int = 3) -> float:
    modelo.compile(
        optimizer=ent.crear_optimizador({"tasa_aprendizaje": 1e-3, "weight_decay": 1e-4}),
        loss=ent.crear_perdida({"perdida": perdida}),
    )
    modelo.fit(x, y, epochs=1, batch_size=LOTE, verbose=0)  # calentamiento
    inicio = time.perf_counter()
    modelo.fit(x, y, epochs=epocas, batch_size=LOTE, verbose=0)
    return (time.perf_counter() - inicio) / epocas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--salida", type=Path, default=Path("iteracion_2/evidencias/prueba_de_humo.json")
    )
    a = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    informe = {"motor": keras.backend.backend(), "redes": {}, "plegado": {}}

    for arquitectura in busqueda_it2.ESPACIOS:
        configuraciones = busqueda_ramas.configuraciones_de(
            arquitectura, busqueda_it2.N_CONFIGURACIONES, busqueda_it2.ESPACIOS
        )
        # La configuración con más parámetros: si esa entrena y se guarda, las demás también.
        candidatas = [
            busqueda_it2.CONSTRUCTORES[arquitectura](0, **busqueda.de_arquitectura(c))
            for c in configuraciones
        ]
        mayor = int(np.argmax([m.count_params() for m in candidatas]))
        modelo = candidatas[mayor]
        x, y = _datos(modelo, rng)
        segundos = _segundos_por_epoca(modelo, x, y, perdida="suavizado")
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "modelo.keras"
            modelo.save(ruta)
            recargado = keras.saving.load_model(ruta, compile=False, safe_mode=True)
        diferencia = float(
            np.abs(
                modelo.predict(x, verbose=0, batch_size=256)
                - recargado.predict(x, verbose=0, batch_size=256)
            ).max()
        )
        if diferencia > 1e-5:
            raise RuntimeError(f"{arquitectura}: el modelo recargado difiere ({diferencia:.2g})")
        informe["redes"][arquitectura] = {
            "configuracion_mayor": busqueda.nombre_de(mayor),
            "parametros": [int(m.count_params()) for m in candidatas],
            "segundos_por_epoca_1024_ejemplos": round(segundos, 3),
            "diferencia_al_recargar": diferencia,
        }
        print(arquitectura, informe["redes"][arquitectura], flush=True)

    for agregador in modelos.AGREGADORES:
        tiempos = {}
        for plegado in (False, True):
            modelo = modelos.m2(0, agregador, plegado=plegado)
            x, y = _datos(modelo, np.random.default_rng(1))
            tiempos["plegado" if plegado else "time_distributed"] = round(
                _segundos_por_epoca(modelo, x, y), 3
            )
        informe["plegado"][agregador] = tiempos
        print("rama 2", agregador, tiempos, flush=True)

    a.salida.parent.mkdir(parents=True, exist_ok=True)
    a.salida.write_text(json.dumps(informe, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("PRUEBA DE HUMO SUPERADA")
    return 0


if __name__ == "__main__":
    sys.exit(main())
