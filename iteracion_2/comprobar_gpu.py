"""Comprueba que el entorno de la iteración 2 entrena en GPU y lo deja por escrito.

Escribe iteracion_2/evidencias/entorno.json con: versiones, si CUDA está disponible, el nombre de
la tarjeta y cuánto tarda una época de la rama 2 (atención) en GPU y en CPU con datos sintéticos.

Uso:  python iteracion_2/comprobar_gpu.py
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
os.environ["KERAS_BACKEND"] = "torch"

import keras  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from temporal import modelos  # noqa: E402


def segundos_por_epoca(dispositivo: str, n: int = 2048) -> float:
    """Entrena 3 épocas de la rama 2 con datos sintéticos; devuelve el tiempo de la más rápida."""
    rng = np.random.default_rng(0)
    entradas = {
        "transitos": rng.normal(size=(n, 32, 61)).astype("float32"),
        "mascara_transitos": np.ones((n, 32), "float32"),
        "consistencia": rng.normal(size=(n, 4)).astype("float32"),
    }
    y = rng.integers(0, 3, n)
    with keras.device(dispositivo):
        modelo = modelos.m2(0, "atencion")
        modelo.compile(optimizer="adamw", loss="sparse_categorical_crossentropy")
        tiempos = []
        for _ in range(3):
            inicio = time.perf_counter()
            modelo.fit(entradas, y, epochs=1, batch_size=64, verbose=0)
            tiempos.append(time.perf_counter() - inicio)
    return round(min(tiempos), 3)


def main() -> int:
    hay_gpu = torch.cuda.is_available()
    informe = {
        "python": platform.python_version(),
        "sistema": platform.platform(),
        "keras": keras.__version__,
        "backend": keras.backend.backend(),
        "torch": torch.__version__,
        "cuda_disponible": hay_gpu,
        "gpu": torch.cuda.get_device_name(0) if hay_gpu else None,
        "memoria_gpu_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        if hay_gpu
        else None,
    }
    try:
        informe["segundos_por_epoca_cpu"] = segundos_por_epoca("cpu")
        if hay_gpu:
            informe["segundos_por_epoca_gpu"] = segundos_por_epoca("cuda")
    except Exception as error:  # se deja constancia del fallo en vez de cortar la preparación
        informe["error_al_entrenar"] = repr(error)[:500]
    destino = RAIZ / "iteracion_2" / "evidencias" / "entorno.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(informe, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(informe, ensure_ascii=False, indent=1))
    return 0 if hay_gpu else 1


if __name__ == "__main__":
    sys.exit(main())
