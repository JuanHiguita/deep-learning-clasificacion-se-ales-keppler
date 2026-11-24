"""Une las dos partes de dataset_temporal.npz y comprueba su huella SHA-256.

GitHub no admite archivos de más de 100 MB, así que el dataset completo (148 MB) se guarda partido en
`dataset_temporal.npz.parte0` y `.parte1`. Este script los une en `dataset_temporal.npz` (en esta misma
carpeta) y verifica que la huella coincide con la de `dataset_temporal.manifiesto.json`.

Uso (desde la raíz del repositorio):  python iteracion_2/datos/unir_dataset.py
"""

import hashlib
import json
import sys
from pathlib import Path

CARPETA = Path(__file__).resolve().parent


def main() -> int:
    partes = sorted(CARPETA.glob("dataset_temporal.npz.parte*"))
    if not partes:
        print("No encuentro las partes dataset_temporal.npz.parte*")
        return 1
    destino = CARPETA / "dataset_temporal.npz"
    huella = hashlib.sha256()
    with open(destino, "wb") as salida:
        for parte in partes:
            datos = parte.read_bytes()
            huella.update(datos)
            salida.write(datos)
    esperada = json.loads((CARPETA / "dataset_temporal.manifiesto.json").read_text(encoding="utf-8"))["sha256"]
    if huella.hexdigest() != esperada:
        destino.unlink()
        print("La huella NO coincide: el archivo unido se borró. Vuelve a descargar las partes.")
        return 1
    print(f"Listo: {destino} ({destino.stat().st_size / 1e6:.0f} MB), huella verificada.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
