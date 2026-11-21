"""Descarga desde MAST las curvas de luz de cadencia larga (Q1–Q17) y el TPF de Q5 de cada estrella.

Reanudable: lo que ya está en disco no se vuelve a bajar. Solo HTTPS a archive.stsci.edu.
Escribe un manifiesto CSV con tamaño y SHA-256 de cada archivo (spec 002, RF-1 y RF-2).

Uso:
  python -I src/temporal/descargar_mast.py --ejemplos compartido/ejemplos.csv --destino data --que llc
  python -I src/temporal/descargar_mast.py --ejemplos compartido/ejemplos.csv --destino data --que tpf
  (también acepta --seleccion data/seleccion_demo.csv si todavía no existe ejemplos.csv)
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from comun.utilidades import (
    descargar_a_archivo,
    leer_url,
    ruta_dentro_de,
    sha256_archivo,
    validar_kepid,
)

BASE = "https://archive.stsci.edu/missions/kepler"
PATRON_LLC = re.compile(r'href="(kplr(\d{9})-(\d{13})_llc\.fits)"')
# Marca de tiempo del trimestre 5 en cadencia larga (el que usó la demo).
MARCA_Q5 = "2010174085026"


def carpeta_remota(tipo: str, kepid: int) -> str:
    """URL de la carpeta de una estrella. tipo ∈ {'lightcurves', 'target_pixel_files'}."""
    if tipo not in ("lightcurves", "target_pixel_files"):
        raise ValueError(tipo)
    kic = f"{validar_kepid(kepid):09d}"
    return f"{BASE}/{tipo}/{kic[:4]}/{kic}/"


def archivos_llc(html: str, kepid: int) -> list[str]:
    """Nombres de los archivos de cadencia larga de la estrella, en orden cronológico."""
    kic = f"{validar_kepid(kepid):09d}"
    return sorted({m.group(1) for m in PATRON_LLC.finditer(html) if m.group(2) == kic})


def nombre_tpf_q5(kepid: int) -> str:
    return f"kplr{validar_kepid(kepid):09d}-{MARCA_Q5}_lpd-targ.fits.gz"


def _ya_esta(ruta: Path) -> bool:
    return ruta.is_file() and ruta.stat().st_size > 0


def bajar_llc(
    kepid: int, destino: Path, pausa: float, reutilizar: Path | None = None
) -> list[Path]:
    """Curvas de cadencia larga de una estrella. Reanudable por estrella y por archivo.

    Una estrella terminada deja un archivo `.completa`: al reanudar no se vuelve a consultar MAST.
    reutilizar: otra carpeta `llc/` ya descargada; lo que esté allí se copia en vez de bajarse.
    """
    carpeta = ruta_dentro_de(destino, f"{kepid:09d}")
    marca = carpeta / ".completa"
    if marca.is_file():
        return sorted(carpeta.glob("*_llc.fits"))
    carpeta.mkdir(parents=True, exist_ok=True)
    anterior = None if reutilizar is None else reutilizar / f"{kepid:09d}"
    if anterior is not None and any(anterior.glob("*_llc.fits")):
        # La carpeta a reutilizar es de una descarga ya terminada: se copia tal cual
        for origen in sorted(anterior.glob("*_llc.fits")):
            ruta = ruta_dentro_de(destino, Path(f"{kepid:09d}") / origen.name)
            if not _ya_esta(ruta):
                shutil.copyfile(origen, ruta)
    else:
        url = carpeta_remota("lightcurves", kepid)
        for nombre in archivos_llc(leer_url(url).decode("utf-8", "replace"), kepid):
            ruta = ruta_dentro_de(destino, Path(f"{kepid:09d}") / nombre)
            if not _ya_esta(ruta):
                descargar_a_archivo(url + nombre, ruta)
                time.sleep(pausa)
    marca.write_text("ok\n", encoding="utf-8")
    return sorted(carpeta.glob("*_llc.fits"))


def bajar_tpf(kepid: int, destino: Path, pausa: float) -> list[Path]:
    nombre = nombre_tpf_q5(kepid)
    # Prefijo 'demo_' para que src/demo/demo_dataset.py los encuentre sin cambios.
    ruta = ruta_dentro_de(destino, "demo_" + nombre)
    if not (ruta.is_file() and ruta.stat().st_size > 0):
        descargar_a_archivo(carpeta_remota("target_pixel_files", kepid) + nombre, ruta)
        time.sleep(pausa)
    return [ruta]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    origen = ap.add_mutually_exclusive_group(required=True)
    origen.add_argument("--ejemplos", type=Path)
    origen.add_argument("--seleccion", type=Path)
    ap.add_argument(
        "--destino", type=Path, required=True, help="carpeta data/; se crean llc/ y tpf/ dentro"
    )
    ap.add_argument("--que", choices=["llc", "tpf"], required=True)
    ap.add_argument(
        "--pausa", type=float, default=0.2, help="segundos entre archivos, para no saturar MAST"
    )
    ap.add_argument("--max-estrellas", type=int, default=None, help="para pruebas")
    ap.add_argument("--hilos", type=int, default=1, help="estrellas en paralelo (máximo 8)")
    ap.add_argument(
        "--reutilizar", type=Path, default=None, help="carpeta llc/ ya descargada, para copiar"
    )
    ap.add_argument("--sin-huellas", action="store_true", help="no calcular SHA-256 (más rápido)")
    a = ap.parse_args(argv)

    tabla = pd.read_csv(a.ejemplos or a.seleccion)
    kepids = sorted({validar_kepid(k) for k in tabla["kepid"]})[: a.max_estrellas]
    carpeta = a.destino / a.que
    carpeta.mkdir(parents=True, exist_ok=True)
    if not 1 <= a.hilos <= 8:
        raise SystemExit("--hilos debe estar entre 1 y 8 (para no saturar MAST)")

    def de_una_estrella(kepid: int):
        """(kepid, rutas, error). Un fallo de red no detiene a las demás estrellas."""
        try:
            if a.que == "llc":
                return kepid, bajar_llc(kepid, carpeta, a.pausa, a.reutilizar), None
            return kepid, bajar_tpf(kepid, carpeta, a.pausa), None
        except urllib.error.HTTPError as e:
            return kepid, [], f"HTTP {e.code}"
        except (ConnectionError, OSError) as e:
            return kepid, [], repr(e)[:80]

    filas, fallidas = [], []
    with ThreadPoolExecutor(max_workers=a.hilos) as hilos:
        for i, (kepid, rutas, error) in enumerate(hilos.map(de_una_estrella, kepids), 1):
            if error is not None:
                fallidas.append((kepid, error))
                print(f"[{i}/{len(kepids)}] KIC {kepid}: FALLO {error}", flush=True)
                continue
            for r in rutas:
                huella = "" if a.sin_huellas else sha256_archivo(r)
                filas.append([kepid, r.name, r.stat().st_size, huella])
            print(f"[{i}/{len(kepids)}] KIC {kepid}: {len(rutas)} archivo(s)", flush=True)
    with open(carpeta / f"manifiesto_{a.que}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["kepid", "archivo", "bytes", "sha256"])
        w.writerows(filas)
    por_estrella = (
        pd.DataFrame(filas, columns=["kepid", "archivo", "bytes", "sha256"]).groupby("kepid").size()
    )
    print(f"\nEstrellas con archivos: {len(por_estrella)} de {len(kepids)}; archivos: {len(filas)}")
    if a.que == "llc":
        print("Estrellas con menos de 4 trimestres:", int((por_estrella < 4).sum()))
    if fallidas:
        print("Fallidas (vuelve a ejecutar el mismo comando para reintentar):", fallidas)
    return 1 if fallidas else 0


if __name__ == "__main__":
    sys.exit(main())
