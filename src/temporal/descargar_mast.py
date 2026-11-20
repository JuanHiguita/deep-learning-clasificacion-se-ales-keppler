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
import sys
import time
import urllib.error
from pathlib import Path

import pandas as pd

from comun.utilidades import descargar_a_archivo, leer_url, ruta_dentro_de, sha256_archivo, validar_kepid

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


def bajar_llc(kepid: int, destino: Path, pausa: float) -> list[Path]:
    url = carpeta_remota("lightcurves", kepid)
    nombres = archivos_llc(leer_url(url).decode("utf-8", "replace"), kepid)
    rutas = []
    for nombre in nombres:
        ruta = ruta_dentro_de(destino, Path(f"{kepid:09d}") / nombre)
        if not (ruta.is_file() and ruta.stat().st_size > 0):
            descargar_a_archivo(url + nombre, ruta)
            time.sleep(pausa)
        rutas.append(ruta)
    return rutas


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
    ap.add_argument("--destino", type=Path, required=True, help="carpeta data/; se crean llc/ y tpf/ dentro")
    ap.add_argument("--que", choices=["llc", "tpf"], required=True)
    ap.add_argument("--pausa", type=float, default=0.2, help="segundos entre archivos, para no saturar MAST")
    ap.add_argument("--max-estrellas", type=int, default=None, help="para pruebas")
    a = ap.parse_args(argv)

    tabla = pd.read_csv(a.ejemplos or a.seleccion)
    kepids = sorted({validar_kepid(k) for k in tabla["kepid"]})[: a.max_estrellas]
    carpeta = a.destino / a.que
    carpeta.mkdir(parents=True, exist_ok=True)
    bajar = bajar_llc if a.que == "llc" else bajar_tpf

    filas, fallidas = [], []
    for i, kepid in enumerate(kepids, 1):
        try:
            rutas = bajar(kepid, carpeta, a.pausa)
            for r in rutas:
                filas.append([kepid, r.name, r.stat().st_size, sha256_archivo(r)])
            print(f"[{i}/{len(kepids)}] KIC {kepid}: {len(rutas)} archivo(s)", flush=True)
        except urllib.error.HTTPError as e:
            fallidas.append((kepid, f"HTTP {e.code}"))
            print(f"[{i}/{len(kepids)}] KIC {kepid}: FALLO HTTP {e.code}", flush=True)
        except (ConnectionError, OSError) as e:
            fallidas.append((kepid, repr(e)[:80]))
            print(f"[{i}/{len(kepids)}] KIC {kepid}: FALLO {repr(e)[:80]}", flush=True)

    with open(carpeta / f"manifiesto_{a.que}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["kepid", "archivo", "bytes", "sha256"])
        w.writerows(filas)
    por_estrella = pd.DataFrame(filas, columns=["kepid", "archivo", "bytes", "sha256"]).groupby("kepid").size()
    print(f"\nEstrellas con archivos: {len(por_estrella)} de {len(kepids)}; archivos: {len(filas)}")
    if a.que == "llc":
        print("Estrellas con menos de 4 trimestres:", int((por_estrella < 4).sum()))
    if fallidas:
        print("Fallidas (vuelve a ejecutar el mismo comando para reintentar):", fallidas)
    return 1 if fallidas else 0


if __name__ == "__main__":
    sys.exit(main())
