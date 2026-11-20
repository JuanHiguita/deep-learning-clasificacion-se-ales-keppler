"""Verificación automática del contrato y de las reglas sin fuga (spec 001, RF-11).

Comprueba todo lo que ya exista en el repositorio y termina con código 1 si algo falla.
No importa Keras: solo abre CSV, JSON y NPZ.

Uso:  python -I src/comun/verificar_contrato.py --raiz .
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from comun.protocolo import cargar_protocolo, comprobar_sin_solape, separar
from comun.utilidades import sha256_archivo

PATRON_ID = re.compile(r"^K\d{5}\.\d{2}\|(transito|sin_senal)$")
RAMAS = ("vistas", "transitos", "imagen", "escalares")


def verificar_ejemplos(ejemplos: pd.DataFrame, protocolo: dict) -> list[str]:
    fallos = []
    n = protocolo["particiones"]["n"]
    if ejemplos["id_ejemplo"].duplicated().any():
        fallos.append("ejemplos.csv: hay id_ejemplo repetidos")
    malos = ejemplos.loc[~ejemplos["id_ejemplo"].astype(str).str.match(PATRON_ID), "id_ejemplo"]
    if len(malos):
        fallos.append(f"ejemplos.csv: {len(malos)} id_ejemplo con formato inválido (ej. {malos.iloc[0]!r})")
    if not set(ejemplos["clase"].unique()) <= {0, 1, 2}:
        fallos.append("ejemplos.csv: clases fuera de {0, 1, 2}")
    if not set(ejemplos["particion"].unique()) <= set(range(n)):
        fallos.append("ejemplos.csv: particiones fuera de rango")
    if ejemplos.groupby("kepid")["particion"].nunique().max() > 1:
        fallos.append("ejemplos.csv: una estrella aparece en más de una partición")
    sin_senal = ejemplos["id_ejemplo"].astype(str).str.endswith("|sin_senal")
    if ((ejemplos["clase"] == 2) != sin_senal).any():
        fallos.append("ejemplos.csv: la clase 2 no coincide con los ejemplos sin_senal")
    for k in range(n):
        try:
            sep = separar(ejemplos["particion"].values, k, n, ejemplos["prerregistrado"].values)
            comprobar_sin_solape(ejemplos["kepid"].values, sep)
        except ValueError as e:
            fallos.append(f"ejemplos.csv: {e}")
    return fallos


def verificar_manifiesto(raiz: Path) -> list[str]:
    ruta = raiz / "compartido" / "manifiesto_base.json"
    if not ruta.is_file():
        return ["falta compartido/manifiesto_base.json"]
    esperado = json.loads(ruta.read_text(encoding="utf-8")).get("sha256_ejemplos")
    real = sha256_archivo(raiz / "compartido" / "ejemplos.csv")
    return [] if esperado == real else ["ejemplos.csv no coincide con el SHA-256 de manifiesto_base.json"]


def verificar_dataset_temporal(ruta: Path, ejemplos: pd.DataFrame, protocolo: dict) -> list[str]:
    fallos = []
    with np.load(ruta, allow_pickle=False) as d:
        ids = d["id_ejemplo"].astype(str)
        n = len(ids)
        validos = set(ejemplos.loc[~ejemplos["prerregistrado"], "id_ejemplo"])
        if len(set(ids)) != n:
            fallos.append(f"{ruta.name}: id_ejemplo repetidos")
        fuera = set(ids) - validos
        if fuera:
            fallos.append(f"{ruta.name}: {len(fuera)} ejemplos prerregistrados o desconocidos (ej. {sorted(fuera)[0]})")
        for clave, forma in protocolo["formas"].items():
            if clave == "img":
                continue
            if clave not in d.files:
                fallos.append(f"{ruta.name}: falta la clave {clave}")
            elif tuple(d[clave].shape) != (n, *forma):
                fallos.append(f"{ruta.name}: {clave} tiene forma {d[clave].shape}, se esperaba {(n, *forma)}")
            elif not np.all(np.isfinite(d[clave])):
                fallos.append(f"{ruta.name}: {clave} contiene NaN o infinitos")
        if "nombres_escalares" not in d.files:
            fallos.append(f"{ruta.name}: falta nombres_escalares")
        else:
            nombres = [str(x) for x in d["nombres_escalares"]]
            prohibidas = set(protocolo["columnas_prohibidas_como_entrada"])
            for nombre in nombres:
                if nombre in prohibidas or nombre.startswith("koi_fpflag"):
                    fallos.append(f"{ruta.name}: FUGA, el escalar {nombre} está prohibido como entrada")
                elif nombre not in protocolo["escalares_permitidos"]:
                    fallos.append(f"{ruta.name}: el escalar {nombre} no está en la lista permitida")
    return fallos


def verificar_rama(carpeta: Path, rama: str, ejemplos: pd.DataFrame, protocolo: dict) -> list[str]:
    fallos = []
    dim = protocolo["dimension_vector"][rama]
    n_part = protocolo["particiones"]["n"]
    esperados = set(ejemplos.loc[~ejemplos["prerregistrado"], "id_ejemplo"])
    prerregistrados = set(ejemplos.loc[ejemplos["prerregistrado"], "id_ejemplo"])
    if not (carpeta / "manifiesto.json").is_file():
        fallos.append(f"{carpeta}: falta manifiesto.json")
    for k in range(n_part):
        if not (carpeta / f"particion_{k}.keras").is_file():
            fallos.append(f"{carpeta}: falta particion_{k}.keras")
        ruta = carpeta / f"vectores_particion_{k}.npz"
        if not ruta.is_file():
            fallos.append(f"{carpeta}: falta {ruta.name}")
            continue
        with np.load(ruta, allow_pickle=False) as d:
            ids = set(d["id_ejemplo"].astype(str))
            z = d["z"]
        if z.ndim != 2 or z.shape[1] != dim:
            fallos.append(f"{ruta}: z tiene forma {z.shape}, se esperaba (n, {dim})")
        if not np.all(np.isfinite(z)):
            fallos.append(f"{ruta}: z contiene NaN o infinitos")
        if ids & prerregistrados:
            fallos.append(f"{ruta}: contiene ejemplos prerregistrados")
        if rama != "imagen" and esperados - ids:
            fallos.append(f"{ruta}: faltan {len(esperados - ids)} ejemplos")
        if ids - esperados - prerregistrados:
            fallos.append(f"{ruta}: {len(ids - esperados - prerregistrados)} ejemplos desconocidos")
    return fallos


def verificar_predicciones(raiz: Path, ejemplos: pd.DataFrame) -> list[str]:
    fallos = []
    prerregistrados = set(ejemplos.loc[ejemplos["prerregistrado"], "id_ejemplo"])
    for ruta in sorted((raiz / "resultados").rglob("pred_semilla*.npz")):
        with np.load(ruta, allow_pickle=False) as d:
            if set(d["id_ejemplo"].astype(str)) & prerregistrados:
                fallos.append(f"{ruta}: contiene ejemplos prerregistrados")
    return fallos


def verificar(raiz: Path, protocolo: dict) -> tuple[list[str], list[str]]:
    """Devuelve (fallos, comprobaciones hechas)."""
    fallos, hechas = [], []
    ruta_ej = raiz / "compartido" / "ejemplos.csv"
    if not ruta_ej.is_file():
        return ["falta compartido/ejemplos.csv"], hechas
    ejemplos = pd.read_csv(ruta_ej, keep_default_na=False, na_values=[""], dtype={"tipo_fp": str})
    ejemplos["prerregistrado"] = ejemplos["prerregistrado"].astype(str).str.lower().isin(["true", "1"])
    fallos += verificar_ejemplos(ejemplos, protocolo)
    hechas.append("ejemplos.csv")
    fallos += verificar_manifiesto(raiz)
    hechas.append("manifiesto_base.json")
    ruta_ds = raiz / "data" / "dataset_temporal.npz"
    if ruta_ds.is_file():
        fallos += verificar_dataset_temporal(ruta_ds, ejemplos, protocolo)
        hechas.append("dataset_temporal.npz")
    for rama in RAMAS:
        for carpeta in sorted((raiz / "modelos" / rama).glob("v*")):
            fallos += verificar_rama(carpeta, rama, ejemplos, protocolo)
            hechas.append(f"modelos/{rama}/{carpeta.name}")
    if (raiz / "resultados").is_dir():
        fallos += verificar_predicciones(raiz, ejemplos)
        hechas.append("predicciones")
    return fallos, hechas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raiz", type=Path, default=Path("."))
    ap.add_argument("--protocolo", type=Path, default=None)
    a = ap.parse_args(argv)
    protocolo = cargar_protocolo(a.protocolo or a.raiz / "config" / "protocolo.json")
    fallos, hechas = verificar(a.raiz, protocolo)
    print("Comprobado:", ", ".join(hechas) or "nada")
    if fallos:
        print(f"CONTRATO: {len(fallos)} fallo(s)")
        for f in fallos:
            print("  -", f)
        return 1
    print("CONTRATO: correcto")
    return 0


if __name__ == "__main__":
    sys.exit(main())
