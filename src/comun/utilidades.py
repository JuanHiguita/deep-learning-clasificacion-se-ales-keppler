"""Utilidades compartidas: hashes, manifiestos, validación de entradas y descargas seguras."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HOSTS_PERMITIDOS = frozenset({
    "exoplanetarchive.ipac.caltech.edu",
    "mast.stsci.edu",
    "archive.stsci.edu",
})
# Solo ASCII: un carácter no ASCII en la cabecera hace que el servidor responda 403.
# Si el primero es rechazado (403), se prueba el siguiente; ambos identifican al proyecto.
AGENTES = (
    "transitos-kepler-dl/0.1 (proyecto academico UNAL; python-urllib)",
    "Mozilla/5.0 (compatible; transitos-kepler-dl/0.1; proyecto academico UNAL)",
)


def sha256_archivo(ruta: str | Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def validar_kepid(valor) -> int:
    """Devuelve el kepid como entero o lanza ValueError. Evita inyectar texto en rutas y consultas."""
    if isinstance(valor, bool):
        raise ValueError("kepid inválido")
    try:
        kepid = int(valor)
    except (TypeError, ValueError) as e:
        raise ValueError(f"kepid inválido: {valor!r}") from e
    if str(valor).strip() not in (str(kepid), f"{kepid}.0"):
        raise ValueError(f"kepid inválido: {valor!r}")
    if not 1 <= kepid <= 99_999_999:
        raise ValueError(f"kepid fuera de rango: {kepid}")
    return kepid


def ruta_dentro_de(base: str | Path, ruta: str | Path) -> Path:
    """Resuelve `ruta` y exige que quede dentro de `base` (sin escapes con '..')."""
    base_r = Path(base).resolve()
    ruta_r = (base_r / ruta).resolve() if not Path(ruta).is_absolute() else Path(ruta).resolve()
    if ruta_r != base_r and base_r not in ruta_r.parents:
        raise ValueError(f"La ruta {ruta} queda fuera de {base}")
    return ruta_r


def _commit_git() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False)
        return r.stdout.strip() or "sin-git"
    except (OSError, subprocess.SubprocessError):
        return "sin-git"


def _versiones() -> dict:
    versiones = {}
    for nombre in ("numpy", "pandas", "sklearn", "scipy", "keras", "lightkurve", "astropy"):
        try:
            modulo = __import__(nombre)
            versiones[nombre] = getattr(modulo, "__version__", "?")
        except Exception:  # noqa: BLE001 - un paquete ausente no debe impedir escribir el manifiesto
            versiones[nombre] = None
    return versiones


def escribir_manifiesto(ruta: str | Path, argumentos: dict, entradas: list[str | Path] | None = None,
                        extra: dict | None = None) -> dict:
    """Escribe un manifiesto JSON con lo necesario para regenerar un resultado."""
    manifiesto = {
        "fecha_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": _commit_git(),
        "argumentos": {k: str(v) for k, v in argumentos.items()},
        "entradas": {Path(e).as_posix(): sha256_archivo(e) for e in (entradas or []) if Path(e).is_file()},
        "versiones": _versiones(),
    }
    if extra:
        manifiesto.update(extra)
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifiesto, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")
    return manifiesto


def comprobar_url(url: str) -> str:
    """Solo HTTPS y solo los hosts de NASA/STScI permitidos."""
    partes = urllib.parse.urlsplit(url)
    if partes.scheme != "https" or partes.hostname not in HOSTS_PERMITIDOS:
        raise ValueError(f"URL no permitida: {url}")
    return url


def leer_url(url: str, espera_s: int = 120, reintentos: int = 3) -> bytes:
    """Descarga el contenido de una URL permitida, con reintentos y espera creciente."""
    comprobar_url(url)
    ultimo: Exception | None = None
    agente = 0
    for intento in range(reintentos + len(AGENTES) - 1):
        try:
            cabeceras = {"User-Agent": AGENTES[agente], "Accept": "*/*"}
            peticion = urllib.request.Request(url, headers=cabeceras)  # noqa: S310 - URL validada
            with urllib.request.urlopen(peticion, timeout=espera_s) as r:  # noqa: S310
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            ultimo = e
            if e.code == 403 and agente + 1 < len(AGENTES):
                agente += 1
                continue
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            ultimo = e
        time.sleep(min(5 * (intento + 1), 20))
    raise ConnectionError(f"No se pudo descargar {url}: {ultimo}")


def descargar_a_archivo(url: str, destino: str | Path, espera_s: int = 120, reintentos: int = 3) -> Path:
    """Descarga a un archivo temporal y lo renombra al final (un corte no deja archivos a medias)."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_name(destino.name + ".parcial")
    temporal.write_bytes(leer_url(url, espera_s, reintentos))
    temporal.replace(destino)
    return destino
