"""Auditoría automática del proyecto (spec 011, tarea T7.7 del plan).

Comprueba lo que se puede comprobar con código y deja constancia en
`resultados/dl/auditoria/`. Lo que exige criterio (alcance, redacción, decisiones) se revisa a
mano y se anota en `docs/AUDITORIA.md`.

Uso:
  python -I src/fusion/auditoria.py --raiz .
Termina con código 0 si no hay ninguna comprobación fallida.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from comun.protocolo import cargar_protocolo
from comun.utilidades import sha256_archivo

PROHIBIDAS = ("koi_score", "koi_fpflag", "koi_pdisposition")
# Archivos que SÍ pueden nombrar columnas prohibidas, y para qué
PERMITIDOS = {
    "comun/lineas_base_tabulares.py": "koi_score solo como comparador externo",
    "comun/construir_ejemplos.py": "banderas del catálogo como estratos de reporte",
    "comun/descargar_koi.py": "columnas que se descargan de la tabla",
    "comun/verificar_contrato.py": "lista de columnas prohibidas que se comprueba",
    "fusion/auditoria.py": "esta auditoría",
    "informe/secciones.py": "solo el nombre del comparador, para colorear la figura del listón",
    "cuaderno_demo/flujo.py": "solo el título de un paso (texto), no una columna de datos",
}
FRASES_VETADAS = {
    r"P\(habitable\)": "la salida se llama P(en zona habitable y rocoso)",
    r"\bno existe (ningún|ninguna|otro|otra) (trabajo|método|estudio|modelo)": (
        "la novedad se redacta como «no encontramos»"
    ),
    r"\bes el primer (modelo|trabajo|método)": "la novedad se redacta como «no encontramos»",
}
DOCUMENTO_DE_RESULTADOS = "docs/EXPLICACION_RESULTADOS.md"


class Auditoria:
    """Lista de comprobaciones con su resultado."""

    def __init__(self):
        self.filas = []

    def anotar(self, grupo: str, que: str, correcto: bool, detalle: str = "") -> None:
        self.filas.append(
            {"grupo": grupo, "comprobacion": que, "correcto": bool(correcto), "detalle": detalle}
        )

    @property
    def fallidas(self) -> list[dict]:
        return [f for f in self.filas if not f["correcto"]]


def _npz(ruta: Path) -> dict:
    with np.load(ruta, allow_pickle=False) as archivo:
        return {clave: archivo[clave] for clave in archivo.files}


def _con_coma(valor: float, decimales: int = 3) -> str:
    return f"{valor:.{decimales}f}".replace(".", ",")


def _redondeos(valor: float) -> tuple[str, ...]:
    """Formas válidas de escribir una cifra de la tabla con 3 decimales.

    Las tablas guardan 4 decimales; un valor que termina en 5 (0,8645) puede venir de 0,86447 o
    de 0,86452, así que se aceptan los dos redondeos.
    """
    return tuple(
        dict.fromkeys((_con_coma(valor), _con_coma(valor - 5e-5), _con_coma(valor + 5e-5)))
    )


# ---------------------------------------------------------------------------------------------
# Comprobaciones
# ---------------------------------------------------------------------------------------------


def fugas(a: Auditoria, raiz: Path, protocolo: dict) -> None:
    dataset = _npz(raiz / "data" / "dataset_temporal.npz")
    escalares = [str(n) for n in dataset["nombres_escalares"]]
    permitidos = set(protocolo["escalares_permitidos"])
    a.anotar(
        "fugas",
        "los escalares de entrada están en la lista permitida del protocolo",
        set(escalares) <= permitidos,
        ", ".join(escalares),
    )
    a.anotar(
        "fugas",
        "ninguna columna prohibida es un escalar de entrada",
        not any(p in nombre for nombre in escalares for p in PROHIBIDAS),
    )
    fuera_de_lugar = []
    for archivo in sorted((raiz / "src").rglob("*.py")):
        relativo = archivo.relative_to(raiz / "src").as_posix()
        if relativo.startswith("demo/") or relativo in PERMITIDOS:
            continue
        texto = archivo.read_text(encoding="utf-8")
        if any(p in texto for p in PROHIBIDAS):
            fuera_de_lugar.append(relativo)
    a.anotar(
        "fugas",
        "las columnas prohibidas solo se nombran en los archivos autorizados",
        not fuera_de_lugar,
        ", ".join(fuera_de_lugar) or "autorizados: " + "; ".join(PERMITIDOS),
    )


def particiones(a: Auditoria, raiz: Path) -> None:
    ejemplos = pd.read_csv(raiz / "compartido" / "ejemplos.csv")
    por_estrella = ejemplos.groupby("kepid").particion.nunique()
    a.anotar(
        "particiones",
        "cada estrella está en una sola partición",
        bool((por_estrella == 1).all()),
        f"{len(por_estrella)} estrellas",
    )
    a.anotar(
        "particiones",
        "ningún ejemplo prerregistrado entra a los datasets",
        not ejemplos.prerregistrado.astype(bool).any()
        or not set(ejemplos[ejemplos.prerregistrado.astype(bool)].id_ejemplo)
        & set(_npz(raiz / "data" / "dataset_temporal.npz")["id_ejemplo"].astype(str)),
        f"{int(ejemplos.prerregistrado.astype(bool).sum())} prerregistrados",
    )
    de_ejemplos = ejemplos.set_index("id_ejemplo").particion
    distintas, revisados = [], 0
    for ruta in sorted((raiz / "resultados" / "dl").rglob("pred_semilla*.npz")):
        if "parciales" in ruta.parts or "busqueda_demo" in ruta.parts:
            continue
        pred = _npz(ruta)
        if "particion" not in pred:
            continue
        revisados += 1
        ids = pred["id_ejemplo"].astype(str)
        if not np.array_equal(de_ejemplos.loc[ids].to_numpy(), pred["particion"]):
            distintas.append(ruta.relative_to(raiz).as_posix())
    a.anotar(
        "particiones",
        "todas las predicciones usan la partición de compartido/ejemplos.csv",
        not distintas,
        f"{revisados} archivos revisados" + (": " + ", ".join(distintas[:3]) if distintas else ""),
    )


def datos(a: Auditoria, raiz: Path) -> None:
    dataset = _npz(raiz / "data" / "dataset_temporal.npz")
    numericos = [c for c, v in dataset.items() if v.dtype.kind == "f"]
    con_nan = [c for c in numericos if not np.isfinite(dataset[c]).all()]
    a.anotar("datos", "las entradas no contienen NaN ni infinitos", not con_nan, ", ".join(con_nan))
    faltantes = dataset["faltantes"].astype(bool)
    a.anotar(
        "datos",
        "todo escalar que falta vale 0 Y tiene su bandera de faltante",
        bool((dataset["escalares"][faltantes] == 0).all()),
        f"{int(faltantes.sum())} valores faltantes marcados",
    )
    manifiesto = json.loads((raiz / "compartido" / "manifiesto_base.json").read_text("utf-8"))
    huella = sha256_archivo(raiz / "compartido" / "ejemplos.csv")
    a.anotar(
        "datos",
        "compartido/ejemplos.csv coincide con la huella de su manifiesto",
        huella in json.dumps(manifiesto),
        huella[:16] + "…",
    )


def reproducibilidad(a: Auditoria, raiz: Path) -> None:
    ruta = raiz / "resultados" / "dl" / "e2e" / "informe.json"
    if ruta.is_file():
        informe = json.loads(ruta.read_text(encoding="utf-8"))
        a.anotar(
            "reproducibilidad",
            "la prueba de extremo a extremo reproduce lo publicado",
            informe["correcto"],
            f"{len(informe['estrellas'])} estrellas, tolerancia {informe['tolerancia']}",
        )
    else:
        a.anotar("reproducibilidad", "existe el informe de la prueba de extremo a extremo", False)
    for carpeta in sorted((raiz / "modelos" / "mejores").glob("*/")):
        manifiesto = json.loads((carpeta / "manifiesto.json").read_text(encoding="utf-8"))
        correctas = all(
            sha256_archivo(carpeta / datos_k["archivo"]) == datos_k["sha256"]
            for datos_k in manifiesto["particiones"].values()
        )
        a.anotar(
            "reproducibilidad",
            f"los modelos guardados de {carpeta.name} coinciden con su huella",
            correctas,
        )


def cifras(a: Auditoria, raiz: Path) -> None:
    """Las cifras clave del documento de resultados deben aparecer tal como están en las tablas."""
    tablas = raiz / "resultados" / "dl" / "tablas"
    texto = (raiz / DOCUMENTO_DE_RESULTADOS).read_text(encoding="utf-8")
    p1 = pd.read_csv(tablas / "p1_arquitecturas.csv")
    p2 = pd.read_csv(tablas / "p2_fusion.csv")
    base = pd.read_csv(tablas / "lineas_base.csv")
    incertidumbre = pd.read_csv(tablas / "p3_incertidumbre.csv").set_index("metodo")
    calibracion = pd.read_csv(tablas / "p3_calibracion.csv")

    esperadas = {}
    for modelo in ("m1_astronet", "m2_promedio", "m2_gru", "m2_atencion", "m4_escalares"):
        fila = p1[(p1.tipo == "modelo") & (p1.modelo == modelo) & (p1.metrica == "auc_planeta_fp")]
        esperadas[f"AUC de {modelo}"] = _redondeos(fila.estimacion.iloc[0])
    for clave in ("e6a_cuatro_ramas", "e7_sin_escalares", "e7_sin_vistas", "e7_solo_escalares"):
        fila = p2[(p2.tipo == "modelo") & (p2.clave == clave) & (p2.metrica == "auc_planeta_fp")]
        esperadas[f"AUC de {clave}"] = _redondeos(fila.estimacion.iloc[0])
    fusion = p2[(p2.tipo == "modelo") & (p2.clave == "e6a_cuatro_ramas")]
    fila = fusion[fusion.metrica == "auc_planeta_fp"].iloc[0]
    esperadas["intervalo de la fusión"] = (f"{_con_coma(fila.ic_inf)}–{_con_coma(fila.ic_sup)}",)
    for _, fila in base.iterrows():
        esperadas[f"línea base: {fila.modelo[:28]}"] = _redondeos(fila.auc)
    for metodo in incertidumbre.index:
        esperadas[f"NLL de {metodo}"] = _redondeos(incertidumbre.loc[metodo, "nll"])
    for cantidad in ("ece_top", "nll"):
        fila = calibracion[(calibracion.experimento == "E8") & (calibracion.cantidad == cantidad)]
        esperadas[f"{cantidad} antes de calibrar"] = _redondeos(fila.antes.iloc[0])
        esperadas[f"{cantidad} después de calibrar"] = _redondeos(fila.despues.iloc[0])

    ausentes = {
        nombre: valores[0]
        for nombre, valores in esperadas.items()
        if not any(valor in texto for valor in valores)
    }
    a.anotar(
        "cifras",
        f"las {len(esperadas)} cifras clave de las tablas aparecen en {DOCUMENTO_DE_RESULTADOS}",
        not ausentes,
        "; ".join(f"{n} = {v}" for n, v in ausentes.items()),
    )


def redaccion(a: Auditoria, raiz: Path) -> None:
    documentos = [*sorted((raiz / "docs").rglob("*.md")), raiz / "README.md"]
    documentos += sorted((raiz / "scripts").glob("construir_notebook*.py"))
    textos = {d.relative_to(raiz).as_posix(): d.read_text(encoding="utf-8") for d in documentos}
    for patron, regla in FRASES_VETADAS.items():
        donde = [
            nombre
            for nombre, t in textos.items()
            # una línea que enuncia la regla ("nunca …") no la incumple
            if any(
                re.search(patron, linea, flags=re.IGNORECASE) and "nunca" not in linea.lower()
                for linea in t.splitlines()
            )
        ]
        a.anotar("redacción", f"no aparece «{patron}» ({regla})", not donde, ", ".join(donde))
    resultados = textos[DOCUMENTO_DE_RESULTADOS]
    a.anotar(
        "redacción",
        "el documento de resultados declara que la rama de imagen no es un aporte del curso",
        "No es un aporte de este curso" in resultados
        or "no es un aporte de este curso" in resultados,
    )
    a.anotar(
        "redacción",
        "el documento de resultados declara que no hay conclusiones sobre zona habitable",
        "zona habitable" in resultados and "Nada sobre el régimen de zona habitable" in resultados,
    )


def escribir(a: Auditoria, destino: Path) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    resumen = {
        "comprobaciones": len(a.filas),
        "fallidas": len(a.fallidas),
        "detalle": a.filas,
    }
    (destino / "auditoria.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    lineas = ["| Grupo | Comprobación | Resultado | Detalle |", "|---|---|---|---|"]
    for f in a.filas:
        marca = "✔" if f["correcto"] else "✘ **FALLA**"
        lineas.append(f"| {f['grupo']} | {f['comprobacion']} | {marca} | {f['detalle']} |")
    (destino / "comprobaciones.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raiz", type=Path, required=True)
    args = ap.parse_args(argv)
    raiz = args.raiz.resolve()
    protocolo = cargar_protocolo(raiz / "config" / "protocolo.json")

    a = Auditoria()
    fugas(a, raiz, protocolo)
    particiones(a, raiz)
    datos(a, raiz)
    reproducibilidad(a, raiz)
    cifras(a, raiz)
    redaccion(a, raiz)
    escribir(a, raiz / "resultados" / "dl" / "auditoria")

    for f in a.filas:
        print(
            "OK   " if f["correcto"] else "FALLA",
            f["grupo"],
            "·",
            f["comprobacion"],
            "·",
            f["detalle"][:90],
        )
    print(f"AUDITORÍA: {len(a.filas) - len(a.fallidas)} de {len(a.filas)} comprobaciones correctas")
    return 0 if not a.fallidas else 1


if __name__ == "__main__":
    sys.exit(main())
