"""Análisis de las ramas temporales (spec 004): comparación pareada, selección del agregador,
prueba de permutación (E3), exportación v1 de la rama 2 y pesos de atención de casos al azar.

Uso:
  python -I src/temporal/analisis_ramas.py --dataset data/dataset_temporal.npz --ejemplos compartido/ejemplos.csv \
      --resultados resultados/dl --modelos modelos --pasos comparar seleccionar permutar exportar pesos
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from comun import entrenamiento as ent
from comun import estadistica, metricas
from comun.protocolo import cargar_protocolo, separar
from comun.utilidades import escribir_manifiesto
from temporal import modelos
from temporal.exp_ramas import EXPERIMENTOS

M2 = (
    "m2_promedio",
    "m2_atencion",
    "m2_gru",
)  # de más simple a más complejo: promedio < atención < GRU
RAMAS = ("m4_escalares", "m1_astronet", *M2)
PARES = (
    ("m2_promedio", "m1_astronet"),
    ("m2_atencion", "m1_astronet"),
    ("m2_gru", "m1_astronet"),
    ("m2_atencion", "m2_promedio"),
    ("m2_gru", "m2_atencion"),
    ("m2_gru", "m2_promedio"),
)


def cargar_predicciones(resultados: Path, nombre: str, semillas) -> list[dict]:
    return [
        dict(np.load(resultados / nombre / f"pred_semilla{s}.npz", allow_pickle=False))
        for s in semillas
    ]


def comparar(resultados: Path, tabla: pd.DataFrame, protocolo: dict) -> pd.DataFrame:
    """Métricas por modelo (media de semillas, IC por bootstrap de estrellas) y diferencias pareadas."""
    cfg, y, grupos = protocolo["metricas"], tabla["clase"].to_numpy(), tabla["kepid"].to_numpy()
    fns = {
        "auc_planeta_fp": metricas.auc_planeta_fp,
        "nll": metricas.nll,
        "ece_top": lambda p, yy: metricas.ece_top(p, yy, cfg["ece_intervalos"]),
        "exactitud": metricas.exactitud,
    }
    P = {
        n: [d["p_prueba"] for d in cargar_predicciones(resultados, n, protocolo["semillas"])]
        for n in RAMAS
    }
    filas = []

    def ic(est):
        r = estadistica.bootstrap_por_estrella(
            grupos, est, cfg["bootstrap_remuestreos"], cfg["bootstrap_semilla"]
        )
        return round(r.estimacion, 4), round(r.ic_inf, 4), round(r.ic_sup, 4)

    for n in RAMAS:
        m = json.loads((resultados / n / "metricas.json").read_text(encoding="utf-8"))
        for nombre, fn in fns.items():
            e, a, b = ic(lambda i, fn=fn, n=n: float(np.mean([fn(p[i], y[i]) for p in P[n]])))
            filas.append(
                dict(
                    tipo="modelo",
                    modelo=n,
                    metrica=nombre,
                    estimacion=e,
                    ic_inf=a,
                    ic_sup=b,
                    parametros=m["parametros"],
                    desviacion_semillas=m["auc_desviacion_semillas"]
                    if nombre == "auc_planeta_fp"
                    else None,
                )
            )
    for a_, b_ in PARES:
        for nombre in ("auc_planeta_fp", "nll"):
            e, lo, hi = ic(estadistica.diferencia_media_semillas(fns[nombre], y, P[a_], P[b_]))
            filas.append(
                dict(
                    tipo="diferencia",
                    modelo=f"{a_} − {b_}",
                    metrica=nombre,
                    estimacion=e,
                    ic_inf=lo,
                    ic_sup=hi,
                    distinguible=not (lo <= 0 <= hi),
                )
            )
    return pd.DataFrame(filas)


def seleccionar(resultados: Path, tabla: pd.DataFrame, protocolo: dict) -> dict:
    """Agregador de la rama 2 v1: menor NLL de VALIDACIÓN INTERNA; ante empate (1 error estándar), el más simple."""
    y = tabla["clase"].to_numpy()
    R = {}
    for n in M2:
        valores = []
        for d in cargar_predicciones(resultados, n, protocolo["semillas"]):
            for k in range(protocolo["particiones"]["n"]):
                m = d["particion_modelo_validacion"] == k
                valores.append(metricas.nll(d["p_validacion"][m], y[m]))
        R[n] = {
            "nll_validacion": round(float(np.mean(valores)), 4),
            "error_estandar": round(float(np.std(valores, ddof=1) / np.sqrt(len(valores))), 4),
            "n": len(valores),
        }
    mejor = min(M2, key=lambda n: R[n]["nll_validacion"])
    umbral = R[mejor]["nll_validacion"] + R[mejor]["error_estandar"]
    elegido = next(
        n for n in M2 if R[n]["nll_validacion"] <= umbral
    )  # el primero (más simple) dentro de 1 EE
    return {
        "candidatos": R,
        "menor_nll": mejor,
        "elegido": elegido,
        "regla": "menor NLL de validación interna; "
        "si uno más simple queda a menos de 1 error estándar, gana el más simple",
        "usa_datos_de_prueba": False,
    }


def _entradas_estandarizadas(exp, entradas, i_ent):
    x, est = dict(entradas), {}
    for clave in exp.estandarizar:
        media, desv = ent.estadisticas_estandarizacion(entradas[clave][i_ent])
        x[clave] = ent.aplicar_estandarizacion(entradas[clave], media, desv)
        est[clave] = (media, desv)
    return x, est


def _modelo(resultados: Path, nombre: str, k: int, semilla: int):
    import keras

    return keras.saving.load_model(
        resultados / nombre / "modelos" / f"particion_{k}_semilla_{semilla}.keras", safe_mode=True
    )


def permutar_transitos(x: dict, rng: np.random.Generator) -> dict:
    """Baraja el orden de los tránsitos presentes de cada ejemplo (los vacíos siguen al final)."""
    s = dict(x)
    tr = x["transitos"].copy()
    for i in range(len(tr)):
        n = int(x["mascara_transitos"][i].sum())
        tr[i, :n] = tr[i, :n][rng.permutation(n)]
    s["transitos"] = tr
    return s


def permutar(
    resultados: Path, entradas: dict, tabla: pd.DataFrame, protocolo: dict, n_perm: int = 10
) -> dict:
    """E3: ¿cambia la predicción al barajar el orden de los tránsitos en la prueba?"""
    y, part = tabla["clase"].to_numpy(), tabla["particion"].to_numpy()
    disp = entradas["disponible_transitos"].astype(bool)
    R = {}
    for nombre in M2:
        exp = EXPERIMENTOS[nombre][0]
        cambios, dif_auc = [], []
        for semilla in protocolo["semillas"]:
            p0 = np.load(resultados / nombre / f"pred_semilla{semilla}.npz")["p_prueba"].astype(
                float
            )
            perms = [p0.copy() for _ in range(n_perm)]
            for k in range(protocolo["particiones"]["n"]):
                sep = separar(part, k, protocolo["particiones"]["n"])
                x, _ = _entradas_estandarizadas(
                    exp, entradas, np.flatnonzero(sep.entrenamiento & disp)
                )
                idx = np.flatnonzero(sep.prueba & disp)
                modelo = _modelo(resultados, nombre, k, semilla)
                lote = {c: x[c][idx] for c in exp.claves_entrada}
                assert np.abs(modelo.predict(lote, verbose=0) - p0[idx]).max() < 1e-3, (
                    "el modelo guardado no reproduce la predicción"
                )
                rng = np.random.default_rng([semilla, k, 3])
                for j in range(n_perm):
                    perms[j][idx] = modelo.predict(permutar_transitos(lote, rng), verbose=0)
            q0 = metricas.q_planeta(p0)
            real = (y < 2) & disp
            cambios.append(
                float(
                    np.mean([np.abs(metricas.q_planeta(p)[real] - q0[real]).mean() for p in perms])
                )
            )
            dif_auc.append(
                float(
                    np.mean([metricas.auc_planeta_fp(p, y) for p in perms])
                    - metricas.auc_planeta_fp(p0, y)
                )
            )
        R[nombre] = {
            "cambio_medio_abs_de_q": round(float(np.mean(cambios)), 5),
            "cambio_de_auc": round(float(np.mean(dif_auc)), 4),
            "permutaciones": n_perm,
        }
    return R


def exportar(
    resultados: Path,
    destino: Path,
    nombre: str,
    entradas: dict,
    tabla: pd.DataFrame,
    protocolo: dict,
) -> None:
    """Exporta como rama 'transitos' v1 los modelos ya entrenados de la semilla congelada (sin reentrenar)."""
    exp = EXPERIMENTOS[nombre][0]
    ids = np.array([str(i) for i in entradas["id_ejemplo"]], dtype="U")
    disp, part = entradas["disponible_transitos"].astype(bool), tabla["particion"].to_numpy()
    semilla = protocolo["semilla_codificador_congelado"]
    for k in range(protocolo["particiones"]["n"]):
        sep = separar(part, k, protocolo["particiones"]["n"])
        x, est = _entradas_estandarizadas(exp, entradas, np.flatnonzero(sep.entrenamiento & disp))
        ent._exportar_particion(
            _modelo(resultados, nombre, k, semilla), x, exp, ids, k, est, destino
        )
    escribir_manifiesto(
        destino / "manifiesto.json",
        {"experimento": nombre, "version": destino.name, "semilla": semilla},
        extra={
            "entradas_modelo": list(exp.claves_entrada),
            "estandarizadas": list(exp.estandarizar),
        },
    )


def pesos_de_atencion(
    resultados: Path, entradas: dict, tabla: pd.DataFrame, protocolo: dict, por_clase: int = 4
) -> dict:
    """Pesos de atención de casos elegidos AL AZAR con semilla fija (no por haber acertado)."""
    exp = EXPERIMENTOS["m2_atencion"][0]
    y, part = tabla["clase"].to_numpy(), tabla["particion"].to_numpy()
    disp = entradas["disponible_transitos"].astype(bool)
    rng = np.random.default_rng(protocolo["muestras_al_azar"]["semilla"])
    elegidos = np.concatenate(
        [rng.choice(np.flatnonzero((y == c) & disp), por_clase, replace=False) for c in range(3)]
    )
    pesos, probs = (
        np.zeros((len(elegidos), 32), np.float32),
        np.zeros((len(elegidos), 3), np.float32),
    )
    for k in sorted(set(part[elegidos])):
        sep = separar(part, k, protocolo["particiones"]["n"])
        x, _ = _entradas_estandarizadas(exp, entradas, np.flatnonzero(sep.entrenamiento & disp))
        m = _modelo(resultados, "m2_atencion", int(k), 0)
        cuales = np.flatnonzero(part[elegidos] == k)
        lote = {c: x[c][elegidos[cuales]] for c in exp.claves_entrada}
        pesos[cuales] = modelos.submodelo_pesos_atencion(m).predict(lote, verbose=0)
        probs[cuales] = m.predict(lote, verbose=0)
    return {
        "id_ejemplo": np.array([str(i) for i in entradas["id_ejemplo"][elegidos]], dtype="U"),
        "clase": y[elegidos],
        "pesos": pesos,
        "p": probs,
        "transitos": entradas["transitos"][elegidos],
        "mascara_transitos": entradas["mascara_transitos"][elegidos],
        "paridad_transitos": entradas["paridad_transitos"][elegidos],
        "vista_local": entradas["vista_local"][elegidos],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--ejemplos", type=Path, required=True)
    ap.add_argument("--resultados", type=Path, required=True)
    ap.add_argument("--modelos", type=Path, default=Path("modelos"))
    ap.add_argument(
        "--pasos",
        nargs="+",
        required=True,
        choices=["comparar", "seleccionar", "permutar", "exportar", "pesos"],
    )
    ap.add_argument("--protocolo", type=Path, default=Path("config/protocolo.json"))
    a = ap.parse_args(argv)
    protocolo = cargar_protocolo(a.protocolo)
    with np.load(a.dataset, allow_pickle=False) as d:
        entradas = {c: d[c] for c in d.files}
    tabla = ent.alinear(pd.read_csv(a.ejemplos), entradas["id_ejemplo"])
    tablas = a.resultados / "tablas"
    tablas.mkdir(parents=True, exist_ok=True)
    if "comparar" in a.pasos:
        t = comparar(a.resultados, tabla, protocolo)
        t.to_csv(tablas / "p1_arquitecturas.csv", index=False, lineterminator="\n")
        print(t[t.metrica == "auc_planeta_fp"].drop(columns=["metrica"]).to_string(index=False))
    if "seleccionar" in a.pasos:
        s = seleccionar(a.resultados, tabla, protocolo)
        (a.resultados / "seleccion_rama2.json").write_text(
            json.dumps(s, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        print(json.dumps(s, ensure_ascii=False))
    if "permutar" in a.pasos:
        r = permutar(a.resultados, entradas, tabla, protocolo)
        (tablas / "p1_permutacion.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        print(json.dumps(r, ensure_ascii=False))
    if "exportar" in a.pasos:
        elegido = json.loads((a.resultados / "seleccion_rama2.json").read_text(encoding="utf-8"))[
            "elegido"
        ]
        exportar(a.resultados, a.modelos / "transitos" / "v1", elegido, entradas, tabla, protocolo)
        print("exportado:", elegido, "→", a.modelos / "transitos" / "v1")
    if "pesos" in a.pasos:
        np.savez_compressed(
            a.resultados / "m2_atencion" / "pesos_atencion_casos.npz",
            **pesos_de_atencion(a.resultados, entradas, tabla, protocolo),
        )
        print("pesos de atención guardados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
