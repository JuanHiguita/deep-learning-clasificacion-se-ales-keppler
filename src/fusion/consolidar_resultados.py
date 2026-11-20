"""Consolida todos los resultados en tablas: único origen de las cifras del informe (spec 007).

Uso:
  python -I src/fusion/consolidar_resultados.py --resultados resultados/dl --ejemplos compartido/ejemplos.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from comun import estadistica, metricas
from comun.protocolo import cargar_protocolo

NOMBRES = {
    "e6a_cuatro_ramas": "Fusión de 4 ramas (E6a)", "e6a_ajuste_cruzado": "Fusión con ajuste cruzado",
    "e6c_fusion_tardia": "Fusión tardía: promedio de probabilidades (E6c)",
    "e7_sin_vistas": "Sin rama 1 (vistas)", "e7_sin_transitos": "Sin rama 2 (tránsitos)",
    "e7_sin_imagen": "Sin rama 3 (imagen v0)", "e7_sin_escalares": "Sin rama 4 (escalares)",
    "e7_solo_vistas": "Solo rama 1 (vistas)", "e7_solo_transitos": "Solo rama 2 (tránsitos)",
    "e7_solo_imagen": "Solo rama 3 (imagen v0, demo)", "e7_solo_escalares": "Solo rama 4 (escalares)",
}


def _leer(ruta: Path):
    return json.loads(ruta.read_text(encoding="utf-8"))


def _pred(carpeta: Path, semillas) -> tuple[np.ndarray, list[np.ndarray]]:
    ps, ids = [], None
    for s in semillas:
        f = carpeta / f"pred_semilla{s}.npz"
        if f.is_file():
            with np.load(f, allow_pickle=False) as d:
                ids, p = d["id_ejemplo"].astype(str), d["p_prueba"].astype(float)
            ps.append(p)
    return ids, ps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--resultados", type=Path, required=True)
    ap.add_argument("--ejemplos", type=Path, required=True)
    ap.add_argument("--protocolo", type=Path, default=Path("config/protocolo.json"))
    a = ap.parse_args(argv)
    protocolo = cargar_protocolo(a.protocolo)
    cfg = protocolo["metricas"]
    R, T = a.resultados, a.resultados / "tablas"
    T.mkdir(parents=True, exist_ok=True)
    ej = pd.read_csv(a.ejemplos).set_index("id_ejemplo")

    def ic(grupos, fn):
        r = estadistica.bootstrap_por_estrella(grupos, fn, cfg["bootstrap_remuestreos"], cfg["bootstrap_semilla"])
        return round(r.estimacion, 4), round(r.ic_inf, 4), round(r.ic_sup, 4)

    # ---------- líneas base ----------
    lb = _leer(R / "lineas_base_tabulares" / "metricas.json")["modelos"]
    filas = [dict(modelo=k, auc=v["auc"], ic_inf=v["ic_inf"], ic_sup=v["ic_sup"], n=v["n"]) for k, v in lb.items()]
    for nombre, etiqueta in (("m0_demo_curva", "Demo vuelta a medir: solo curva (M0)"),
                             ("m0_demo_completo", "Demo vuelta a medir: imagen + curva (M0)"), ("imagen_v0", "Imagen v0 de la demo, sola")):
        ids, ps = _pred(R / nombre, protocolo["semillas"])
        y, g = ej.loc[ids, "clase"].to_numpy(), ej.loc[ids, "kepid"].to_numpy()
        e, lo, hi = ic(g, lambda i: float(np.mean([metricas.auc_planeta_fp(p[i], y[i]) for p in ps])))
        filas.append(dict(modelo=etiqueta, auc=e, ic_inf=lo, ic_sup=hi, n=int((y < 2).sum())))
    pd.DataFrame(filas).to_csv(T / "lineas_base.csv", index=False, lineterminator="\n")

    # ---------- P2: fusión y ablación ----------
    sem_f = protocolo["semillas_cabeza_fusion"]
    P, filas = {}, []
    for c in NOMBRES:
        ids, ps = _pred(R / "fusion" / c, sem_f)
        P[c] = ps
    y, g = ej.loc[ids, "clase"].to_numpy(), ej.loc[ids, "kepid"].to_numpy()
    fns = {"auc_planeta_fp": metricas.auc_planeta_fp, "nll": metricas.nll, "exactitud": metricas.exactitud,
           "ece_top": lambda p, yy: metricas.ece_top(p, yy, cfg["ece_intervalos"])}
    for c, ps in P.items():
        for m, fn in fns.items():
            e, lo, hi = ic(g, lambda i, fn=fn, ps=ps: float(np.mean([fn(p[i], y[i]) for p in ps])))
            filas.append(dict(tipo="modelo", modelo=NOMBRES[c], clave=c, metrica=m, estimacion=e, ic_inf=lo, ic_sup=hi, semillas=len(ps)))
    ref = "e6a_cuatro_ramas"
    for c in [k for k in NOMBRES if k != ref]:
        for m in ("auc_planeta_fp", "nll"):
            e, lo, hi = ic(g, estadistica.diferencia_media_semillas(fns[m], y, P[ref], P[c]))
            filas.append(dict(tipo="diferencia", modelo=f"Fusión (E6a) − {NOMBRES[c]}", clave=c, metrica=m, estimacion=e,
                              ic_inf=lo, ic_sup=hi, distinguible=not (lo <= 0 <= hi)))
    # fusión frente a la demo completa, en los ejemplos que tienen imagen
    ids_d, ps_d = _pred(R / "m0_demo_completo", protocolo["semillas"])
    pos = {i: j for j, i in enumerate(ids)}
    sel = np.array([pos[i] for i in ids_d])
    yd, gd = y[sel], g[sel]
    e, lo, hi = ic(gd, estadistica.diferencia_media_semillas(metricas.auc_planeta_fp, yd, [p[sel] for p in P[ref]], ps_d))
    filas.append(dict(tipo="diferencia", modelo="Fusión (E6a) − demo completa (M0), ejemplos con imagen", clave="m0_demo_completo",
                      metrica="auc_planeta_fp", estimacion=e, ic_inf=lo, ic_sup=hi, distinguible=not (lo <= 0 <= hi)))
    p2 = pd.DataFrame(filas)
    p2.to_csv(T / "p2_fusion.csv", index=False, lineterminator="\n")

    # ---------- P3 ----------
    cal, inc = _leer(R / "calibracion" / "metricas.json"), _leer(R / "incertidumbre" / "metricas.json")
    e8 = cal["E8_temperatura"]
    filas = [dict(experimento="E8", cantidad=m, antes=e8["antes"][m]["estimacion"], despues=e8["despues"][m]["estimacion"],
                  diferencia=e8["diferencia_despues_menos_antes"][m]["estimacion"],
                  ic_inf=e8["diferencia_despues_menos_antes"][m]["ic_inf"], ic_sup=e8["diferencia_despues_menos_antes"][m]["ic_sup"])
             for m in e8["antes"]]
    for n, v in cal["E9_prior"]["por_tasa"].items():
        d = v["diferencia_corregido_menos_sin"]
        filas.append(dict(experimento="E9", cantidad=f"ECE ponderado, tasa base {v['tasa']}", antes=v["ece_ponderado_sin_corregir"],
                          despues=v["ece_ponderado_corregido"], diferencia=d["estimacion"], ic_inf=d["ic_inf"], ic_sup=d["ic_sup"]))
    pd.DataFrame(filas).to_csv(T / "p3_calibracion.csv", index=False, lineterminator="\n")
    pd.DataFrame([{"metodo": k, **{kk: vv for kk, vv in v.items() if kk != "nll_ic"}, "nll_ic_inf": v["nll_ic"][0], "nll_ic_sup": v["nll_ic"][1]}
                  for k, v in inc.items() if isinstance(v, dict) and "nll" in v]).to_csv(T / "p3_incertidumbre.csv", index=False, lineterminator="\n")

    # ---------- veredicto mecánico de las hipótesis ----------
    p1 = pd.read_csv(T / "p1_arquitecturas.csv")
    perm = _leer(T / "p1_permutacion.json")

    def d1(nombre):
        f = p1[(p1.tipo == "diferencia") & (p1.modelo == nombre) & (p1.metrica == "auc_planeta_fp")].iloc[0]
        return f.estimacion, f.ic_inf, f.ic_sup

    def d2(clave, m="auc_planeta_fp"):
        f = p2[(p2.tipo == "diferencia") & (p2.clave == clave) & (p2.metrica == m)].iloc[0]
        return f.estimacion, f.ic_inf, f.ic_sup

    def fila(h, pregunta, cantidad, e, lo, hi, a_favor_si_positivo=True):
        sostiene = (lo > 0) if a_favor_si_positivo else (hi < 0)
        return dict(hipotesis=h, pregunta=pregunta, cantidad=cantidad, estimacion=e, ic_inf=lo, ic_sup=hi,
                    veredicto="se sostiene" if sostiene else "no se sostiene (el intervalo incluye el 0 o va en contra)")

    e10 = inc["diferencia_nll_ensamble_menos_mc"]
    e9 = cal["E9_prior"]["por_tasa"]["rejilla_0.05"]["diferencia_corregido_menos_sin"]
    dn = e8["diferencia_despues_menos_antes"]["nll"]
    H = [fila("P1a", "¿La lectura tránsito por tránsito supera a la curva plegada?", "AUC: rama 2 con atención − rama 1 (AstroNet)", *d1("m2_atencion − m1_astronet")),
         fila("P1b", "¿La atención aporta frente al promedio?", "AUC: atención − promedio", *d1("m2_atencion − m2_promedio")),
         fila("P1c", "¿La GRU supera a la atención?", "AUC: GRU − atención", *d1("m2_gru − m2_atencion")),
         fila("P2", "¿La fusión supera a la mejor rama sola?", "AUC: fusión − solo rama 4 (escalares)", *d2("e7_solo_escalares")),
         *[fila("P2", f"¿Aporta la rama? ({NOMBRES['e7_sin_' + r]})", f"AUC: fusión − sin {r}", *d2(f"e7_sin_{r}")) for r in ("vistas", "transitos", "imagen", "escalares")],
         fila("P2", "¿La fusión supera a la demo vuelta a medir?", "AUC: fusión − demo completa", *d2("m0_demo_completo")),
         fila("P3a", "¿Mejora el NLL con temperature scaling?", "NLL: después − antes", dn["estimacion"], dn["ic_inf"], dn["ic_sup"], False),
         fila("P3b", "¿El prior explícito mejora la calibración (tasa base 0,05)?", "ECE ponderado: corregido − sin corregir", e9["estimacion"], e9["ic_inf"], e9["ic_sup"], False),
         fila("P3c", "¿El ensamble supera a MC Dropout?", "NLL: ensamble − MC Dropout", e10["estimacion"], e10["ic_inf"], e10["ic_sup"], False)]
    h = pd.DataFrame(H)
    h.to_csv(T / "resumen_hipotesis.csv", index=False, lineterminator="\n")
    (T / "p1_permutacion_resumen.csv").write_text(
        "modelo,cambio_medio_abs_de_q,cambio_de_auc\n" + "".join(f"{k},{v['cambio_medio_abs_de_q']},{v['cambio_de_auc']}\n" for k, v in perm.items()), encoding="utf-8")
    print(p2[(p2.metrica == "auc_planeta_fp")][["tipo", "modelo", "estimacion", "ic_inf", "ic_sup"]].to_string(index=False))
    print(h[["hipotesis", "cantidad", "estimacion", "ic_inf", "ic_sup", "veredicto"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
