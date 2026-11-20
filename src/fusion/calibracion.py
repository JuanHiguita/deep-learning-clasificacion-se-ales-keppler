"""Calibración del sistema (spec 006): temperature scaling (E8) y prior explícito (E9).

La temperatura de cada partición se ajusta SOLO con su validación interna y se aplica a su prueba.

Uso:
  python -I src/fusion/calibracion.py --predicciones resultados/dl/fusion/e6a_cuatro_ramas --ejemplos compartido/ejemplos.csv \
      --koi data/koi_dr25.csv --salida resultados/dl/calibracion
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import kendalltau

from comun import estadistica, metricas
from comun.protocolo import cargar_protocolo, separar
from comun.utilidades import escribir_manifiesto


def aplicar_temperatura(p: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(p, 1e-12, 1.0)) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ajustar_temperatura(p_validacion: np.ndarray, y_validacion: np.ndarray) -> float:
    """Temperatura que minimiza el NLL en la validación interna. No recibe datos de prueba."""
    r = minimize_scalar(lambda lt: metricas.nll(aplicar_temperatura(p_validacion, np.exp(lt)), y_validacion),
                        bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(r.x))


def calibrar_por_particion(pred: dict, y: np.ndarray, n_part: int) -> tuple[np.ndarray, list[float]]:
    p_cal, temperaturas = pred["p_prueba"].astype(float).copy(), []
    for k in range(n_part):
        val = pred["particion_modelo_validacion"] == k
        T = ajustar_temperatura(pred["p_validacion"][val].astype(float), y[val])
        prueba = pred["particion"] == k
        p_cal[prueba] = aplicar_temperatura(pred["p_prueba"][prueba].astype(float), T)
        temperaturas.append(round(T, 3))
    return p_cal, temperaturas


def tasa_base_dr25(koi: pd.DataFrame, periodo_min: float = 1.0, periodo_max: float = 30.0) -> tuple[float, int, int]:
    """Confirmados / (confirmados + FP) en DR25 dentro del rango de periodo. Calculada, nunca escrita a mano."""
    m = koi["koi_period"].between(periodo_min, periodo_max)
    c = int((m & (koi["koi_disposition"] == "CONFIRMED")).sum())
    f = int((m & (koi["koi_disposition"] == "FALSE POSITIVE")).sum())
    return c / (c + f), c, f


def _ic(grupos, fn, cfg):
    r = estadistica.bootstrap_por_estrella(grupos, fn, cfg["bootstrap_remuestreos"], cfg["bootstrap_semilla"])
    return {"estimacion": round(r.estimacion, 4), "ic_inf": round(r.ic_inf, 4), "ic_sup": round(r.ic_sup, 4)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predicciones", type=Path, required=True)
    ap.add_argument("--ejemplos", type=Path, required=True)
    ap.add_argument("--koi", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--protocolo", type=Path, default=Path("config/protocolo.json"))
    a = ap.parse_args(argv)
    protocolo = cargar_protocolo(a.protocolo)
    cfg, n_part = protocolo["metricas"], protocolo["particiones"]["n"]
    pred = dict(np.load(a.predicciones / "pred_semilla0.npz", allow_pickle=False))
    tabla = pd.read_csv(a.ejemplos).set_index("id_ejemplo").loc[[str(i) for i in pred["id_ejemplo"]]].reset_index()
    y, grupos = tabla["clase"].to_numpy().astype(int), tabla["kepid"].to_numpy()
    nb = cfg["ece_intervalos"]

    # ---------- E8: temperature scaling ----------
    p0 = pred["p_prueba"].astype(float)
    p1, temperaturas = calibrar_por_particion(pred, y, n_part)
    fns = {"nll": metricas.nll, "brier": metricas.brier, "ece_top": lambda p, yy: metricas.ece_top(p, yy, nb),
           "exactitud": metricas.exactitud, "auc_planeta_fp": metricas.auc_planeta_fp}
    E8 = {"temperaturas": temperaturas, "antes": {}, "despues": {}, "diferencia_despues_menos_antes": {}}
    for nombre, fn in fns.items():
        E8["antes"][nombre] = _ic(grupos, lambda i, fn=fn: fn(p0[i], y[i]), cfg)
        E8["despues"][nombre] = _ic(grupos, lambda i, fn=fn: fn(p1[i], y[i]), cfg)
        E8["diferencia_despues_menos_antes"][nombre] = _ic(grupos, lambda i, fn=fn: fn(p1[i], y[i]) - fn(p0[i], y[i]), cfg)

    # ---------- E9: prior explícito sobre q = P(planeta | hay señal) ----------
    real = y < 2
    q = metricas.q_planeta(p1)[real]
    pos, g_real, part = (y[real] == 0), grupos[real], tabla["particion"].to_numpy()[real]
    pi_ent = np.zeros(len(q))
    for k in range(n_part):
        sep = separar(tabla["particion"].to_numpy(), k, n_part)
        pi_ent[part == k] = (y[sep.entrenamiento & real] == 0).mean()
    pi_dr25, n_conf, n_fp = tasa_base_dr25(pd.read_csv(a.koi))
    tasas = {"dr25_periodo_1_30_d": pi_dr25, **{f"rejilla_{t}": t for t in protocolo["calibracion"]["rejilla_tasa_base"]}}
    p3_sin = p1[real]
    E9 = {"tasa_dr25": {"valor": round(pi_dr25, 4), "confirmados": n_conf, "falsos_positivos": n_fp},
          "tasa_entrenamiento_media": round(float(pi_ent.mean()), 4), "por_tasa": {}}
    for nombre, pi in tasas.items():
        qc = np.array([metricas.corregir_prior(np.array([a_]), b_, pi)[0] for a_, b_ in zip(q, pi_ent)])

        def ece(i, v):
            return metricas.ece_binario(v[i], pos[i], nb, metricas.pesos_para_tasa_base(pos[i], pi))

        def dif(i):
            return ece(i, qc) - ece(i, q) if 0 < pos[i].sum() < len(i) else np.nan

        # efecto sobre el orden de P(planeta) de 3 clases: p0' = q' · (1 − p_sin_senal)
        antes, despues = p3_sin[:, 0], qc * (1 - p3_sin[:, 2])
        top = lambda v: set(np.argsort(-v)[:20])  # noqa: E731
        E9["por_tasa"][nombre] = {
            "tasa": round(float(pi), 4),
            "ece_ponderado_sin_corregir": round(ece(np.arange(len(q)), q), 4),
            "ece_ponderado_corregido": round(ece(np.arange(len(q)), qc), 4),
            "diferencia_corregido_menos_sin": _ic(g_real, dif, cfg),
            "candidatos_con_q_mayor_0.5": [int((q > 0.5).sum()), int((qc > 0.5).sum())],
            "candidatos_con_q_mayor_0.9": [int((q > 0.9).sum()), int((qc > 0.9).sum())],
            "tau_kendall_orden": round(float(kendalltau(antes, despues).statistic), 4),
            "cambios_en_los_20_primeros": int(len(top(antes) - top(despues))),
        }
    a.salida.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.salida / "pred_calibrada.npz", id_ejemplo=pred["id_ejemplo"], particion=pred["particion"],
                        p_antes=p0.astype(np.float32), p_despues=p1.astype(np.float32))
    R = {"E8_temperatura": E8, "E9_prior": E9,
         "nota": "E9 supone que solo cambia la tasa base (no el aspecto de las señales); no valida el régimen de zona habitable."}
    (a.salida / "metricas.json").write_text(json.dumps(R, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    escribir_manifiesto(a.salida / "manifiesto.json", vars(a), entradas=[a.predicciones / "pred_semilla0.npz", a.koi])
    print("T:", temperaturas)
    for m in ("nll", "ece_top", "auc_planeta_fp"):
        print(m, E8["antes"][m]["estimacion"], "→", E8["despues"][m]["estimacion"], "dif", E8["diferencia_despues_menos_antes"][m])
    print("tasa DR25:", E9["tasa_dr25"])
    for n, v in E9["por_tasa"].items():
        print(n, v["ece_ponderado_sin_corregir"], "→", v["ece_ponderado_corregido"], v["diferencia_corregido_menos_sin"],
              v["candidatos_con_q_mayor_0.5"], v["tau_kendall_orden"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
