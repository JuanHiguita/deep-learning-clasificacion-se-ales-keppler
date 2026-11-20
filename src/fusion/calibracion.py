"""Calibración del sistema (spec 006): temperature scaling (E8) y prior explícito (E9).

E8 · Temperature scaling: se divide el logaritmo de las probabilidades por una temperatura T
     (T > 1 las suaviza, T < 1 las afila). La T de cada partición se ajusta SOLO con su
     validación interna y se aplica a su prueba.
E9 · Prior explícito: se corrige q = P(planeta | hay señal) para una tasa base de planetas
     distinta de la del entrenamiento (Saerens et al., 2002) y se mide el efecto.

Uso:
  python -I src/fusion/calibracion.py --predicciones resultados/dl/fusion/e6a_cuatro_ramas \
      --ejemplos compartido/ejemplos.csv --koi data/koi_dr25.csv --salida resultados/dl/calibracion
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

NOTA_E9 = (
    "E9 supone que solo cambia la tasa base (no el aspecto de las señales); "
    "no valida el régimen de zona habitable."
)
N_PRIMEROS = 20  # tamaño de la lista de candidatos prioritarios que se compara en E9


# ---------------------------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------------------------


def aplicar_temperatura(p: np.ndarray, temperatura: float) -> np.ndarray:
    """softmax(log p / T): reescala la confianza sin cambiar el orden de las clases."""
    z = np.log(np.clip(p, 1e-12, 1.0)) / temperatura
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ajustar_temperatura(p_validacion: np.ndarray, y_validacion: np.ndarray) -> float:
    """Temperatura que minimiza el NLL en la validación interna. No recibe datos de prueba."""

    def nll_con(log_temperatura: float) -> float:
        p = aplicar_temperatura(p_validacion, np.exp(log_temperatura))
        return metricas.nll(p, y_validacion)

    optimo = minimize_scalar(nll_con, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(optimo.x))


def calibrar_por_particion(
    pred: dict, y: np.ndarray, n_part: int
) -> tuple[np.ndarray, list[float]]:
    """Probabilidades de prueba calibradas y la temperatura de cada partición."""
    p_calibrada, temperaturas = pred["p_prueba"].astype(float).copy(), []
    for k in range(n_part):
        validacion = pred["particion_modelo_validacion"] == k
        prueba = pred["particion"] == k
        temperatura = ajustar_temperatura(
            pred["p_validacion"][validacion].astype(float), y[validacion]
        )
        p_calibrada[prueba] = aplicar_temperatura(
            pred["p_prueba"][prueba].astype(float), temperatura
        )
        temperaturas.append(round(temperatura, 3))
    return p_calibrada, temperaturas


def tasa_base_dr25(
    koi: pd.DataFrame, periodo_min: float = 1.0, periodo_max: float = 30.0
) -> tuple[float, int, int]:
    """Confirmados / (confirmados + FP) en DR25 dentro del rango de periodo.

    Se calcula de la tabla, nunca se escribe a mano. Devuelve (tasa, confirmados, FP).
    """
    en_rango = koi["koi_period"].between(periodo_min, periodo_max)
    confirmados = int((en_rango & (koi["koi_disposition"] == "CONFIRMED")).sum())
    falsos = int((en_rango & (koi["koi_disposition"] == "FALSE POSITIVE")).sum())
    return confirmados / (confirmados + falsos), confirmados, falsos


def _intervalo(grupos, estadistico, cfg) -> dict:
    r = estadistica.bootstrap_por_estrella(
        grupos, estadistico, cfg["bootstrap_remuestreos"], cfg["bootstrap_semilla"]
    )
    return {
        "estimacion": round(r.estimacion, 4),
        "ic_inf": round(r.ic_inf, 4),
        "ic_sup": round(r.ic_sup, 4),
    }


# ---------------------------------------------------------------------------------------------
# E8: ¿mejora la calibración con temperature scaling?
# ---------------------------------------------------------------------------------------------


def experimento_e8(
    p_antes: np.ndarray, p_despues: np.ndarray, temperaturas, y, grupos, cfg: dict
) -> dict:
    """Métricas antes y después de calibrar, y su diferencia pareada con intervalo."""
    n_intervalos = cfg["ece_intervalos"]
    funciones = {
        "nll": metricas.nll,
        "brier": metricas.brier,
        "ece_top": lambda p, yy: metricas.ece_top(p, yy, n_intervalos),
        "exactitud": metricas.exactitud,
        "auc_planeta_fp": metricas.auc_planeta_fp,
    }
    resultado = {
        "temperaturas": temperaturas,
        "antes": {},
        "despues": {},
        "diferencia_despues_menos_antes": {},
    }
    for nombre, f in funciones.items():
        resultado["antes"][nombre] = _intervalo(grupos, lambda i, f=f: f(p_antes[i], y[i]), cfg)
        resultado["despues"][nombre] = _intervalo(grupos, lambda i, f=f: f(p_despues[i], y[i]), cfg)
        resultado["diferencia_despues_menos_antes"][nombre] = _intervalo(
            grupos, lambda i, f=f: f(p_despues[i], y[i]) - f(p_antes[i], y[i]), cfg
        )
    return resultado


# ---------------------------------------------------------------------------------------------
# E9: prior explícito sobre q = P(planeta | hay señal)
# ---------------------------------------------------------------------------------------------


def _tasa_de_entrenamiento(tabla: pd.DataFrame, y, real, n_part: int) -> np.ndarray:
    """Fracción de planetas con la que se entrenó el modelo que predijo cada señal real."""
    particion = tabla["particion"].to_numpy()
    tasa = np.zeros(int(real.sum()))
    for k in range(n_part):
        sep = separar(particion, k, n_part)
        tasa[particion[real] == k] = (y[sep.entrenamiento & real] == 0).mean()
    return tasa


def _efecto_de_una_tasa(tasa: float, q, tasa_ent, es_planeta, p_real, grupos, cfg: dict) -> dict:
    """Corrige q para `tasa` y mide el cambio en calibración, umbrales y orden de candidatos."""
    n_intervalos = cfg["ece_intervalos"]
    pares = zip(q, tasa_ent, strict=True)
    q_corregida = np.array(
        [metricas.corregir_prior(np.array([q_i]), tasa_i, tasa)[0] for q_i, tasa_i in pares]
    )

    def ece_ponderado(i, valores):
        """ECE con los ejemplos ponderados para que la muestra tenga la tasa base objetivo."""
        pesos = metricas.pesos_para_tasa_base(es_planeta[i], tasa)
        return metricas.ece_binario(valores[i], es_planeta[i], n_intervalos, pesos)

    def diferencia(i):
        if not 0 < es_planeta[i].sum() < len(i):  # remuestreo con una sola clase
            return np.nan
        return ece_ponderado(i, q_corregida) - ece_ponderado(i, q)

    def primeros(valores) -> set:
        return set(np.argsort(-valores)[:N_PRIMEROS])

    # efecto sobre el orden de P(planeta) de 3 clases: p0' = q' · (1 − p_sin_senal)
    orden_antes = p_real[:, 0]
    orden_despues = q_corregida * (1 - p_real[:, 2])
    todos = np.arange(len(q))
    return {
        "tasa": round(float(tasa), 4),
        "ece_ponderado_sin_corregir": round(ece_ponderado(todos, q), 4),
        "ece_ponderado_corregido": round(ece_ponderado(todos, q_corregida), 4),
        "diferencia_corregido_menos_sin": _intervalo(grupos, diferencia, cfg),
        "candidatos_con_q_mayor_0.5": [int((q > 0.5).sum()), int((q_corregida > 0.5).sum())],
        "candidatos_con_q_mayor_0.9": [int((q > 0.9).sum()), int((q_corregida > 0.9).sum())],
        "tau_kendall_orden": round(float(kendalltau(orden_antes, orden_despues).statistic), 4),
        "cambios_en_los_20_primeros": int(len(primeros(orden_antes) - primeros(orden_despues))),
    }


def experimento_e9(
    p_calibrada: np.ndarray, tabla: pd.DataFrame, koi: pd.DataFrame, protocolo: dict
) -> dict:
    """Efecto de corregir el prior para la tasa de DR25 y para una rejilla de tasas."""
    cfg, n_part = protocolo["metricas"], protocolo["particiones"]["n"]
    y, grupos = tabla["clase"].to_numpy().astype(int), tabla["kepid"].to_numpy()
    real = y < 2  # solo señales reales: planeta o falso positivo
    q = metricas.q_planeta(p_calibrada)[real]
    es_planeta = y[real] == 0
    tasa_ent = _tasa_de_entrenamiento(tabla, y, real, n_part)

    tasa_dr25, confirmados, falsos = tasa_base_dr25(koi)
    tasas = {
        "dr25_periodo_1_30_d": tasa_dr25,
        **{f"rejilla_{t}": t for t in protocolo["calibracion"]["rejilla_tasa_base"]},
    }
    return {
        "tasa_dr25": {
            "valor": round(tasa_dr25, 4),
            "confirmados": confirmados,
            "falsos_positivos": falsos,
        },
        "tasa_entrenamiento_media": round(float(tasa_ent.mean()), 4),
        "por_tasa": {
            nombre: _efecto_de_una_tasa(
                tasa, q, tasa_ent, es_planeta, p_calibrada[real], grupos[real], cfg
            )
            for nombre, tasa in tasas.items()
        },
    }


# ---------------------------------------------------------------------------------------------
# Programa
# ---------------------------------------------------------------------------------------------


def _imprimir_resumen(e8: dict, e9: dict) -> None:
    print("T:", e8["temperaturas"])
    for metrica in ("nll", "ece_top", "auc_planeta_fp"):
        antes, despues = e8["antes"][metrica]["estimacion"], e8["despues"][metrica]["estimacion"]
        print(metrica, antes, "→", despues, "dif", e8["diferencia_despues_menos_antes"][metrica])
    print("tasa DR25:", e9["tasa_dr25"])
    for nombre, v in e9["por_tasa"].items():
        print(
            nombre,
            v["ece_ponderado_sin_corregir"],
            "→",
            v["ece_ponderado_corregido"],
            v["diferencia_corregido_menos_sin"],
            v["candidatos_con_q_mayor_0.5"],
            v["tau_kendall_orden"],
        )


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
    ids = [str(i) for i in pred["id_ejemplo"]]
    tabla = pd.read_csv(a.ejemplos).set_index("id_ejemplo").loc[ids].reset_index()
    y, grupos = tabla["clase"].to_numpy().astype(int), tabla["kepid"].to_numpy()

    p_antes = pred["p_prueba"].astype(float)
    p_despues, temperaturas = calibrar_por_particion(pred, y, n_part)
    e8 = experimento_e8(p_antes, p_despues, temperaturas, y, grupos, cfg)
    e9 = experimento_e9(p_despues, tabla, pd.read_csv(a.koi), protocolo)

    a.salida.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.salida / "pred_calibrada.npz",
        id_ejemplo=pred["id_ejemplo"],
        particion=pred["particion"],
        p_antes=p_antes.astype(np.float32),
        p_despues=p_despues.astype(np.float32),
    )
    resultado = {"E8_temperatura": e8, "E9_prior": e9, "nota": NOTA_E9}
    (a.salida / "metricas.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    escribir_manifiesto(
        a.salida / "manifiesto.json",
        vars(a),
        entradas=[a.predicciones / "pred_semilla0.npz", a.koi],
    )
    _imprimir_resumen(e8, e9)
    return 0


if __name__ == "__main__":
    sys.exit(main())
