"""Evidencia cuantitativa a partir de la tabla KOI 'cumulative' completa (NASA Exoplanet Archive).

Uso: python -I evidencia_koi.py <koi.csv> <carpeta_salida>
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore")
csv, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
k = pd.read_csv(csv)
R = {"n_total": int(len(k))}
R["disposicion"] = k.koi_disposition.value_counts().to_dict()

# ---------- 1. Tipos de falso positivo (banderas del Robovetter) ----------
fp = k[k.koi_disposition == "FALSE POSITIVE"]
flags = {"nt (no parece tránsito)": "koi_fpflag_nt", "ss (eclipse secundario)": "koi_fpflag_ss",
         "co (centroide desplazado)": "koi_fpflag_co", "ec (contaminación de efemérides)": "koi_fpflag_ec"}
R["fp_n"] = int(len(fp))
R["fp_banderas_pct"] = {n: round(100 * fp[c].mean(), 1) for n, c in flags.items()}
R["fp_solo_co_pct"] = round(100 * ((fp.koi_fpflag_co == 1) & (fp.koi_fpflag_nt == 0) & (fp.koi_fpflag_ss == 0) & (fp.koi_fpflag_ec == 0)).mean(), 1)

# ---------- 2. Regímenes ----------
hz = (k.koi_insol.between(0.25, 2.2)) & (k.koi_prad <= 2.5)        # tamaño y zona habitable (aprox. optimista PHL)
corto = k.koi_period < 50
largo = k.koi_period >= 200


def resumen(m, nombre):
    s = k[m]
    c = s[(s.koi_disposition == "CANDIDATE") & s.koi_score.notna()]   # koi_score solo existe para KOIs del DR25
    return {
        "regimen": nombre, "n": int(len(s)),
        "CONFIRMED": int((s.koi_disposition == "CONFIRMED").sum()),
        "CANDIDATE": int((s.koi_disposition == "CANDIDATE").sum()),
        "FALSE POSITIVE": int((s.koi_disposition == "FALSE POSITIVE").sum()),
        "pct_confirmados": round(100 * (s.koi_disposition == "CONFIRMED").mean(), 1),
        "mediana_MES": round(float(s.koi_max_mult_ev.median()), 1),
        "mediana_n_transitos": round(float(s.koi_num_transits.median()), 1),
        "candidatos_con_score": int(len(c)),
        "candidatos_score_mediana": round(float(c.koi_score.median()), 2) if len(c) else None,
        "candidatos_score<0.5_pct": round(100 * (c.koi_score < 0.5).mean(), 1) if len(c) else None,
    }


R["regimenes"] = [resumen(corto, "periodo < 50 d"), resumen(largo, "periodo ≥ 200 d"),
                  resumen(hz, "zona habitable aprox. (0.25–2.2 S⊕, Rp ≤ 2.5 R⊕)"),
                  resumen(hz & largo, "zona habitable aprox. y periodo ≥ 200 d")]

# ---------- 3. Experimento de fuga: modelo tabular ----------
lab = k[k.koi_disposition.isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
y = (lab.koi_disposition == "CONFIRMED").astype(int).values
fisicas = ["koi_period", "koi_duration", "koi_depth", "koi_impact", "koi_prad", "koi_teq", "koi_insol",
           "koi_model_snr", "koi_max_mult_ev", "koi_num_transits", "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag"]
fuga = ["koi_score", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]
centroide = ["koi_dikco_msky", "koi_dicco_msky"]
cv = StratifiedKFold(5, shuffle=True, random_state=0)
hz_lab = (hz.loc[lab.index]).values


def evaluar(cols, nombre):
    X = lab[cols].values
    p = cross_val_predict(HistGradientBoostingClassifier(random_state=0), X, y, cv=cv, method="predict_proba")[:, 1]
    yhat = (p >= 0.5).astype(int)
    r = {"modelo": nombre, "n": int(len(y)), "accuracy": round(accuracy_score(y, yhat), 3),
         "AUC": round(roc_auc_score(y, p), 3), "recall_planeta": round(recall_score(y, yhat), 3),
         "precision_planeta": round(precision_score(y, yhat), 3)}
    yh, ph, yy = yhat[hz_lab], p[hz_lab], y[hz_lab]
    r["zona_habitable"] = {"n": int(hz_lab.sum()), "planetas": int(yy.sum()), "accuracy": round(accuracy_score(yy, yh), 3),
                           "AUC": round(roc_auc_score(yy, ph), 3) if 0 < yy.sum() < len(yy) else None,
                           "recall_planeta": round(recall_score(yy, yh), 3) if yy.sum() else None}
    return r


R["experimento_tabular"] = [
    evaluar(fisicas + fuga, "A · con columnas que filtran la respuesta (koi_score + banderas)"),
    evaluar(fisicas, "B · solo parámetros físicos y de tránsito"),
    evaluar(fisicas + centroide, "C · B + desplazamiento del centroide (información de píxeles)"),
]

# ---------- 4. Etiquetas de habitabilidad: cuántos ejemplos positivos ----------
R["koi_en_zona_habitable_aprox"] = {
    "total": int(hz.sum()), "CONFIRMED": int((hz & (k.koi_disposition == "CONFIRMED")).sum()),
    "CANDIDATE": int((hz & (k.koi_disposition == "CANDIDATE")).sum())}

json.dump(R, open(os.path.join(out, "evidencia_koi.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
