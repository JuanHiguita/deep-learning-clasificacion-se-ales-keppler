"""Demo: CNN multirrama (imagen de diferencia 2D + vistas 1D) sobre datos reales de Kepler.

- Validación cruzada de 5 particiones AGRUPADA POR ESTRELLA: cada predicción viene de un modelo
  que nunca vio esa estrella.
- Ablación: modelo completo vs solo imagen vs solo curva vs escalar de centroide (tabla KOI).
- Demo: 5 planetas, 5 falsos positivos y 5 negativos elegidos al azar (semilla fija) entre los
  casos fuera de muestra, con explicabilidad por oclusión.

Uso: KERAS_BACKEND=jax python -I demo_cnn.py <dataset.npz> <carpeta_salida>
"""

import json
import os
import sys
import warnings

os.environ.setdefault("KERAS_BACKEND", "jax")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import keras
from keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
d = np.load(DS)
IMG, SEQ, Y = d["img"], d["seq"], d["y"]
meta = pd.read_csv(DS.replace(".npz", "_meta.csv"))
NOMBRES = ["Planeta", "Falso positivo", "Sin señal"]


def modelo(usar_img=True, usar_seq=True, semilla=0):
    keras.utils.set_random_seed(semilla)
    ent, ramas = [], []
    a = keras.Input((11, 11, 4), name="img")
    b = keras.Input((61, 2), name="seq")
    if usar_img:  # CNN 2D SIN pooling: las imágenes miden pocos píxeles
        x = layers.Conv2D(16, 3, padding="same", activation="relu")(a)
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(0.4)(x)
        ramas.append(layers.Dense(32, activation="relu")(x))
    if usar_seq:  # CNN 1D sobre vista local y vista en fase 0.5
        z = layers.Conv1D(16, 5, padding="same", activation="relu")(b)
        z = layers.MaxPooling1D(2)(z)
        z = layers.Conv1D(32, 5, padding="same", activation="relu")(z)
        z = layers.GlobalMaxPooling1D()(z)
        ramas.append(layers.Dense(32, activation="relu")(z))
    h = ramas[0] if len(ramas) == 1 else layers.Concatenate()(ramas)
    h = layers.Dropout(0.3)(layers.Dense(32, activation="relu")(h))
    sal = layers.Dense(3, activation="softmax")(h)
    m = keras.Model([a, b], sal)
    m.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


def aumentar(img, seq, y, rng):
    """Rotaciones de 90° (la estrella cambia de orientación cada trimestre) + ruido leve."""
    k = rng.integers(0, 4, len(img))
    im = np.stack([np.rot90(i, kk, axes=(0, 1)) for i, kk in zip(img, k)])
    im[..., 0] += rng.normal(0, 0.05, im[..., 0].shape)
    sq = seq + rng.normal(0, 0.05, seq.shape)
    return im.astype(np.float32), sq.astype(np.float32), y


def entrenar(tr, va, usar_img, usar_seq, semilla):
    rng = np.random.default_rng(semilla)
    m = modelo(usar_img, usar_seq, semilla)
    pesos = {c: len(tr) / (3 * max((Y[tr] == c).sum(), 1)) for c in range(3)}
    mejor, paciencia, w_mejor = 1e9, 0, None
    for ep in range(80):
        im, sq, yy = aumentar(IMG[tr], SEQ[tr], Y[tr], rng)
        m.fit([im, sq], yy, epochs=1, batch_size=32, verbose=0, class_weight=pesos)
        perd = m.evaluate([IMG[va], SEQ[va]], Y[va], verbose=0)[0]
        if perd < mejor - 1e-4:
            mejor, paciencia, w_mejor = perd, 0, m.get_weights()
        else:
            paciencia += 1
            if paciencia >= 12:
                break
    m.set_weights(w_mejor)
    return m


grupos = meta.kepid.values
estrat = np.where(meta.tipo == "nulo", 2, Y)
cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
oof = {v: np.zeros((len(Y), 3)) for v in ["completo", "solo_imagen", "solo_curva"]}
modelos, fold_de = {}, np.zeros(len(Y), int)
for f, (tr_all, te) in enumerate(cv.split(IMG, estrat, grupos)):
    # validación interna agrupada por estrella para el early stopping
    estrellas = np.unique(grupos[tr_all])
    rng = np.random.default_rng(f)
    va_est = set(rng.choice(estrellas, size=max(1, len(estrellas) // 6), replace=False))
    va = np.array([i for i in tr_all if grupos[i] in va_est])
    tr = np.array([i for i in tr_all if grupos[i] not in va_est])
    for v, (ui, us) in {"completo": (1, 1), "solo_imagen": (1, 0), "solo_curva": (0, 1)}.items():
        m = entrenar(tr, va, ui, us, semilla=f)
        oof[v][te] = m.predict([IMG[te], SEQ[te]], verbose=0)
        if v == "completo":
            modelos[f] = m
    fold_de[te] = f
    print("partición", f, "lista", flush=True)

# ---------- métricas ----------
R = {
    "n_ejemplos": int(len(Y)),
    "por_clase": {NOMBRES[c]: int((Y == c).sum()) for c in range(3)},
    "n_estrellas": int(len(np.unique(grupos))),
}
for v, p in oof.items():
    pred = p.argmax(1)
    real = Y < 2
    pp = p[real, 0] / (p[real, 0] + p[real, 1] + 1e-9)
    R[v] = {
        "accuracy_3_clases": round(accuracy_score(Y, pred), 3),
        "matriz_confusion": confusion_matrix(Y, pred, labels=[0, 1, 2]).tolist(),
        "AUC_planeta_vs_FP": round(roc_auc_score((Y[real] == 0).astype(int), pp), 3),
    }
# línea base: escalar de centroide de la tabla KOI (misma partición por estrella)
real = np.flatnonzero(Y < 2)
Xc = meta.loc[real, ["dikco", "dicco"]].fillna(meta[["dikco", "dicco"]].median()).values
pc = np.zeros(len(real))
for f in range(5):
    tr, te = real[fold_de[real] != f], real[fold_de[real] == f]
    lr = LogisticRegression().fit(
        meta.loc[tr, ["dikco", "dicco"]].fillna(0).values, (Y[tr] == 0).astype(int)
    )
    pc[np.isin(real, te)] = lr.predict_proba(meta.loc[te, ["dikco", "dicco"]].fillna(0).values)[
        :, 1
    ]
R["escalar_centroide"] = {
    "AUC_planeta_vs_FP": round(roc_auc_score((Y[real] == 0).astype(int), pc), 3)
}
# desglose de errores del modelo completo por tipo de falso positivo
fp = np.flatnonzero(Y == 1)
pred = oof["completo"].argmax(1)
for nombre, col in [
    ("centroide (co)", "fp_co"),
    ("eclipse secundario (ss)", "fp_ss"),
    ("no parece tránsito (nt)", "fp_nt"),
]:
    s = fp[meta.loc[fp, col].values == 1]
    R.setdefault("acierto_FP_por_tipo", {})[nombre] = f"{int((pred[s] == 1).sum())}/{len(s)}"
json.dump(R, open(os.path.join(OUT, "metricas.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))

# ---------- selección de la demo (al azar, sin elegir los aciertos) ----------
rng = np.random.default_rng(7)
pl = rng.choice(np.flatnonzero(Y == 0), 5, replace=False)
fps = []
for col, n in [("fp_co", 2), ("fp_ss", 2), ("fp_nt", 1)]:
    cand = [i for i in np.flatnonzero((Y == 1) & (meta[col].values == 1)) if i not in fps]
    fps += list(rng.choice(cand, n, replace=False))
usadas = set(grupos[np.r_[pl, fps]])
nul = rng.choice([i for i in np.flatnonzero(Y == 2) if grupos[i] not in usadas], 5, replace=False)
demo = {
    "Planetas (CONFIRMED)": list(pl),
    "Falsos positivos": fps,
    "Negativos (sin señal)": list(nul),
}


def oclusion(i):
    """Caída de la probabilidad de la clase predicha al tapar cada píxel / tramo de la curva."""
    m = modelos[fold_de[i]]
    base = m.predict([IMG[i : i + 1], SEQ[i : i + 1]], verbose=0)[0]
    c = base.argmax()
    lote = []
    for y in range(11):
        for x in range(11):
            im = IMG[i].copy()
            im[y, x, 0] = 0.0
            lote.append(im)
    p = m.predict([np.array(lote), np.repeat(SEQ[i : i + 1], len(lote), 0)], verbose=0)[:, c]
    mapa = (base[c] - p).reshape(11, 11)
    lote_s = []
    for s in range(0, 61, 4):
        sq = SEQ[i].copy()
        sq[s : s + 6, :] = 0.0
        lote_s.append(sq)
    ps = m.predict([np.repeat(IMG[i : i + 1], len(lote_s), 0), np.array(lote_s)], verbose=0)[:, c]
    return base, c, mapa, base[c] - ps


resumen = []
for titulo, idx in demo.items():
    fig, ejes = plt.subplots(
        4, 5, figsize=(17, 12.5), gridspec_kw={"height_ratios": [1, 1, 0.8, 0.55]}
    )
    for j, i in enumerate(idx):
        base, c, mapa, occ_s = oclusion(i)
        mt = meta.iloc[i]
        tipo_fp = ",".join(
            n for n, col in [("co", "fp_co"), ("ss", "fp_ss"), ("nt", "fp_nt")] if mt[col] == 1
        )
        etiqueta = NOMBRES[Y[i]] + (f" ({tipo_fp})" if Y[i] == 1 else "")
        a = ejes[0, j]
        a.imshow(IMG[i, ..., 0] * 10, origin="lower", cmap="RdBu_r", vmin=-15, vmax=15)
        a.plot(5, 5, "+", color="black", ms=14, mew=2)
        a.set_title(f"{mt.koi} · KIC {mt.kepid}\nreal: {etiqueta}", fontsize=9)
        a.set_xticks([])
        a.set_yticks([])
        if j == 0:
            a.set_ylabel("Imagen de diferencia\n(SNR por píxel)", fontsize=9)
        a = ejes[1, j]
        a.imshow(mapa, origin="lower", cmap="magma", vmin=0, vmax=max(mapa.max(), 0.05))
        a.plot(5, 5, "+", color="white", ms=14, mew=2)
        a.set_xticks([])
        a.set_yticks([])
        if j == 0:
            a.set_ylabel("Oclusión: píxeles que\nusó el modelo", fontsize=9)
        a = ejes[2, j]
        xs = np.linspace(-2, 2, 61)
        a.plot(xs, SEQ[i, :, 0] * 10, color="black", lw=1, label="vista local")
        a.plot(xs, SEQ[i, :, 1] * 10, color="tab:orange", lw=1, label="fase 0.5")
        for k, s in enumerate(range(0, 61, 4)):
            if occ_s[k] > 0.05:
                a.axvspan(xs[s], xs[min(s + 5, 60)], color="tab:purple", alpha=min(0.5, occ_s[k]))
        a.set_ylim(-25, 8)
        a.set_xlabel("tiempo / duración", fontsize=8)
        if j == 0:
            a.set_ylabel("Curva (SNR)\nmorado = tramo usado", fontsize=9)
            a.legend(fontsize=7, loc="lower left")
        a = ejes[3, j]
        colores = ["tab:blue", "tab:red", "tab:gray"]
        a.barh(NOMBRES, base, color=colores)
        a.set_xlim(0, 1)
        a.invert_yaxis()
        ok = "✓" if c == Y[i] else "✗"
        a.set_title(
            f"{ok} predicción: {NOMBRES[c]} ({base[c]:.2f})",
            fontsize=9,
            color="darkgreen" if c == Y[i] else "darkred",
        )
        a.tick_params(labelsize=8)
        resumen.append(
            {
                "grupo": titulo,
                "koi": mt.koi,
                "kepid": int(mt.kepid),
                "real": etiqueta,
                "predicho": NOMBRES[c],
                "p_planeta": round(float(base[0]), 2),
                "p_fp": round(float(base[1]), 2),
                "p_sin_senal": round(float(base[2]), 2),
                "snr_max_imagen": round(float(IMG[i, ..., 0].max() * 10), 1),
                "n_transitos": int(mt.n_transitos),
                "dikco_arcsec": None if pd.isna(mt.dikco) else round(float(mt.dikco), 2),
            }
        )
    fig.suptitle(
        f"Demo con datos REALES de Kepler · {titulo} · predicciones fuera de muestra (CV agrupada por estrella)",
        fontsize=12,
    )
    fig.tight_layout()
    nombre = {
        "Planetas (CONFIRMED)": "planetas",
        "Falsos positivos": "falsos_positivos",
        "Negativos (sin señal)": "negativos",
    }[titulo]
    fig.savefig(os.path.join(OUT, f"demo_{nombre}.png"), dpi=110)
    plt.close(fig)
pd.DataFrame(resumen).to_csv(os.path.join(OUT, "demo_15_casos.csv"), index=False)
print(pd.DataFrame(resumen).to_string())

# ---------- figura resumen ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
cm = np.array(R["completo"]["matriz_confusion"])
a1.imshow(cm, cmap="Blues")
for (r, c), v in np.ndenumerate(cm):
    a1.text(
        c,
        r,
        v,
        ha="center",
        va="center",
        fontsize=13,
        color="white" if v > cm.max() / 2 else "black",
    )
a1.set_xticks(range(3), NOMBRES)
a1.set_yticks(range(3), NOMBRES)
a1.set_xlabel("predicho")
a1.set_ylabel("real")
a1.set_title(
    f"Modelo completo · accuracy {R['completo']['accuracy_3_clases']:.2f} ({len(Y)} ejemplos, fuera de muestra)"
)
vs = ["escalar_centroide", "solo_curva", "solo_imagen", "completo"]
et = [
    "Escalar de centroide\n(tabla KOI)",
    "Solo curva\n(CNN 1D)",
    "Solo imagen\n(CNN 2D)",
    "Completo\n(2D + 1D)",
]
vals = [R[v]["AUC_planeta_vs_FP"] for v in vs]
a2.bar(et, vals, color=["0.6", "0.6", "tab:blue", "tab:blue"])
for k, v in enumerate(vals):
    a2.text(k, v + 0.01, f"{v:.3f}", ha="center")
a2.set_ylim(0.5, 1.02)
a2.set_title("Ablación: AUC planeta vs falso positivo")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "demo_resumen.png"), dpi=110)
