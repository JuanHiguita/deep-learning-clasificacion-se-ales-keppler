"""Genera los cinco cuadernos por fase de la iteración 2 (spec 014) y la muestra de datos.

    iteracion_2/notebooks/01_datos_y_preprocesamiento.ipynb
    iteracion_2/notebooks/02_arquitecturas.ipynb
    iteracion_2/notebooks/03_busqueda_de_hiperparametros.ipynb
    iteracion_2/notebooks/04_modelos_finales_y_fusion.ipynb
    iteracion_2/notebooks/05_analisis_de_resultados.ipynb

Los cuadernos no entrenan: leen las evidencias guardadas en `iteracion_2/` (rutas relativas) y
solo necesitan numpy, pandas y matplotlib. Así se ejecutan igual en cualquier entorno.

Uso:
    python scripts/construir_notebooks_it2.py [--muestra]
    jupyter nbconvert --to notebook --execute --inplace iteracion_2/notebooks/0*.ipynb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat

CARPETA = Path("iteracion_2/notebooks")
SEMILLA_MUESTRA = 20261122

PREPARACION = """
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

import apoyo  # iteracion_2/notebooks/apoyo.py: rutas, lectura de evidencias y figuras

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 80)
apoyo.estilo()
print("Raíz del repositorio:", apoyo.RAIZ.name, "| evidencias en", apoyo.EVIDENCIAS.relative_to(apoyo.RAIZ))
"""


def crear_muestra() -> None:
    """60 ejemplos al azar (20 por clase, semilla fija) para que el cuaderno 01 corra sin el dataset."""
    import numpy as np
    import pandas as pd

    datos = Path("iteracion_2/datos")
    with np.load(datos / "dataset_temporal.npz", allow_pickle=False) as d:
        ds = {k: d[k] for k in d.files}
    ej = pd.read_csv(datos / "ejemplos.csv").set_index("id_ejemplo")
    ids = [str(i) for i in ds["id_ejemplo"]]
    y = ej.loc[ids, "clase"].to_numpy()
    usable = (
        (ds["disponible_vistas"] == 1)
        & (ds["disponible_transitos"] == 1)
        & ~ej.loc[ids, "prerregistrado"].to_numpy().astype(bool)
    )
    rng = np.random.default_rng(SEMILLA_MUESTRA)
    elegidos = np.sort(
        np.concatenate(
            [rng.choice(np.flatnonzero(usable & (y == c)), 20, replace=False) for c in range(3)]
        )
    )
    np.savez_compressed(
        datos / "muestra_dataset.npz",
        **{k: (v if k == "nombres_escalares" else v[elegidos]) for k, v in ds.items()},
    )


class Cuaderno:
    def __init__(self, archivo: str, titulo: str, pregunta: str):
        self.archivo, self.celdas = archivo, []
        self.texto(
            f"# {titulo}\n\n"
            "**Detección de exoplanetas en Kepler con una red multirrama — iteración 2** "
            "(más datos, entrenamiento en GPU)\n\n"
            "Curso de Deep Learning · Especialización en Inteligencia Artificial · "
            "Universidad Nacional de Colombia\n\n"
            "**Autores:** Thomas Monnier Granda, Esteban Felipe Genes Noya, "
            "Juan Sebastian Higuita Riascos, Alexandra Arias Mazo y Eliel Segura Perez\n\n"
            f"> **Pregunta de este cuaderno:** {pregunta}\n\n"
            "Este cuaderno no entrena nada: lee las evidencias que dejaron los pasos 1 a 5 "
            "(`iteracion_2/*.bat`) y las explica. Cada cifra sale de un archivo del repositorio."
        )
        self.codigo(PREPARACION)

    def texto(self, markdown: str) -> None:
        self.celdas.append(nbformat.v4.new_markdown_cell(markdown.strip("\n")))

    def codigo(self, fuente: str) -> None:
        self.celdas.append(nbformat.v4.new_code_cell(fuente.strip("\n")))

    def escribir(self) -> None:
        cuaderno = nbformat.v4.new_notebook(
            cells=self.celdas,
            metadata={
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"},
            },
        )
        CARPETA.mkdir(parents=True, exist_ok=True)
        nbformat.write(cuaderno, CARPETA / self.archivo)
        print("escrito", CARPETA / self.archivo)


# ---------------------------------------------------------------------------------------------
def c01() -> Cuaderno:
    c = Cuaderno(
        "01_datos_y_preprocesamiento.ipynb",
        "Fase 1 · Datos y preprocesamiento",
        "¿con qué datos se entrena, cómo se preparan y cómo se evita que la red haga trampa?",
    )
    c.texto("""
## 1. De 262 a 4.547 señales

La iteración 1 usó 262 señales. Con tan pocas, una red no puede ser grande ni mostrar que aprende de
las curvas. Aquí se toman **todas** las señales de Kepler (catálogo DR25) confirmadas como planeta o
descartadas como falso positivo, con periodo entre 1 y 30 días (así cada una tiene muchos tránsitos).

Cada señal produce **dos ejemplos**: la curva con su tránsito (planeta o falso positivo) y la misma
curva con el tránsito borrado, que es la clase *sin señal*. Por eso hay el doble de ejemplos que de señales.
""")
    c.codigo("""
base = apoyo.leer_json(apoyo.DATOS / "manifiesto_base.json")["resumen"]
ej = apoyo.ejemplos()
print(f"{base['n_kois']} señales de {base['n_estrellas']} estrellas → {base['n_ejemplos']} ejemplos")
por_clase = ej.groupby("clase").size().rename(index=dict(enumerate(apoyo.CLASES)))
display(por_clase.to_frame("ejemplos"))
""")
    c.texto("""
## 2. Particiones por estrella: la primera defensa contra la fuga

Una estrella puede tener varias señales. Si una cae en entrenamiento y otra en prueba, la red podría
reconocer la estrella en lugar de la señal. Por eso las cinco particiones se reparten **por estrella**
(con un resumen criptográfico de su identificador, que no depende del azar). La celda lo comprueba.
""")
    c.codigo("""
por_estrella = ej.groupby("kepid")["particion"].nunique()
print("Estrellas repartidas en más de una partición:", int((por_estrella > 1).sum()), "(debe ser 0)")
tabla = ej.pivot_table(index="particion", columns="clase", values="kepid", aggfunc="size")
tabla.columns = apoyo.CLASES
display(tabla)
fig, eje = plt.subplots(figsize=(6, 2.6))
tabla.plot.bar(ax=eje, color=apoyo.COLORES, width=0.8)
eje.set_ylabel("ejemplos"); eje.set_title("Las cinco particiones quedan balanceadas", loc="left")
apoyo.guardar(fig, "01_particiones")
""")
    c.texto("""
## 3. Señales prerregistradas: lo que no se toca

Antes de entrenar se apartaron las señales cuya disposición difiere entre el catálogo DR25 y la tabla
acumulada del archivo de exoplanetas (su etiqueta cambió). No entran al
entrenamiento, ni a la búsqueda, ni a ninguna cifra de estos cuadernos: se evalúan **una sola vez**, al final.
""")
    c.codigo("""
print("Señales prerregistradas apartadas:", base["n_prerregistrados_kois"])
print("Ejemplos marcados como prerregistrados:", int(ej["prerregistrado"].sum()))
""")
    c.texto("""
## 4. Descarga y construcción de las entradas

Para cada estrella se descargaron sus curvas de luz (flujo PDCSAP, trimestres 1 a 17). El
preprocesamiento hace, en orden: quitar cadencias malas, quitar la tendencia lenta de la estrella
**sin** tocar los tránsitos, y plegar la curva con el periodo de la señal. De ahí salen tres entradas:

| Entrada | Forma | Rama que la usa |
|---|---|---|
| Vista global y vista local | 2001 y 201 puntos | Rama 1 (CNN 1D) |
| Tránsitos individuales | hasta 32 tránsitos de 61 puntos | Rama 2 (codificador por tránsito) |
| Escalares medidos en la curva | 7 números y 7 avisos de faltante | Rama 4 |

No todas las estrellas tienen datos suficientes; el registro lo deja contado.
""")
    c.codigo("""
registro = pd.read_csv(apoyo.DATOS / "dataset_temporal_registro.csv")
resumen = apoyo.leer_json(apoyo.DATOS / "dataset_temporal.manifiesto.json")["resumen"]
n = resumen["n_ejemplos"]
print(f"Ejemplos construidos: {n}")
print(f"  con vistas utilizables:    {resumen['disponible_vistas']} ({resumen['disponible_vistas']/n:.1%})")
print(f"  con tránsitos utilizables: {resumen['disponible_transitos']} ({resumen['disponible_transitos']/n:.1%})")
fig, ejes = plt.subplots(1, 2, figsize=(9, 2.6))
registro["trimestres"].plot.hist(bins=range(0, 19), ax=ejes[0], color="#2166ac")
ejes[0].set_title("Trimestres disponibles por ejemplo", loc="left")
registro["transitos_validos"].clip(upper=120).plot.hist(bins=40, ax=ejes[1], color="#2166ac")
ejes[1].set_title("Tránsitos válidos por ejemplo (recortado en 120)", loc="left")
fig.tight_layout(); apoyo.guardar(fig, "01_cobertura")
""")
    c.texto("""
## 5. Cómo ve la red cada clase

Tres ejemplos tomados **al azar con semilla fija** (no elegidos a mano), uno por clase. Si el dataset
completo no está en disco, se usa la muestra versionada de 60 ejemplos.
""")
    c.codigo("""
datos, origen = apoyo.dataset()
print("Usando:", origen, "·", len(datos["id_ejemplo"]), "ejemplos")
clase_de = ej.set_index("id_ejemplo")["clase"]
y = clase_de.loc[[str(i) for i in datos["id_ejemplo"]]].to_numpy()
rng = np.random.default_rng(protocolo_semilla := 0)
for c in range(3):
    i = int(rng.choice(np.flatnonzero((y == c) & (datos["disponible_vistas"] == 1) & (datos["disponible_transitos"] == 1))))
    fig = apoyo.ver_ejemplo(datos, i, f"{apoyo.CLASES[c].upper()} · {datos['id_ejemplo'][i]}")
    apoyo.guardar(fig, f"01_ejemplo_{c}")
    plt.show()
""")
    c.texto("""
## 6. Escalares: qué entra y qué está prohibido

La rama 4 recibe siete números **medidos en la curva o propios de la geometría**, nunca la opinión
del catálogo. Las columnas que contienen la decisión de los revisores (`koi_score`, las banderas
`koi_fpflag_*`, `koi_pdisposition`) están prohibidas como entrada. La celda verifica que ninguna
se coló. Un dato que falta **no se rellena con cero a escondidas**: viaja con su aviso de faltante.
La segunda tabla muestra en qué clase falta el centroide.
""")
    c.codigo("""
p = apoyo.protocolo()
nombres = [str(n) for n in datos["nombres_escalares"]]
prohibidas = set(p["columnas_prohibidas_como_entrada"])
coladas = [n for n in nombres if n in prohibidas or n.startswith("koi_fpflag")]
print("Escalares de entrada:", nombres)
print("Columnas prohibidas :", sorted(prohibidas))
print("¿Alguna prohibida entre las entradas?:", coladas or "ninguna")
faltan = pd.Series(datos["faltantes"].mean(axis=0), index=nombres).rename("fracción faltante")
display(faltan.map("{:.1%}".format).to_frame())
centroide = pd.Series(datos["faltantes"][:, nombres.index("koi_dikco_msky")], name="centroide faltante")
display(centroide.groupby(pd.Series(y).map(dict(enumerate(apoyo.CLASES))).rename("clase")).mean().map("{:.1%}".format).to_frame())
""")
    c.texto("""
**Un atajo que hay que conocer.** Los ejemplos *sin señal* nunca tienen centroide (no hay tránsito del
cual medirlo), así que su aviso de faltante los delata. Por eso separar *sin señal* del resto es fácil
para la rama 4 y los aciertos de tres clases se leen con cautela. La métrica principal del proyecto,
el AUC de **planeta frente a falso positivo**, no usa los ejemplos *sin señal* y no se beneficia de ese
atajo. Entre planetas y falsos positivos el centroide también falta con distinta frecuencia: es una
señal leve que viene del catálogo y que el modelo tabular con centroide ve igualmente.
""")
    c.texto("""
## Qué queda de esta fase

- 17 veces más señales que en la iteración 1, con las mismas reglas.
- Particiones por estrella verificadas, prerregistrados apartados, columnas con fuga fuera.
- Tres representaciones de la misma curva, una por rama. El siguiente cuaderno explica qué red lee cada una.
""")
    return c


# ---------------------------------------------------------------------------------------------
def c02() -> Cuaderno:
    c = Cuaderno(
        "02_arquitecturas.ipynb",
        "Fase 2 · Arquitecturas",
        "¿qué red lee cada entrada, por qué tiene esa forma y cómo se juntan en una sola respuesta?",
    )
    c.texto("""
## 1. Por qué varias ramas

Una señal de tránsito se puede mirar de varias maneras, y cada una delata un tipo distinto de impostor:

| Rama | Qué mira | Qué puede delatar | Red |
|---|---|---|---|
| 1 · vistas | La forma media del tránsito y la órbita completa | Eclipses secundarios, tránsitos en V | CNN 1D de dos torres (tipo AstroNet, Shallue y Vanderburg 2018) |
| 2 · tránsitos | Cada tránsito por separado | Tránsitos que cambian de uno a otro, artefactos aislados | Codificador compartido por tránsito + agregador |
| 3 · imagen | De qué píxel viene la señal | Estrellas vecinas contaminantes | CNN 2D (curso de Visión; **no** se usa aquí) |
| 4 · escalares | Periodo, duración, profundidad, centroide | Combinaciones físicamente raras | Red densa pequeña |

Esta iteración entrena las ramas 1, 2 y 4. La rama 3 pertenece al curso de Visión por Computador y
queda **declarada como ausente** (bandera en cero), de modo que pueda sumarse después sin cambiar nada.

## 2. La rama 2 y sus tres agregadores

La rama 2 aplica la **misma** red pequeña a cada tránsito y luego tiene que resumir hasta 32 vectores
en uno. Se comparan tres formas de hacerlo:

- **Promedio:** todos los tránsitos pesan igual. No depende del orden.
- **GRU bidireccional:** lee los tránsitos en orden temporal; puede notar cambios a lo largo del tiempo.
- **Atención:** aprende cuánto pesa cada tránsito; puede ignorar los malos.

A los tres se les añade un vector de **consistencia** (por ejemplo, diferencia entre tránsitos pares e
impares: en un sistema binario eclipsante los pares y los impares suelen tener distinta profundidad).
""")
    c.codigo("""
filas = []
for red, nombre in apoyo.REDES.items():
    cfg = apoyo.configuraciones(red)
    filas.append({"red": nombre, "parámetros (protocolo)": int(cfg.loc[0, "parametros"]),
                  "parámetros (mayor probada)": int(cfg["parametros"].max()),
                  "vector de salida": 16 if red == "m4_escalares" else 32})
tamanos = pd.DataFrame(filas).set_index("red")
display(tamanos)
""")
    c.texto("""
Las redes del protocolo son pequeñas a propósito (se diseñaron para 262 señales). Con 4.547 señales
la búsqueda del cuaderno 3 prueba redes decenas de veces mayores para ver si los datos nuevos las aprovechan.

## 3. Resumen de capas (opcional)

Si Keras está instalado, la celda construye las redes y muestra sus capas. Si no, se salta: el resto
del cuaderno no lo necesita.
""")
    c.codigo("""
try:
    import os, sys
    os.environ.setdefault("KERAS_BACKEND", "torch")
    sys.path.insert(0, str(apoyo.RAIZ / "src"))
    from temporal import modelos
    for red in (modelos.m4_escalares(0), modelos.m1_astronet(0), modelos.m2(0, "atencion", plegado=True)):
        red.summary(line_length=90)
except Exception as error:  # sin Keras o sin backend: no es un fallo del cuaderno
    print("Resumen de capas omitido:", type(error).__name__, "-", str(error)[:100])
""")
    c.texto("""
## 4. Entrenar en GPU

El requisito de esta iteración era entrenar en la GPU del equipo. Se usó Keras sobre PyTorch con CUDA.
La celda muestra el entorno que quedó registrado y una medición hecha antes de la búsqueda.
""")
    c.codigo("""
entorno = apoyo.leer_json(apoyo.EVIDENCIAS / "entorno.json")
display(pd.Series(entorno).to_frame("entorno de entrenamiento"))
humo = apoyo.leer_json(apoyo.EVIDENCIAS / "prueba_de_humo.json")
plegado = pd.DataFrame(humo["plegado"]).T
plegado["aceleración"] = plegado["time_distributed"] / plegado["plegado"]
display(plegado.rename(columns={"time_distributed": "s/época, tránsito a tránsito", "plegado": "s/época, todos a la vez"}).round(2))
""")
    c.texto("""
**Qué es esa aceleración.** La forma habitual de aplicar una red a cada tránsito los recorre uno por
uno. Aquí los 32 tránsitos se pliegan dentro del lote y se procesan de una vez. Es la misma función
con los mismos pesos (hay una prueba automática que lo verifica); solo cambia la velocidad.

## 5. Cómo se juntan: la cabeza de fusión

Cada rama entrega un vector (32, 32 y 16 números). La cabeza de fusión los concatena con una bandera
por rama que dice si esa rama tenía dato, y pasa todo por `Densa(64) → Dropout(0,3) → Densa(3)`.
Las ramas quedan **congeladas**: la cabeza no las modifica. Hay un codificador por partición, de modo
que ningún ejemplo de prueba fue visto por el codificador que lo describe. El cuaderno 4 muestra el resultado.
""")
    return c


# ---------------------------------------------------------------------------------------------
def c03() -> Cuaderno:
    c = Cuaderno(
        "03_busqueda_de_hiperparametros.ipynb",
        "Fase 3 · Búsqueda y selección de hiperparámetros",
        "¿las redes del protocolo eran las mejores posibles, y cómo se elige sin hacer trampa?",
    )
    c.texto("""
## 1. Las reglas de la búsqueda

1. **El espacio se declara antes de entrenar** (`src/temporal/busqueda_it2.py`) y no se cambia al ver resultados.
2. **12 configuraciones por red**, las mismas para todas: la del protocolo (configuración 0) y 11 sorteadas con semilla fija.
3. **Selección anidada:** cada partición elige su configuración mirando solo su validación interna. La prueba nunca decide.
4. **Criterio:** la menor NLL de validación. La NLL (pérdida logarítmica) premia a la vez ordenar bien y dar probabilidades honestas.

Se buscó sobre el tamaño de la red (ancho, bloques extra), la activación, la normalización, la pérdida
(entropía cruzada o con suavizado de etiquetas), la tasa de aprendizaje, la regularización y el tamaño de lote.
""")
    c.codigo("""
red = "m2_promedio"   # cambia por: m4_escalares, m1_astronet, m2_atencion, m2_gru
cfg = apoyo.configuraciones(red)
columnas = [c for c in cfg.columns if not c.startswith("nll_val_k")]
display(Markdown(f"**{apoyo.REDES[red]}** — las 12 configuraciones (la 00 es el protocolo)"))
display(cfg[columnas].set_index("configuracion"))
""")
    c.texto("""
## 2. ¿Sirvió buscar?

Para cada red se compara, sobre la prueba, el protocolo fijo con el sistema "con búsqueda" (cada
partición con su configuración elegida). La diferencia es **pareada**: los mismos ejemplos, remuestreados
por estrella 2.000 veces. Una diferencia es *distinguible* si su intervalo del 95 % no contiene el cero.
""")
    c.codigo("""
display(apoyo.tabla_busqueda())
fig, eje = plt.subplots(figsize=(7, 3))
apoyo.bosque({r: apoyo.resumen_busqueda(r)["auc_prueba"]["diferencia"] for r in apoyo.REDES},
             "Ganancia de AUC por buscar (con búsqueda − protocolo fijo)", referencia=0.0, eje=eje)
apoyo.guardar(fig, "03_ganancia_busqueda")
""")
    c.texto("""
**Lectura.** Tres de las cinco redes mejoran de forma distinguible, todas por milésimas. La rama 1 y
la GRU se quedan casi siempre con la red del protocolo. La búsqueda ayuda, pero poco.

## 3. ¿Las redes grandes ganan?

Cada punto es una configuración: su tamaño frente a su NLL de validación (menor es mejor).
""")
    c.codigo("""
fig, ejes = plt.subplots(1, 5, figsize=(15, 2.8), sharey=False)
for eje, (red, nombre) in zip(ejes, apoyo.REDES.items()):
    cfg = apoyo.configuraciones(red)
    eje.scatter(cfg["parametros"], cfg["nll_val_media"], color="#2166ac")
    eje.scatter(cfg.loc[0, "parametros"], cfg.loc[0, "nll_val_media"], color="#b2182b", label="protocolo", zorder=3)
    eje.set_xscale("log"); eje.set_title(nombre, loc="left", fontsize=9); eje.set_xlabel("parámetros")
ejes[0].set_ylabel("NLL de validación"); ejes[0].legend(fontsize=8)
fig.tight_layout(); apoyo.guardar(fig, "03_tamano_vs_nll")
""")
    c.texto("""
Solo en la rama 2 con promedio gana con claridad una red mayor (el doble de ancha y un bloque más).
En las demás, crecer no garantiza mejorar: las redes más grandes de la rama 1 y de la rama 4 quedan peor.

## 4. Análisis de sensibilidad (exploratorio): ¿y si se eligiera por AUC?

La NLL castiga al suavizado de etiquetas, que baja la confianza a propósito. Para saber si la conclusión
depende del criterio, se repitió la selección usando el AUC de validación. **Este análisis se añadió
después de ver resultados, así que no reemplaza al principal:** solo informa.
""")
    c.codigo("""
filas = []
for red, nombre in apoyo.REDES.items():
    s = apoyo.sensibilidad(red)
    filas.append({"red": nombre,
                  "AUC eligiendo por NLL": s["auc_prueba"]["seleccion_por_nll"]["estimacion"],
                  "AUC eligiendo por AUC": s["auc_prueba"]["seleccion_por_auc"]["estimacion"],
                  "diferencia de AUC": apoyo.con_intervalo(s["auc_prueba"]["auc_menos_nll"], signo=True),
                  "diferencia de NLL": apoyo.con_intervalo(s["nll_prueba"]["auc_menos_nll"], signo=True),
                  "particiones con la misma elección": f"{s['particiones_con_la_misma_eleccion']} de 5"})
display(pd.DataFrame(filas).set_index("red"))
""")
    c.texto("""
**Lectura.** No hay un ganador uniforme. Solo en la rama 1 elegir por AUC discrimina mejor, y lo paga
con peor calibración. En las otras cuatro redes, elegir por AUC no mejora el AUC y en varias empeora
la NLL. Se mantiene la NLL como criterio principal; el cuaderno 5 muestra que a nivel de sistema
tampoco conviene cambiarlo.

## 5. Una adaptación que sí se hizo, y por qué es legítima

En la rama 2 la profundidad se limitó a un bloque extra (no dos). Se decidió **antes de entrenar esa
rama**, por coste: una prueba con datos aleatorios midió horas por partición para la combinación mayor.
El motivo no dependió de ningún resultado; queda documentado en el código y en el historial.
""")
    return c


# ---------------------------------------------------------------------------------------------
def c04() -> Cuaderno:
    c = Cuaderno(
        "04_modelos_finales_y_fusion.ipynb",
        "Fase 4 · Modelos finales y fusión",
        "¿qué modelos quedaron elegidos, dónde están guardados y cómo construyen la respuesta conjunta?",
    )
    c.texto("""
## 1. Los modelos elegidos

Para cada rama y cada partición se eligió la red y la configuración con menor NLL en la validación de
esa partición. La rama 2 eligió además entre sus tres agregadores. Los codificadores elegidos y sus
vectores están en `iteracion_2/modelos_finales/<rama>/v1/`, cada uno con su huella SHA-256.
""")
    c.codigo("""
filas = []
for rama in ("vistas", "transitos", "escalares"):
    manifiesto = apoyo.leer_json(apoyo.MODELOS / rama / "v1" / "manifiesto.json")
    for k, e in manifiesto["particiones"].items():
        filas.append({"rama": rama, "partición": int(k), "red": apoyo.REDES[e["red"]],
                      "configuración": e["configuracion"], "parámetros": e["parametros"],
                      "NLL de validación": round(e["puntaje_validacion"], 4)})
elegidos = pd.DataFrame(filas)
display(elegidos.pivot(index="partición", columns="rama", values="red"))
display(elegidos.set_index(["rama", "partición"]))
""")
    c.texto("""
**Lectura.** En la rama 2 ningún agregador domina: cada partición prefiere uno distinto. Coincide con
lo que mostró la búsqueda: promedio, atención y GRU rinden parecido con estos datos.

## 2. Qué hay guardado y cómo volver a usarlo

Cada carpeta tiene, por partición: el codificador (`particion_k.keras`), los vectores de todos los
ejemplos (`vectores_particion_k.npz`) y, si aplica, la estandarización usada. Las cabezas de fusión están
en `evidencias/resultados/fusion/<configuración>/cabezas/`.
""")
    c.codigo("""
for rama in ("vistas", "transitos", "escalares"):
    archivos = sorted(p.name for p in (apoyo.MODELOS / rama / "v1").iterdir())
    print(f"{rama:10s}", len(archivos), "archivos ·", ", ".join(archivos[:4]), "…")
with np.load(apoyo.MODELOS / "vistas" / "v1" / "vectores_particion_0.npz") as d:
    print("\\nVectores de la rama 1, partición 0:", d["z"].shape, "(ejemplos, dimensiones)")
""")
    c.texto("""
Para cargar un codificador (requiere Keras ≥ 3.15, la versión con la que se guardaron):

```python
import keras
from temporal import modelos          # registra las capas propias
codificador = keras.saving.load_model("iteracion_2/modelos_finales/transitos/v1/particion_0.keras",
                                      compile=False, safe_mode=True)
```

## 3. La fusión, sin la rama de imagen

La cabeza recibe los vectores de las ramas 1, 2 y 4 y una bandera por rama. Se entrenó con cinco
semillas. Además de la fusión completa se entrenaron variantes para saber **qué aporta cada parte**:
""")
    c.codigo("""
filas = []
for nombre in ("tres_ramas", "tres_ramas_ajuste_cruzado", "solo_curvas", "solo_vistas", "solo_transitos", "solo_escalares"):
    m = apoyo.leer_json(apoyo.RESULTADOS / "fusion" / nombre / "metricas.json")
    g = m["semilla_0"]["global"]
    filas.append({"configuración": apoyo.FUSIONES[nombre], "ramas": " + ".join(m["ramas"]),
                  "AUC por semilla": " ".join(f"{a:.4f}" for a in m["auc_por_semilla"]),
                  "desviación entre semillas": m["auc_desviacion_semillas"],
                  "aciertos (semilla 0)": g["exactitud"]["estimacion"], "NLL": g["nll"]["estimacion"]})
display(pd.DataFrame(filas).set_index("configuración"))
""")
    c.texto("""
La variación entre semillas de la cabeza es mínima: el resultado no depende de la suerte del arranque.

## 4. Un riesgo conocido: el apilado

La cabeza se entrena con vectores de ejemplos que los codificadores **ya vieron**, y se evalúa con
vectores de ejemplos que no vieron. Si los codificadores memorizaron, la cabeza aprende de vectores
demasiado buenos. Se mide comparando su pérdida en entrenamiento y en validación, y se prueba un
remedio (*ajuste cruzado*: la cabeza solo ve vectores que los codificadores no usaron).
""")
    c.codigo("""
m = apoyo.leer_json(apoyo.RESULTADOS / "fusion" / "tres_ramas" / "metricas.json")
print("Apilado en la fusión de tres ramas:", m["apilado"])
dif = apoyo.comparacion()["diferencias"]
clave = next(k for k in dif if "efecto del apilado" in k)
print("Fusión normal − fusión con ajuste cruzado (AUC):", apoyo.con_intervalo(dif[clave], signo=True),
      "→ distinguible:", apoyo.distinguible(dif[clave]))
""")
    c.texto("""
**Lectura.** La pérdida de validación es cerca del doble de la de entrenamiento: los codificadores sí
memorizan algo. Pero el remedio no cambia el AUC de forma distinguible, así que el resultado principal
no está inflado por el apilado.

## 5. Y la rama de imagen

No se inventó ni se rellenó: su bandera vale 0 y su vector no entra. Para sumarla basta con poner sus
vectores en `modelos_finales/imagen/<versión>/` con el mismo formato y volver a entrenar la cabeza
(minutos). Ningún codificador de este curso cambia.
""")
    return c


# ---------------------------------------------------------------------------------------------
def c05() -> Cuaderno:
    c = Cuaderno(
        "05_analisis_de_resultados.ipynb",
        "Fase 5 · Análisis de resultados",
        "¿la red multirrama aprende de las curvas, y es mejor que un modelo tabular sobre los mismos datos?",
    )
    c.texto("""
## 1. La comparación que decide

Todos los modelos se evalúan sobre **las mismas señales**, con las mismas particiones. Los modelos
tabulares son *gradient boosting* entrenados con los parámetros del catálogo (sin columnas con fuga):
uno sin el centroide y otro con él. La métrica es el AUC de planeta frente a falso positivo; los
intervalos remuestrean por estrella.
""")
    c.codigo("""
r = apoyo.comparacion()
print("Señales comparadas:", r["n_kois"], "| sin predicción de la fusión:", r["n_kois_sin_prediccion_de_la_fusion"])
orden = ["tabular_sin_centroide", "solo_transitos", "solo_vistas", "solo_curvas", "solo_escalares",
         "tabular_con_centroide", "tres_ramas_ajuste_cruzado", "tres_ramas"]
fig, eje = plt.subplots(figsize=(7.5, 4.2))
apoyo.bosque({k: r["auc"][k] for k in orden[::-1]}, "AUC planeta vs. falso positivo (intervalo del 95 %)", eje=eje)
apoyo.guardar(fig, "05_bosque_auc")
plt.show()
display(pd.DataFrame({apoyo.FUSIONES[k]: {"AUC": apoyo.con_intervalo(r["auc"][k])} for k in orden[::-1]}).T)
""")
    c.texto("""
## 2. Las diferencias pareadas

Comparar intervalos a ojo engaña. Lo correcto es calcular la **diferencia** en cada remuestreo.
""")
    c.codigo("""
filas = [{"comparación": k.replace("tres_ramas", "fusión").replace("_", " "),
          "diferencia de AUC": apoyo.con_intervalo(v, signo=True), "¿distinguible de cero?": apoyo.distinguible(v)}
         for k, v in r["diferencias"].items()
         if k.startswith(("tres_ramas -", "solo_curvas -"))]
display(pd.DataFrame(filas).set_index("comparación"))
criterio = r["criterio_declarado"]
print("Criterio declarado antes de entrenar:", criterio["regla"])
print("  diferencia:", apoyo.con_intervalo(criterio["diferencia"], signo=True), "→ se cumple:", criterio["se_cumple"])
""")
    c.texto("""
**Cómo leer esta tabla, sin sobrevender:**

- **El criterio declarado se cumple**, pero era fácil: la rama 4 incluye el centroide y ella sola ya
  supera al tabular sin centroide. Cumplirlo no prueba que las curvas aporten.
- **Solo curvas − tabular sin centroide** es la prueba de que la red aprende de la luz: sin ver ningún
  dato del catálogo, las ramas 1 y 2 superan a un modelo tabular de los parámetros del tránsito.
- **Fusión − tabular con centroide** es la comparación exigente. La fusión gana por poco, pero de
  forma distinguible. En la iteración 1 (262 señales) esta comparación había quedado en empate.
- **Las curvas solas no reemplazan al centroide:** quedan por debajo del tabular con centroide. Esa
  información espacial es la que debe aportar la rama de imagen.
- **Aporte de las curvas a la fusión:** la fusión completa supera a la rama de escalares sola. Las
  curvas suman información que los escalares no tienen.

## 3. Dónde acierta y dónde falla
""")
    c.codigo("""
tabla, p = apoyo.predicciones("tres_ramas")
y = tabla["clase"].to_numpy()
confusion = pd.crosstab(pd.Series(y, name="real").map(dict(enumerate(apoyo.CLASES))),
                        pd.Series(p.argmax(1), name="predicho").map(dict(enumerate(apoyo.CLASES))))
display(confusion.reindex(index=apoyo.CLASES, columns=apoyo.CLASES, fill_value=0))
print(f"Aciertos: {(p.argmax(1) == y).mean():.1%} de {len(y)} ejemplos (media de 5 semillas de la cabeza)")
""")
    c.codigo("""
m = apoyo.leer_json(apoyo.RESULTADOS / "fusion" / "tres_ramas" / "metricas.json")["semilla_0"]
banderas = {"fp_ss": "eclipse estelar", "fp_co": "desplazamiento de centroide",
            "fp_nt": "no parece tránsito", "fp_ec": "contaminación"}
por_tipo = pd.DataFrame({banderas[k]: {"falsos positivos": v["n"], "detectados": v["aciertos"],
                                        "fracción": v["aciertos"] / v["n"]} for k, v in m["acierto_fp_por_bandera"].items()}).T
por_tipo["fracción"] = por_tipo["fracción"].map("{:.1%}".format)
display(por_tipo.astype({"falsos positivos": int, "detectados": int}))
por_mes = pd.DataFrame(m["auc_por_tercil_mes"]).T.rename(columns={"n": "señales", "estimacion": "AUC"})
display(por_mes)
""")
    c.texto("""
Las banderas del catálogo se usan aquí **solo para describir** los errores; nunca fueron entradas.
El tipo más difícil es el que "no parece tránsito", y las señales débiles (tercil bajo de intensidad)
cuestan más que las fuertes, como era de esperar.

Errores tomados al azar con semilla fija, no elegidos a mano:
""")
    c.codigo("""
real = y < 2
errores = np.flatnonzero(real & (p.argmax(1) != y))
rng = np.random.default_rng(apoyo.protocolo()["metricas"]["bootstrap_semilla"])
muestra = np.sort(rng.choice(errores, size=min(8, len(errores)), replace=False))
vista = tabla.loc[muestra, ["id_ejemplo", "periodo_d", "mes"]].copy()
vista["real"] = [apoyo.CLASES[c] for c in y[muestra]]
vista["predicho"] = [apoyo.CLASES[c] for c in p[muestra].argmax(1)]
vista["P(planeta)"] = p[muestra, 0].round(3)
print(f"{len(errores)} errores entre {int(real.sum())} señales reales; se muestran {len(muestra)} al azar")
display(vista.set_index("id_ejemplo"))
""")
    c.texto("""
## 4. ¿Se puede confiar en la probabilidad?

Un buen orden no basta: si el sistema dice 0,9, unas nueve de cada diez deberían ser planetas.
""")
    c.codigo("""
q = p[real, 0] / (p[real, 0] + p[real, 1])   # P(planeta | planeta o falso positivo)
fig, eje = plt.subplots(figsize=(4.2, 4.2))
ece = apoyo.confiabilidad(q, y[real] == 0, apoyo.protocolo()["metricas"]["ece_intervalos"], eje)
apoyo.guardar(fig, "05_confiabilidad")
print(f"Error de calibración esperado (ECE): {ece:.3f}")
""")
    c.texto("""
## 5. ¿Cambia algo si los modelos se eligen por AUC? (exploratorio)

La misma fusión, con codificadores elegidos por AUC de validación en lugar de NLL.
""")
    c.codigo("""
e = apoyo.comparacion("fusion_exploratoria_auc")
print(e["caracter"])
display(pd.DataFrame({"elegidos por NLL (principal)": {apoyo.FUSIONES[k]: r["auc"][k]["estimacion"] for k in orden if not k.startswith("tabular")},
                      "elegidos por AUC (exploratorio)": {apoyo.FUSIONES[k]: e["auc"][k]["estimacion"] for k in orden if not k.startswith("tabular")}}))
""")
    c.texto("""
Las ramas que leen curvas mejoran un poco eligiendo por AUC, pero **la fusión completa no**. El criterio
declarado (NLL) queda respaldado a nivel de sistema.

## 6. La evaluación única de las señales prerregistradas

Las 76 señales apartadas antes de entrenar resultaron ser, todas, FALSE POSITIVE en DR25 y CANDIDATE
en la tabla acumulada: señales que alguien reabrió. **No tienen respuesta correcta**, así que no se
miden aciertos. La pregunta, declarada por escrito antes de ejecutar
(`evidencias/prerregistro_declaracion.md`), fue: ¿el sistema, entrenado con etiquetas DR25, les da
más probabilidad de planeta que a los falsos positivos que nadie ha reabierto?
""")
    c.codigo("""
pre = apoyo.leer_json(apoyo.RESULTADOS / "prerregistrados" / "resultado.json")
g = pre["global"]
print("A = P(una prerregistrada puntúa más que un falso positivo normal):", apoyo.con_intervalo(g["A"]))
print("Hipótesis A > 0,5 →", g["hipotesis_A_mayor_que_0.5"])
display(pd.DataFrame({"mediana de q": g["mediana_q"], "fracción con q > 0,5": g["fraccion_q_mayor_que_0.5"], "señales": g["n"]}))
display(pd.DataFrame(pre["por_tipo_de_fp_dr25"]).T)
detalle = pd.read_csv(apoyo.RESULTADOS / "prerregistrados" / "detalle.csv")
fig, eje = plt.subplots(figsize=(6, 2.6))
eje.hist(detalle["q_planeta"], bins=20, color="#2166ac")
eje.set_xlabel("q = P(planeta | planeta o falso positivo)"); eje.set_ylabel("señales prerregistradas")
eje.set_title("Cómo puntúa la fusión a las 76 señales reabiertas", loc="left")
apoyo.guardar(fig, "05_prerregistradas")
print("Con curva utilizable:", pre["disponibilidad"], "de", g["n"]["prerregistradas"])
""")
    c.texto("""
**Lectura, con cuidado.** La hipótesis queda respaldada: el sistema puntúa a las señales reabiertas
claramente por encima de los falsos positivos normales. Es coherente con que mire algo real en los
datos y no repita la etiqueta del catálogo. **No es una validación:** «candidato» no es «planeta»,
la mayoría de las 76 sigue recibiendo q bajo, y quienes reabrieron estas señales pudieron fijarse en
rasgos parecidos a los que ve la red. La evaluación se ejecutó una sola vez; el script se niega a repetirla.

## 7. Conclusiones y límites

**Lo que los datos sostienen**

1. Con 17 veces más señales, la fusión de tres ramas supera de forma distinguible al modelo tabular con
   centroide sobre las mismas señales. Con 262 señales no lo lograba.
2. Las ramas que leen las curvas, sin ningún dato del catálogo, superan a un modelo tabular sin centroide:
   la red aprende de la luz.
3. Las curvas suman a la fusión algo que los escalares solos no dan.
4. La búsqueda de hiperparámetros ayuda poco; la mejora grande vino de los datos.

**Lo que no se puede afirmar**

- La ventaja sobre el tabular con centroide es pequeña (milésimas de AUC).
- Los codificadores usan una sola semilla; no se midió su variación.
- `koi_score`, el comparador externo que contiene la decisión del Robovetter, sigue por encima; no es
  una comparación justa (ve lo que aquí está prohibido), pero es el techo de referencia.
- Las señales prerregistradas no tienen etiqueta verdadera: su evaluación es descriptiva, no una
  validación ni una prueba de calibración.
- Los aciertos de tres clases incluyen la clase *sin señal*, que la rama 4 reconoce por un atajo (nunca
  tiene centroide; ver cuaderno 1). La conclusión se apoya en el AUC planeta vs. falso positivo, que no lo usa.
- La rama de imagen no participa. Nada aquí habla de habitabilidad ni de atmósferas.
- Las cifras de la iteración 1 y la 2 no son comparables directamente: las muestras son distintas.
""")
    return c


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--muestra", action="store_true", help="regenera datos/muestra_dataset.npz")
    a = ap.parse_args(argv)
    if a.muestra:
        crear_muestra()
    for construir in (c01, c02, c03, c04, c05):
        construir().escribir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
