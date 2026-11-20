"""Genera notebooks/03_estudio_de_diseno.ipynb (specs 010 y 011).

Uso:
    python scripts/construir_notebook_diseno.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/03_estudio_de_diseno.ipynb
"""

from pathlib import Path

import nbformat

CELDAS = []


def texto(markdown: str) -> None:
    CELDAS.append(nbformat.v4.new_markdown_cell(markdown.strip("\n")))


def codigo(fuente: str) -> None:
    CELDAS.append(nbformat.v4.new_code_cell(fuente.strip("\n")))


texto("""
# 03 · Estudio de diseño, menos tránsitos y sistema conjunto

**Proyecto del curso de Deep Learning** · Juan Sebastian Higuita Riascos · Universidad Nacional de Colombia

El notebook 02 varió hiperparámetros **numéricos**. Este responde lo que faltaba: ¿y si se cambia una **decisión de diseño**?

| Sección | Pregunta |
|---|---|
| 1 | ¿Qué se probó y qué no? |
| 2 | ¿Qué pasa al cambiar una sola decisión? (activación, normalización, optimizador, profundidad, pérdida, preprocesamiento) |
| 3 | ¿Las mejoras de la rama 1 se confirman con 3 semillas? |
| 4 | ¿Qué aportan los estadísticos entre tránsitos? ¿Y qué pasa con menos tránsitos? |
| 5 | El sistema conjunto y la prueba de extremo a extremo |
| 6 | Conclusiones y lo que queda por decidir |

Este notebook **no entrena**: lee `resultados/dl/diseno/`, `resultados/dl/e4/` y `resultados/dl/e2e/`. La sección 5 sí ejecuta el sistema (solo inferencia).
""")

codigo("""
import sys
import warnings
from pathlib import Path

from IPython.display import display

warnings.filterwarnings("ignore")

RAIZ = next(
    carpeta
    for carpeta in [Path.cwd(), *Path.cwd().parents]
    if (carpeta / "config" / "protocolo.json").is_file()
)
sys.path.insert(0, str(RAIZ / "src"))

from informe import datos, diseno, graficos
from informe.interactivo import Lista, controles

graficos.aplicar_estilo()
res = datos.cargar(RAIZ)
MODELOS = [(diseno.NOMBRE[a], a) for a in diseno.disponibles(res)]
print("Modelos con estudio de diseño:", ", ".join(nombre for nombre, _ in MODELOS))
""")

texto("""
## 1 · Qué se probó y qué no

Cada variante cambia **una sola cosa** respecto al protocolo y se entrena con el mismo motor, las mismas particiones y la semilla 0. La comparación es la diferencia pareada de AUC en prueba, con su intervalo del 95 %.

| Factor | Protocolo | Variantes probadas |
|---|---|---|
| Activación | ReLU | GELU, tanh |
| Normalización | ninguna | por lote (BatchNorm), por capa (LayerNorm) |
| Optimizador | AdamW | Adam sin decaimiento, SGD con momento de Nesterov |
| Profundidad | 2–3 bloques (ramas 1 y 2), 1 capa oculta (rama 4) | una capa o bloque menos, una más |
| Pérdida | entropía cruzada con pesos por clase | sin pesos, suavizado de etiquetas 0,1, focal γ = 2 |
| Preprocesamiento | unidades de profundidad + aumentación | sin aumentación, *z-score* por serie; rama 4 sin estandarizar |
| Pooling (rama 1) | máximo | promedio, convolución con paso 2 |
| Entradas (rama 2) | con 4 estadísticos de consistencia | sin ellos |

**No se probó:** combinaciones de factores, otras familias de arquitectura (por ejemplo un Transformer completo o una BiGRU sobre la curva plegada), ajuste fino conjunto de las ramas ni un ensamble de codificadores. Tampoco existe una variante "sin banderas de dato faltante": rellenar con 0 sin avisar va contra las reglas del proyecto.

> **Cómo leer todo lo que sigue.** Son unas 15 comparaciones por modelo con una sola semilla. Con ese número, que alguna "se distinga de cero" por puro azar es esperable. Por eso se fijó de antemano (spec 010) que una variante solo cuenta si además se **confirma con 3 semillas**.
""")

texto("""
## 2 · ¿Qué pasa al cambiar una sola decisión?

**Cómo leer la figura.** Cada fila es una variante. El punto es cuánto cambia el AUC respecto al protocolo; a la derecha de 0 mejora, a la izquierda empeora. En verde, las que se distinguen de cero.

**Prueba esto.** Cambia de modelo: las tres ramas no reaccionan igual.
""")

codigo("""
controles(
    lambda arquitectura: diseno.efecto_de_cada_decision(res, arquitectura),
    arquitectura=Lista("Modelo:", MODELOS, inicial="m1_astronet"),
)
""")

codigo("""
# Las otras dos ramas, fijas, para poder comparar sin ejecutar
display(diseno.efecto_de_cada_decision(res, "m2_atencion"))
display(diseno.efecto_de_cada_decision(res, "m4_escalares"))
""")

texto("""
**Lectura.**

- **Rama 4 (escalares):** nada la mejora ni la empeora de forma distinguible, salvo quitar la estandarización, que la destruye. Es un modelo tan pequeño que estas decisiones no importan; la estandarización sí es imprescindible.
- **Rama 2 (atención):** ninguna variante la mejora. Dos la empeoran de forma distinguible: entrenar **sin pesos por clase** y quitar los **estadísticos de consistencia**.
- **Rama 1 (AstroNet):** es la que más reacciona. Con una semilla, dos variantes mejoran de forma distinguible (suavizado de etiquetas y *z-score*) y varias más apuntan en la misma dirección (tanh, pooling promedio, normalización por capa). La siguiente sección comprueba si eso se sostiene.

### El mismo resumen, eligiendo con la validación

Para no elegir mirando la prueba, esta tabla toma de cada factor la variante con **mejor NLL de validación** y muestra qué le pasó en la prueba.
""")

codigo("""
display(diseno.resumen_por_factor(res))
""")

texto("""
## 3 · ¿Las mejoras de la rama 1 se confirman?

Se repitieron con las 3 semillas del protocolo tres variantes de la rama 1: las dos que se distinguieron en la prueba y la mejor en validación (tanh). Se compara la media de las 3 semillas con la del protocolo, y también **sin la semilla 0**, que fue la que motivó elegirlas.
""")

codigo("""
display(diseno.confirmacion(res))
""")

texto("""
**Lectura.** Las tres se **confirman**: la mejora se distingue de cero con 3 semillas, y también sin la semilla que las seleccionó. La rama 1 del protocolo estaba por debajo de lo que la misma arquitectura puede dar.

**Qué significa y qué no.**

- Es un resultado **de la rama sola**. No se sabe si mejora el sistema: la fusión ya recibe por otras ramas parte de lo que estas variantes añaden. Para saberlo hay que reexportar el codificador y rehacer fusión, calibración e incertidumbre, y eso es una decisión que cambia el resultado principal del proyecto.
- **Sobre el *z-score*** (interpretación, no comprobada): el proyecto normaliza por profundidad a propósito, para que la rama 1 mire solo la *forma*. El *z-score* deja pasar además cuán fuerte es la señal respecto al ruido, que es información que ya tiene la rama 4. Parte de su ganancia puede ser esa información duplicada.
- Las variantes se eligieron mirando resultados de prueba de la semilla 0. Aunque la confirmación sin esa semilla también se sostiene, sigue siendo el mismo conjunto de estrellas.
""")

texto("""
## 4 · Estadísticos entre tránsitos y menos tránsitos

### E2b · ¿Qué pierde cada agregador sin los estadísticos de consistencia?

La rama 2 recibe, además de los tránsitos, 4 números calculados a mano (dispersión de la profundidad, diferencia impar − par, número de tránsitos y tránsito más desviado). Aquí se le quitan.
""")

codigo("""
display(diseno.sin_consistencia(res))
""")

texto("""
**Lectura.** El **promedio** es el que más pierde: por construcción no puede comparar un tránsito con otro, así que dependía de esos números. La **atención** también pierde de forma distinguible. La **GRU** pierde menos y su caída no se distingue de cero.

### E4 · ¿Y con menos tránsitos?

Con los modelos **sin** estadísticos de consistencia (para que lo único que cambie sea cuántos tránsitos ve el agregador), en la prueba se dejan 32, 16, 8, 4 y 2 tránsitos elegidos al azar.
""")

codigo("""
display(diseno.menos_transitos(res))
""")

texto("""
**Lectura.**

- **Sin los estadísticos hechos a mano, la atención sí supera al promedio** de forma distinguible con 32, 16 y 8 tránsitos. Es decir: la atención aprende sola lo que al promedio había que dárselo calculado. Frente a la GRU no hay diferencia.
- **Con 4 y 2 tránsitos la ventaja deja de distinguirse.** Es lo contrario de lo que se esperaba (que la atención ayudara más con pocos tránsitos).

**Límites.** Esto simula pocos tránsitos con señales de periodo corto, donde cada tránsito sigue siendo limpio; no es el régimen de zona habitable. Los modelos se entrenaron viendo entre 4 y 32 tránsitos, así que 2 es extrapolar. Una semilla.
""")

texto("""
## 5 · El sistema conjunto

`fusion/sistema.py` encadena las piezas tal como se usan juntas:

```
entradas ─► 4 codificadores congelados ─► cabeza de fusión ─► temperatura ─► prior explícito
```

Hay un sistema por partición. Una estrella del conjunto se responde con el de **su** partición (codificadores y cabeza que nunca la vieron); una estrella nueva, con el promedio de los cinco.

**Prueba esto.** Elige una señal y cambia la tasa base declarada.
""")

codigo("""
senales = ["K00022.01|transito", "K03615.01|transito"]
senales += [s for s in sorted(res.ids[res.y < 2]) if s not in senales]

controles(
    lambda id_ejemplo, tasa_base: diseno.respuesta_del_sistema(res, id_ejemplo, tasa_base),
    id_ejemplo=Lista("Señal:", senales),
    tasa_base=Lista("Tasa base:", [0.05, 0.02, 0.1, 0.2, 0.5]),
)
""")

texto("""
### Prueba de extremo a extremo

`scripts/prueba_e2e.py` recorre todo el camino para estrellas elegidas al azar con semilla fija, desde los archivos FITS de Kepler, y compara con lo publicado en tres puntos: las entradas, el vector de cada rama y la probabilidad calibrada. Las columnas son la diferencia máxima en cada punto.
""")

codigo("""
display(diseno.prueba_de_extremo_a_extremo(res))
""")

texto("""
**Límites del sistema.**

- La imagen de píxeles (rama 3) no se vuelve a derivar en la prueba: se toma del archivo que produce el script de la demo, sin modificar.
- Llega hasta la probabilidad calibrada con prior explícito. **Lo que la tesis añade y aquí no existe:** inyecciones, falsas alarmas reconstruidas y la combinación con P(en zona habitable y rocoso) calculada con Gaia.
- Rango de validez: señales de Kepler de 1 a 30 días.

## 6 · Conclusiones

1. **Rama 4 y rama 2:** las decisiones de diseño del protocolo se sostienen. Ninguna alternativa las mejora, y dos de ellas (pesos por clase y estadísticos de consistencia en la rama 2) resultan necesarias.
2. **Rama 1:** el protocolo **no** era la mejor opción. Suavizado de etiquetas, *z-score* y tanh la mejoran de forma distinguible y confirmada con 3 semillas.
3. **Atención frente a promedio:** depende de lo que se le dé al promedio. Con los estadísticos hechos a mano empatan; sin ellos, la atención gana. No gana más con pocos tránsitos.
4. **El sistema conjunto funciona de extremo a extremo** y reproduce lo publicado.

**Lo que queda por decidir:** si se adopta una variante de la rama 1 y se rehace el resultado principal (fusión, calibración, incertidumbre). Hacerlo eligiendo con la validación, no con la prueba, y declararlo como cambio de protocolo.
""")


def main() -> None:
    notebook = nbformat.v4.new_notebook(
        cells=CELDAS,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
    )
    destino = Path(__file__).resolve().parents[1] / "notebooks" / "03_estudio_de_diseno.ipynb"
    nbformat.write(notebook, destino)
    print("escrito", destino, "·", len(CELDAS), "celdas")


if __name__ == "__main__":
    main()
