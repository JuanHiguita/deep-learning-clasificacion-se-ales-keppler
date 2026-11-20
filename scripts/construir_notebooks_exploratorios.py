"""Genera los dos notebooks exploratorios (spec 009).

    notebooks/01_exploracion_datos.ipynb               qué muestran los datos y qué rama motivan
    notebooks/02_experimentacion_hiperparametros.ipynb  la búsqueda de hiperparámetros, paso a paso

Uso:
    python scripts/construir_notebooks_exploratorios.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploracion_datos.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/02_experimentacion_hiperparametros.ipynb
"""

from pathlib import Path

import nbformat

PREPARACION = """
import sys
import warnings
from pathlib import Path

from IPython.display import display

warnings.filterwarnings("ignore")

# Ubicar la carpeta del repositorio y hacer visible su código (src/)
RAIZ = next(
    carpeta
    for carpeta in [Path.cwd(), *Path.cwd().parents]
    if (carpeta / "config" / "protocolo.json").is_file()
)
sys.path.insert(0, str(RAIZ / "src"))

from informe import datos, graficos
from informe.interactivo import Lista, controles

graficos.aplicar_estilo()
res = datos.cargar(RAIZ)
print(f"{len(res.ids)} ejemplos de {res.tabla.kepid.nunique()} estrellas, listos en `res`.")
"""


class Cuaderno:
    """Acumula celdas y escribe el notebook."""

    def __init__(self, nombre: str):
        self.nombre = nombre
        self.celdas = []

    def texto(self, markdown: str) -> None:
        self.celdas.append(nbformat.v4.new_markdown_cell(markdown.strip("\n")))

    def codigo(self, fuente: str) -> None:
        self.celdas.append(nbformat.v4.new_code_cell(fuente.strip("\n")))

    def escribir(self) -> None:
        notebook = nbformat.v4.new_notebook(
            cells=self.celdas,
            metadata={
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"},
            },
        )
        destino = Path(__file__).resolve().parents[1] / "notebooks" / self.nombre
        nbformat.write(notebook, destino)
        print("escrito", destino, "·", len(self.celdas), "celdas")


# =============================================================================================
# 01 · Exploración de datos
# =============================================================================================


def exploracion() -> Cuaderno:
    nb = Cuaderno("01_exploracion_datos.ipynb")
    nb.texto("""
# 01 · Exploración de datos: ¿por qué estas arquitecturas?

**Proyecto del curso de Deep Learning** · Juan Sebastian Higuita Riascos · Universidad Nacional de Colombia

Antes de entrenar una red hay que mirar los datos y preguntarse **qué información separa un planeta de un impostor y dónde está esa información**. Cada sección de este notebook hace una pregunta a los datos, y la respuesta motiva una rama del modelo:

| Sección | Pregunta a los datos | Decisión de arquitectura que motiva |
|---|---|---|
| 1 | ¿Qué variables sueltas separan las clases? | Rama 4: red densa pequeña sobre escalares, con aviso de dato faltante |
| 2 | ¿La forma del tránsito es distinta entre clases? | Rama 1: CNN 1D sobre la curva plegada, con 4 canales |
| 3 | ¿Cuántos tránsitos hay y qué tan limpio es cada uno? | Rama 2: leer tránsito por tránsito, máximo 32 |
| 4 | ¿Qué se pierde al promediar los tránsitos? | Rama 2: agregador (promedio, GRU o atención) |
| 5 | Resumen: de los datos a la arquitectura | — |

> Todo aquí es **descriptivo**: se mira el conjunto completo con sus etiquetas para entender el problema. Nada de este notebook elige un modelo ni un hiperparámetro; eso se hace en `02_experimentacion_hiperparametros.ipynb` con validación interna.
""")
    nb.codigo(PREPARACION + "\nfrom informe import exploracion, secciones\n")

    nb.texto("""
## 0 · Qué hay en el conjunto

Cada señal (KOI) da dos ejemplos: el tránsito real, etiquetado *planeta* o *falso positivo* por el catálogo DR25, y la misma estrella en una época sin tránsito (*sin señal*).
""")
    nb.codigo("secciones.resumen_de_datos(res)")

    nb.texto("""
## 1 · ¿Qué variables sueltas separan las clases?

Primero lo más simple: los números que resumen la señal (periodo, duración, profundidad, señal/ruido, número de tránsitos y desplazamiento de centroide).

**Qué mirar.** Si dos histogramas se superponen, esa variable por sí sola no separa. La tabla cuenta cuántas veces falta cada dato.
""")
    nb.codigo("""
tabla_faltantes = exploracion.escalares_por_clase(res)
display(tabla_faltantes)
""")
    nb.texto("""
**Lectura.**

- Ningún escalar separa las clases por sí solo: los histogramas se superponen. El **centroide** es la excepción parcial: los desplazamientos grandes solo aparecen en falsos positivos, pero muchos falsos positivos tienen un desplazamiento tan pequeño como el de un planeta.
- La información está en la **combinación**, y para combinar 7 números basta una red densa pequeña (una capa oculta). No hay estructura espacial ni temporal que justifique algo más grande.
- **Hay datos que faltan**, y no al azar: el centroide falta en varios falsos positivos y en todos los ejemplos *sin señal*. Por eso la rama 4 recibe, junto a cada valor, una **bandera de "falta"**: rellenar con 0 sin avisar le enseñaría al modelo un valor falso.
- Ese mismo patrón es una advertencia: "centroide ausente" delata la clase *sin señal*. No afecta a la métrica principal (planeta frente a falso positivo), y se declara como limitación.

### ¿Cuánto separa cada variable sola?

El AUC de una variable sola mide qué tan bien ordena planetas y falsos positivos sin ningún modelo (0,5 = nada; 1 = perfecto). En morado, estadísticos que solo existen si se miran los **tránsitos uno por uno**.
""")
    nb.codigo("""
tabla_poder = exploracion.poder_de_cada_variable(res)
display(tabla_poder)
""")
    nb.texto("""
**Lectura.** Las variables que más separan por sí solas **no son los escalares clásicos, sino las que comparan un tránsito con otro** (la dispersión de la profundidad entre tránsitos y el tránsito más desviado). Esa es la primera evidencia a favor de una rama que lea los tránsitos por separado: al plegar la curva, esa información se promedia y se pierde.

Es una descripción con todos los datos, no un resultado del modelo.
""")

    nb.texto("""
## 2 · ¿La forma del tránsito es distinta entre clases?

La **vista local** es la curva plegada alrededor del tránsito. Tiene 4 canales: todos los tránsitos, solo los pares, solo los impares y la fase 0,5 (media órbita después, donde una estrella doble mostraría su eclipse secundario).

**Qué mirar.** La línea es la forma típica (mediana) de cada clase y la franja, cuánto varía (cuartiles). De la clase *sin señal* solo se dibuja la mediana, punteada. Debajo de la figura se imprimen tres medidas de forma.
""")
    nb.codigo("exploracion.forma_media_por_clase(res)")
    nb.texto("""
**Lectura.**

- Las dos formas típicas se parecen mucho: la diferencia es sutil. El falso positivo típico es **más puntiagudo** (su fondo baja más allá de −1 y es más angosto a media profundidad: forma en V), mientras el planeta tiene el fondo más plano.
- En el canal de **fase 0,5**, una parte de los falsos positivos muestra una caída (el eclipse secundario de una estrella doble) que en los planetas casi no existe.
- Son diferencias de **forma local** (pendiente de los bordes, fondo plano o en punta), y aparecen siempre en la misma posición de la ventana. Eso es justo lo que detecta una **convolución 1D**: un filtro pequeño que recorre la curva buscando un patrón. Una red densa tendría que aprender el mismo patrón en cada posición por separado.
- Los **4 canales** permiten que el mismo filtro compare pares con impares, o el tránsito con la fase 0,5, en la misma posición. Por eso se apilan como canales y no como entradas separadas.
- La mediana de la clase *sin señal* es plana, como debe ser: la época desplazada no contiene el tránsito.
""")
    nb.codigo("""
# Mira señales concretas: compara un planeta con una estrella doble
senales = sorted(res.ids[res.y < 2])
ilustrativos = ["K03615.01|transito", "K00022.01|transito"]
opciones = ilustrativos + [s for s in senales if s not in ilustrativos]

controles(
    lambda id_ejemplo: graficos.ver_ejemplo(res, id_ejemplo),
    id_ejemplo=Lista("Señal:", opciones),
)
""")

    nb.texto("""
## 3 · ¿Cuántos tránsitos hay y qué tan limpio es cada uno?

Para leer tránsito por tránsito hacen falta dos cosas: que haya varios, y que cada uno se distinga del ruido.
""")
    nb.codigo("exploracion.transitos_por_senal(res)")
    nb.codigo("exploracion.ruido_de_un_transito(res)")
    nb.texto("""
**Lectura.**

- En periodo corto hay **muchos tránsitos por señal**, casi siempre más de 32. La rama 2 toma hasta 32, repartidos en el tiempo y mitad pares, mitad impares: suficiente para comparar tránsitos entre sí sin que el modelo crezca con el número de tránsitos.
- En la mayoría de las señales **un tránsito individual se ve**: el ruido de cada punto es bastante menor que la profundidad. Tiene sentido entonces darle a la red cada tránsito por separado.
- Esto también anticipa un límite: con tantos tránsitos limpios, **promediar ya funciona bien**. La ventaja de un agregador más flexible (atención) debería notarse con *pocos* tránsitos o más ruido, que es el caso de la zona habitable y no el de estos datos.
""")

    nb.texto("""
## 4 · ¿Qué se pierde al promediar los tránsitos?

Tres medidas que solo existen tránsito por tránsito: la diferencia entre impares y pares (firma de una estrella doble con dos eclipses distintos), la dispersión de la profundidad y el tránsito que más se aparta de los demás.
""")
    nb.codigo("exploracion.consistencia_entre_transitos(res)")
    nb.texto("""
**Lectura.** En los falsos positivos estas medidas tienen **colas largas** que los planetas casi no tienen: tránsitos que no se repiten igual. La curva plegada las borra por construcción.

De aquí salen las tres variantes de la rama 2, que comparten el mismo codificador por tránsito (una CNN 1D pequeña con los **mismos pesos** para todos los tránsitos, porque un tránsito es un tránsito sin importar cuál sea):

| Agregador | Qué supone sobre los datos | Qué puede aprovechar |
|---|---|---|
| Promedio | Todos los tránsitos valen lo mismo y el orden no importa | Nada más que la curva plegada; es la referencia |
| GRU | El orden cronológico importa | Cambios a lo largo del tiempo |
| Atención | Algunos tránsitos importan más; el orden no importa | El tránsito que se aparta de los demás |

Cuál gana es una pregunta empírica: la responden `informe_dl.ipynb` (protocolo fijo) y `02_experimentacion_hiperparametros.ipynb` (con búsqueda).
""")

    nb.texto("""
## 5 · De los datos a la arquitectura

| Lo que muestran los datos | Decisión | Dónde está en el código |
|---|---|---|
| 7 números que solo separan combinados; algunos faltan | Red densa de una capa oculta + banderas de faltante | `temporal/modelos.py · m4_escalares` |
| Diferencias de forma local, en posición fija, entre 4 vistas | CNN 1D con las vistas como canales | `m1_astronet` |
| Muchos tránsitos, cada uno visible | Codificador compartido por tránsito, hasta 32 | `codificador_por_transito`, `m2` |
| La información entre tránsitos se pierde al plegar | Agregador: promedio, GRU o atención | `m2(agregador=…)` |
| Ramas con información distinta y a veces ausente | Fusión de vectores con banderas de disponibilidad | `fusion/fusionar.py` |
| Pocas señales (245) | Modelos pequeños, dropout, aumentación, parada temprana y validación cruzada por estrella | `comun/entrenamiento.py` |

**Lo que este notebook no puede decir:** qué tamaño de red, qué tasa de aprendizaje o cuánto dropout conviene. Eso no se deduce mirando datos: se mide. Sigue en el notebook 02.
""")
    return nb


# =============================================================================================
# 02 · Experimentación con hiperparámetros
# =============================================================================================


def experimentacion() -> Cuaderno:
    nb = Cuaderno("02_experimentacion_hiperparametros.ipynb")
    nb.texto("""
# 02 · Experimentación: búsqueda de hiperparámetros

**Proyecto del curso de Deep Learning** · Juan Sebastian Higuita Riascos · Universidad Nacional de Colombia

El resultado principal del proyecto usa hiperparámetros **fijados de antemano** (`config/protocolo.json`). Este notebook responde dos preguntas que eso deja abiertas:

1. ¿Cuánto mejora cada modelo si se ajustan sus hiperparámetros?
2. ¿Cambian las conclusiones sobre las arquitecturas cuando cada una tiene su propia búsqueda?

| Sección | Contenido |
|---|---|
| 1 | Las reglas de la búsqueda (para no hacer trampa) |
| 2 | Un experimento en vivo: esta celda **entrena** |
| 3 | Resultados por arquitectura: qué configuración ganó y qué hiperparámetros importan |
| 4 | ¿Mejoró? Protocolo fijo frente a búsqueda, en la prueba |
| 5 | ¿Cambia la comparación entre arquitecturas? |
| 6 | Modelos guardados: cargar uno y volver a predecir |
| 7 | Conclusiones y límites |
""")
    nb.codigo(
        PREPARACION
        + """
import pandas as pd

from comun.protocolo import cargar_protocolo
from informe import experimentacion
from temporal import busqueda_ramas

protocolo = cargar_protocolo(RAIZ / "config" / "protocolo.json")
ejemplos = pd.read_csv(RAIZ / "compartido" / "ejemplos.csv")
ARQUITECTURAS = experimentacion.arquitecturas_disponibles(res)
print("Búsquedas terminadas:", ", ".join(experimentacion.NOMBRE[a] for a in ARQUITECTURAS))
"""
    )

    nb.texto("""
## 1 · Las reglas de la búsqueda

Buscar hiperparámetros es la forma más fácil de engañarse: si se prueban muchas configuraciones y se reporta la mejor **en la prueba**, el número sale inflado. Estas son las reglas (spec 009):

1. **El espacio se declara antes de entrenar** y no se toca después. Está en `src/temporal/busqueda_ramas.py`.
2. **Búsqueda aleatoria** con semilla fija: 8 configuraciones por arquitectura, la del protocolo (`cfg00`) y 7 sorteadas. Con 5 o 6 hiperparámetros, sortear cubre más valores distintos de cada uno que una rejilla del mismo tamaño (Bergstra y Bengio, 2012).
3. **Mismo presupuesto para todas las arquitecturas**: si una recibiera más intentos, ganaría por eso.
4. **Selección anidada.** Hay 5 modelos por configuración, uno por partición. El modelo de la partición *k* elige su configuración con su **validación interna** (la partición *k* + 1). La partición de prueba *k* nunca interviene.
5. **Se reportan las dos versiones**, protocolo fijo y con búsqueda, con su diferencia pareada e intervalo del 95 %.

El tamaño de los vectores (32, 32, 16) no se busca: lo fija el contrato de integración entre ramas.

**Prueba esto.** Cambia la arquitectura para ver su espacio de búsqueda.
""")
    nb.codigo("""
OPCIONES = [(experimentacion.NOMBRE[a], a) for a in ARQUITECTURAS]

controles(
    lambda arquitectura: experimentacion.espacio_de_busqueda(arquitectura),
    arquitectura=Lista("Arquitectura:", OPCIONES),
)
""")

    nb.texto("""
## 2 · Un experimento en vivo

La celda siguiente **entrena de verdad**: lanza una búsqueda pequeña de la rama 4 (la más rápida) en una carpeta aparte, para no tocar los resultados del proyecto. Cada configuración entrena 5 modelos (uno por partición) con parada temprana.

- Lo ya entrenado se guarda: la segunda vez que la ejecutes termina en segundos.
- Sube `N_CONFIGURACIONES` y se entrenan solo las que falten.
- Cambia `ARQUITECTURA` por `"m2_atencion"` para ver la misma búsqueda sobre una red más grande (tarda varios minutos por configuración).
""")
    nb.codigo("""
ARQUITECTURA = "m4_escalares"
N_CONFIGURACIONES = 3
CARPETA_DE_PRUEBA = RAIZ / "resultados" / "dl" / "busqueda_demo"  # no se versiona

# 1. Qué configuraciones se van a entrenar (la primera es siempre la del protocolo)
for indice, configuracion in enumerate(busqueda_ramas.configuraciones_de(ARQUITECTURA, N_CONFIGURACIONES)):
    print(f"cfg{indice:02d}", configuracion)

# 2. Entrenar las que falten y resumir con selección anidada
resumen = busqueda_ramas.ejecutar(
    ARQUITECTURA, res.dataset, ejemplos, protocolo, CARPETA_DE_PRUEBA, n=N_CONFIGURACIONES
)

# 3. Qué eligió cada partición y qué pasó en la prueba
print("\\nConfiguración elegida por cada partición:", resumen["elegida_por_particion"])
print("NLL de validación · protocolo:", resumen["nll_validacion"]["fijo"],
      "· con búsqueda:", resumen["nll_validacion"]["con_busqueda"])
print("AUC en prueba · protocolo:", resumen["auc_prueba"]["fijo"]["estimacion"],
      "· con búsqueda:", resumen["auc_prueba"]["con_busqueda"]["estimacion"])
""")
    nb.texto("""
**Cómo leerlo.** La NLL de validación "con búsqueda" siempre es menor o igual que la del protocolo: se eligió mirándola, así que es **optimista por construcción**. Lo que cuenta es el AUC en prueba, que no participó en la elección. Con 3 configuraciones esto es solo una demostración del mecanismo; los resultados del proyecto están en las secciones siguientes.
""")

    nb.texto("""
## 3 · Resultados por arquitectura

**Cómo leer la figura.** Cada columna es una configuración, ordenadas de mejor a peor. Los 5 puntos son los 5 modelos (uno por partición) y el rombo su media. En naranja, el protocolo. Con borde negro, los modelos que ganó cada partición en la selección anidada.

**Qué mirar.** Si los puntos de una misma configuración están muy separados, la partición pesa más que el hiperparámetro: la búsqueda tiene poco de dónde agarrarse.
""")
    nb.codigo("""
controles(
    lambda arquitectura: experimentacion.configuraciones(res, arquitectura),
    arquitectura=Lista("Arquitectura:", OPCIONES, inicial="m2_atencion"),
)
""")
    nb.texto("""
### ¿Qué hiperparámetros importan?

**Prueba esto.** Elige un hiperparámetro y mira si la NLL de validación sigue alguna tendencia. Con 8 puntos y todo cambiando a la vez, solo se pueden ver tendencias fuertes.
""")
    nb.codigo("""
HIPERPARAMETROS = [
    (nombre, clave) for clave, nombre in experimentacion.NOMBRE_DE_HIPERPARAMETRO.items()
]

controles(
    lambda arquitectura, hiperparametro: experimentacion.sensibilidad(res, arquitectura, hiperparametro),
    arquitectura=Lista("Arquitectura:", OPCIONES, inicial="m2_atencion"),
    hiperparametro=Lista("Hiperparámetro:", HIPERPARAMETROS),
)
""")
    nb.texto("""
### ¿Cómo entrenó una configuración frente al protocolo?

Azul: pérdida de entrenamiento. Rojo: validación interna. La línea vertical es la época que eligió la parada temprana.
""")
    nb.codigo("""
CONFIGURACIONES = [f"cfg{i:02d}" for i in range(1, busqueda_ramas.N_CONFIGURACIONES)]

controles(
    lambda arquitectura, configuracion: experimentacion.curvas(res, arquitectura, configuracion),
    arquitectura=Lista("Arquitectura:", OPCIONES[:-1], inicial="m2_atencion"),
    configuracion=Lista("Configuración:", CONFIGURACIONES, inicial="cfg05"),
)
""")

    nb.texto("""
## 4 · ¿Mejoró? Protocolo fijo frente a búsqueda

Ahora sí se mira la **prueba**. Para cada modelo: el AUC del protocolo, el AUC con la configuración que eligió cada partición, y la diferencia pareada (mismas estrellas remuestreadas) con su intervalo del 95 %.
""")
    nb.codigo("""
tabla_mejora = experimentacion.fijo_contra_busqueda(res)
display(tabla_mejora)
""")
    nb.texto("""
**Lectura.**

- **En ningún modelo la mejora se distingue de cero.** El caso más favorable es la rama 2 con promedio, cuyo intervalo queda rozando el 0.
- En la rama 4, la rama 1 y la cabeza de fusión, la búsqueda quedó igual o apenas por debajo del protocolo: los valores fijados de antemano ya estaban en una zona razonable.
- **Cuidado con el caso del promedio.** Su punto de partida (semilla 0 del protocolo) fue la más baja de sus tres semillas, así que parte de la subida puede ser simple variación entre entrenamientos y no efecto de los hiperparámetros. Con una sola semilla no se pueden separar las dos cosas.
- La validación mejoró en todos (se eligió mirándola), pero esa mejora **no se trasladó** a la prueba de forma medible. Es la señal clásica de que, con ~100 ejemplos de validación por partición, la selección persigue ruido.
""")

    nb.texto("""
## 5 · ¿Cambia la comparación entre arquitecturas?

La pregunta 1 del proyecto compara arquitecturas con un protocolo común. Aquí se repite la comparación cuando **cada una tuvo su propia búsqueda**.
""")
    nb.codigo("""
tabla_pares = experimentacion.pares_con_busqueda(res)
display(tabla_pares)
""")
    nb.texto("""
**Lectura.** Con búsqueda, **ninguna diferencia entre arquitecturas temporales se distingue de cero**. La atención sigue teniendo el AUC más alto, pero su ventaja sobre el promedio, la GRU y la CNN de curva plegada cabe dentro del intervalo.

Esto **refuerza** la conclusión del protocolo fijo ("la atención no supera al promedio de forma demostrable") y matiza la única diferencia que allí salía distinguible (atención sobre GRU): con una semilla y búsqueda, tampoco se distingue.
""")

    nb.texto("""
## 6 · Modelos guardados

El modelo que eligió cada partición quedó guardado completo en `modelos/mejores/<arquitectura>/`, con la estandarización de sus entradas y un manifiesto (hiperparámetros y huella SHA-256). Al guardarlo se comprobó que reproduce las predicciones de la búsqueda.
""")
    nb.codigo("""
guardados = experimentacion.modelos_guardados(res)
display(guardados[["modelo", "partición", "archivo", "configuración"]])
""")
    nb.texto("""
### Cargar un modelo y volver a predecir

`temporal/despliegue.py` carga el modelo, verifica su huella y aplica la misma estandarización. Se predice la partición que ese modelo **nunca vio** y se compara con lo reportado: la diferencia debe ser 0.
""")
    nb.codigo("""
controles(
    lambda arquitectura, particion: experimentacion.probar_modelo_guardado(res, arquitectura, particion),
    arquitectura=Lista("Arquitectura:", OPCIONES[:-1], inicial="m2_atencion"),
    particion=Lista("Partición:", [0, 1, 2, 3, 4]),
)
""")

    nb.texto("""
## 7 · Conclusiones y límites

**Qué se concluye**

1. Con el presupuesto usado (8 configuraciones, una semilla), ajustar hiperparámetros **no mejora ningún modelo de forma distinguible** en la prueba.
2. La comparación entre arquitecturas **no depende** de haber fijado el protocolo: con búsqueda, las arquitecturas temporales siguen sin distinguirse.
3. Por eso **el resultado principal sigue siendo el del protocolo fijo**, y esta búsqueda se reporta como análisis de sensibilidad (decisión D-3 de la spec 009). No se rehízo la fusión con los modelos buscados.

**Qué no se puede concluir**

- Que estos modelos sean "los mejores posibles". Se exploró un espacio pequeño; una búsqueda mayor, o con más datos, podría encontrar algo mejor.
- Que la búsqueda no sirva: un resultado no distinguible con una semilla y 245 señales no prueba que no haya mejora, solo que aquí no se puede medir.

**Límites declarados**

- **Una semilla.** La variación entre semillas de una misma rama es del mismo orden que las diferencias observadas.
- **Optimismo de selección.** La validación interna también decide la parada temprana.
- **Límite de cómputo en la rama 1:** su espacio no incluye el ancho doble ni el lote de 16.
- **Dependencia numérica.** El entrenamiento es reproducible en la misma máquina y configuración de hilos; al cambiarla, una partición dio pesos distintos y hubo que reentrenarla. Diferencias así de pequeñas bastan para mover un intervalo que roza el 0.
""")
    return nb


if __name__ == "__main__":
    exploracion().escribir()
    experimentacion().escribir()
