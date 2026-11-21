# Guía paso a paso: correr, recorrer y leer el proyecto

**Proyecto del curso de Deep Learning** · Clasificación calibrada de señales de tránsito de Kepler

Esta guía es para ti, sentado frente al PC. No hace falta leerla entera: elige tu ruta.

| Si quieres… | Ve a | Tiempo |
|---|---|---|
| Entender de qué va en un minuto | [1. La idea](#1-la-idea-en-un-minuto) | 1 min |
| Ver los resultados sin correr nada | [Ruta A](#ruta-a--solo-mirar-5-minutos) | 5 min |
| Comprobar que todo funciona en tu PC | [Ruta B](#ruta-b--comprobar-que-todo-funciona) | unos minutos |
| Reproducir todo desde cero | [Ruta C](#ruta-c--reproducir-desde-cero) | horas |
| Saber leer una figura | [4. Cómo leer los resultados](#4-cómo-leer-los-resultados) | 5 min |
| Encontrar un archivo o una cifra | [5. Dónde está cada cosa](#5-dónde-está-cada-cosa) | — |

> **Aviso.** Todo se desarrolló y se ejecutó en Linux (Python 3.10, Keras 3.12). En tu Windows (Python 3.13, Keras 3.15) todavía no se ha corrido nada salvo la descarga de datos. Si algo falla, mira la [sección 7](#7-si-algo-falla).

---

## 1. La idea en un minuto

Kepler midió el brillo de miles de estrellas durante 4 años. Cuando un planeta pasa por delante de su estrella, el brillo baja un poco: eso es un **tránsito**. El problema es que hay impostores que producen la misma bajada (estrellas dobles, estrellas vecinas, fallas del instrumento).

El proyecto mira cada señal de **cuatro formas** y combina las cuatro opiniones:

```
curva plegada ───────────► Rama 1 · CNN 1D ─────────────┐
tránsitos uno por uno ───► Rama 2 · atención ───────────┤
imagen de píxeles ───────► Rama 3 · CNN 2D (de la demo) ┼─► fusión ─► calibración ─► probabilidad
7 números de resumen ────► Rama 4 · red densa ──────────┘
```

Y responde tres preguntas:

1. **¿Qué arquitectura lee mejor la señal?**
2. **¿Cuánto mejora la decisión al fusionar las cuatro ramas?**
3. **¿Se puede confiar en las probabilidades?**

La rama 3 es código anterior a los cursos y se usa congelada: no es un aporte de este curso.

---

## 2. Tres rutas

Todos los comandos se escriben en una terminal (`cmd`) abierta en la carpeta del proyecto:

```bat
cd /d "D:\Proyectos Especializacion\Deep Learning\Proyecto_Final"
```

También puedes hacer doble clic sobre cada `.bat` en el Explorador.

### Ruta A · Solo mirar (5 minutos)

```bat
scripts\abrir_notebook.bat
```

Se abre Jupyter con la carpeta `notebooks`. Los notebooks ya traen las figuras. Léelos en este orden:

| Orden | Notebook | Qué responde |
|---|---|---|
| 1 | `01_exploracion_datos` | ¿Qué hay en los datos y por qué cada rama es como es? |
| 2 | `informe_dl` | **Los resultados del proyecto.** Si solo abres uno, que sea este |
| 3 | `02_experimentacion_hiperparametros` | ¿Y si se ajustan los hiperparámetros? |
| 4 | `03_estudio_de_diseno` | ¿Y si se cambia la activación, la pérdida, el optimizador…? ¿Cómo responde el sistema completo? |

**Para activar los controles** (listas y deslizadores): menú *Run → Run All Cells*. Sin ejecutar ves cada figura con sus valores iniciales; ejecutando puedes elegir la señal, la métrica, la temperatura, etc.

> Si no quieres abrir Jupyter, el mismo contenido está resumido en `docs\EXPLICACION_RESULTADOS.md`.

### Ruta B · Comprobar que todo funciona

```bat
scripts\verificar.bat
```

Hace cinco comprobaciones y se detiene en la primera que falle:

| Paso | Qué comprueba | Qué debes ver |
|---|---|---|
| 1 | Pruebas rápidas del código | `111 passed` |
| 2 | Pruebas que entrenan un modelo pequeño | `2 passed` |
| 3 | Contrato entre ramas (formas, identificadores, sin columnas prohibidas) | `CONTRATO: correcto` |
| 4 | **Extremo a extremo:** de los archivos de Kepler a la probabilidad final, en 10 estrellas al azar | `PRUEBA E2E: correcta` |
| 5 | Auditoría (fugas, particiones, cifras de los documentos) | `22 de 22 comprobaciones correctas` |

Si termina en `TODO CORRECTO`, tu copia produce lo mismo que está publicado.

### Ruta C · Reproducir desde cero

Son tres pasos, y cada uno es **reanudable**: si se corta, lo vuelves a lanzar y sigue donde iba.

| Paso | Comando | Qué hace | Tiempo |
|---|---|---|---|
| 1 | `scripts\descargar_recursos.bat` | Crea el entorno, descarga las tablas de NASA y las curvas de luz (1,6 GB) y construye las entradas | Una hora o más (ya lo hiciste) |
| 2 | `scripts\reproducir_todo.bat` | Entrena las ramas, la fusión, la calibración y la incertidumbre, y rehace `informe_dl` | Del orden de una hora |
| 3 | `scripts\experimentos_extra.bat` | Búsqueda de hiperparámetros, estudio de diseño, sistema conjunto y copia experimental | Varias horas (en el entorno de desarrollo fueron unas 4; en tu PC no está medido) |

**Importante:** lo ya entrenado no se repite. Para volver a entrenar algo de verdad, borra su carpeta dentro de `resultados\dl\`.

---

## 3. El mapa del repositorio

```
Proyecto_Final/
├── notebooks/          ← EMPIEZA AQUÍ: lo que se lee
├── docs/               ← explicaciones en texto
│   ├── GUIA_PASO_A_PASO.md        esta guía
│   ├── EXPLICACION_RESULTADOS.md  todo el proyecto en un documento
│   ├── AUDITORIA.md               qué se revisó y qué queda abierto
│   └── planeacion/BITACORA_DL.md  diario: qué se hizo cada día y qué salió
├── resultados/dl/      ← las cifras
│   ├── tablas/                    ÚNICO origen de las cifras de los documentos
│   ├── fusion/, calibracion/, incertidumbre/   resultado principal
│   ├── busqueda/, diseno/, e4/    estudios complementarios
│   └── experimental_v2/           copia experimental (no es el resultado principal)
├── modelos/            ← los pesos guardados (ver modelos/LEEME.md)
├── src/                ← el código
│   ├── comun/                     datos, particiones, motor de entrenamiento, métricas
│   ├── temporal/                  ramas 1, 2 y 4
│   ├── fusion/                    imagen v0, fusión, calibración, sistema, auditoría
│   └── informe/                   figuras de los notebooks
├── scripts/            ← los .bat y los generadores de notebooks
├── specs/              ← qué se pidió y por qué, antes de programar cada parte
├── config/protocolo.json   reglas fijas: particiones, semillas, hiperparámetros
└── data/               ← datos descargados (no van a git)
```

**Regla para no perderse:** una cifra de un documento → está en `resultados/dl/tablas/` → la produjo un script de `src/` → que se pidió en una carpeta de `specs/`.

---

## 4. Cómo leer los resultados

### 4.1 La figura que más vas a ver

Casi todo se muestra así: un punto con una barra horizontal.

```
Fusión de 4 ramas      ├────●────┤   0.985
Solo escalares    ├────────●────────┤   0.954
```

- **El punto** es la estimación.
- **La barra** es el intervalo de confianza del 95 %: el rango donde razonablemente está el valor real. Con solo 245 señales, las barras son anchas.
- **Dos barras que se solapan no prueban nada.** Para comparar dos modelos se usa la figura de *diferencias*.

### 4.2 La figura de diferencias (la que decide)

```
                         0
fusión − solo escalares  │   ├───●───┤      ← toda a la derecha de 0: SE DISTINGUE
atención − promedio   ├──┼──●─────┤         ← cruza el 0: NO se distingue
```

- Línea punteada en 0 = "no hay diferencia".
- Si **toda** la barra queda a un lado del 0, la diferencia **se distingue** (en los notebooks sale en verde).
- Si la barra **cruza el 0**, no se puede afirmar que un modelo sea mejor. **No es lo mismo que "son iguales"**: puede haber una diferencia demasiado pequeña para verla con estos datos.

### 4.3 Vocabulario mínimo

| Palabra | Qué es | Cómo leerla |
|---|---|---|
| **AUC** | Qué tan bien ordena planetas por encima de falsos positivos | 0,5 = al azar · 1 = perfecto |
| **NLL** | Castigo por estar seguro y equivocarse | Menor es mejor |
| **ECE** | Cuánto miente la confianza (dice 90 % y acierta 80 %) | Menor es mejor · 0 = perfecto |
| **IC 95 %** | Rango razonable para el valor real | Barra de la figura |
| **Partición** | Uno de 5 grupos de estrellas | Cada estrella se evalúa con un modelo que nunca la vio |
| **Semilla** | Número que fija el azar del entrenamiento | Se repite con varias para ver cuánto cambia el resultado |
| **Validación interna** | Estrellas apartadas para tomar decisiones (cuándo parar, qué elegir) | La prueba nunca decide nada |
| **Protocolo** | Los hiperparámetros fijados antes de entrenar | `config/protocolo.json` |
| **Calibrar** | Ajustar las probabilidades para que digan la verdad | Se hace con un solo número, la temperatura |
| **Tasa base (prior)** | Qué fracción de las señales son planetas en la población | Si los planetas son raros, hay que bajar las probabilidades |

### 4.4 Cinco trampas al leer

1. **"No se distingue" no significa "es igual".** Significa que con 245 señales no se puede saber.
2. **Una sola semilla puede engañar.** Las búsquedas y el estudio de diseño usan una; por eso lo importante se confirmó con tres.
3. **Mejorar una rama no mejora el sistema.** Pasó: la rama 1 subió de 0,865 a 0,917 y la fusión quedó igual (ver 6.5).
4. **Todo es periodo corto (1–30 días).** Nada de esto dice algo sobre planetas en zona habitable.
5. **El modelo tabular simple ya llega a 0,98–0,99.** La fusión queda a su nivel, no por encima.

---

## 5. Dónde está cada cosa

| Pregunta | Tabla (`resultados/dl/tablas/`) | Notebook y sección |
|---|---|---|
| ¿Cuál es el listón? | `lineas_base.csv` | `informe_dl` · 2 |
| P1 · ¿Qué arquitectura? | `p1_arquitecturas.csv` | `informe_dl` · 3 y 4 |
| P2 · ¿Cuánto aporta la fusión? | `p2_fusion.csv` | `informe_dl` · 5 |
| P3 · ¿Probabilidades confiables? | `p3_calibracion.csv`, `p3_incertidumbre.csv` | `informe_dl` · 6 |
| Veredicto de cada hipótesis | `resumen_hipotesis.csv` | `informe_dl` · 8 |
| ¿Sirve ajustar hiperparámetros? | `busqueda_resumen.csv`, `busqueda_pares.csv` | `02` · 4 y 5 |
| ¿Sirve cambiar activación, pérdida…? | `diseno.csv`, `diseno_confirmacion.csv` | `03` · 2 y 3 |
| ¿Y con menos tránsitos? | `e4_menos_transitos.csv` | `03` · 4 |
| ¿Funciona el sistema completo? | `../e2e/informe.json` | `03` · 5 |
| ¿Mejora el sistema con la rama 1 nueva? | `../experimental_v2/comparacion.csv` | `docs/EXPLICACION_RESULTADOS.md` · 5.9 |
| Mirar una señal concreta | — | `informe_dl` · 7 (explorador) |

---

## 6. Los resultados, en una frase cada uno

### 6.1 Arquitectura (pregunta 1)
La atención tuvo el mejor AUC de las ramas temporales (0,883), pero su ventaja sobre el promedio (0,863) **no se distingue**.
- **Matiz que salió después:** si al promedio se le quitan los 4 números calculados a mano que comparan tránsitos, la atención **sí** gana. La atención aprende sola lo que al promedio hay que dárselo hecho.

### 6.2 Fusión (pregunta 2)
Unir las cuatro ramas da **0,985 [0,974–0,994]** y **sí supera** a cualquier rama sola (la mejor, 0,954) y a la demo.
- **Límite:** queda al nivel del modelo tabular con centroide (0,979–0,989), no por encima.

### 6.3 Calibración (pregunta 3)
El sistema salió algo sobreconfiado. La temperatura lo corrige en la dirección esperada, sin que la mejora se distinga. **El prior explícito sí funciona** cuando los planetas son escasos. La duda del modelo **señala sus errores**.

### 6.4 Hiperparámetros
Con 8 configuraciones por modelo, ajustar hiperparámetros **no mejoró ninguno** de forma distinguible. El protocolo fijado de antemano era razonable.

### 6.5 Decisiones de diseño
- Ramas 2 y 4: ninguna alternativa las mejora.
- Rama 1: tanh, suavizado de etiquetas y *z-score* **sí** la mejoran, confirmado con 3 semillas.
- **Pero el sistema no mejora:** en la copia experimental con la rama 1 nueva, la fusión queda en 0,985 (diferencia −0,000 [−0,007 a +0,006]). Lo que la rama 1 ganó ya lo aportaban las otras.

### 6.6 Qué es el "resultado principal"
El del **protocolo fijo**: `resultados/dl/fusion`, `calibracion`, `incertidumbre` y `tablas`. La búsqueda, el estudio de diseño y `experimental_v2` son análisis complementarios y no lo reemplazan.

---

## 7. Si algo falla

| Síntoma | Qué hacer |
|---|---|
| `Falta el entorno .venv` | Ejecuta primero `scripts\descargar_recursos.bat` |
| `ModuleNotFoundError: comun` (o `temporal`, `fusion`, `informe`) | Con el entorno activo: `pip install -e .` |
| Keras intenta usar TensorFlow | `set KERAS_BACKEND=jax` antes del comando (los `.bat` ya lo hacen) |
| Los notebooks no muestran controles | Ejecuta todas las celdas; `abrir_notebook.bat` instala `ipywidgets` |
| Una prueba falla solo en tu PC | Probablemente es la diferencia de versiones (Python 3.13, Keras 3.15). Copia el mensaje y pásamelo |
| `prueba_e2e` dice `CON DIFERENCIAS` | Los pesos o los datos de tu copia no son los publicados. No sigas: avísame con la salida |
| Un entrenamiento se cortó | Vuelve a lanzar el mismo comando: continúa donde iba |
| Quiero repetir un experimento de verdad | Borra su carpeta en `resultados\dl\` y vuelve a lanzarlo |
| Descarga de NASA con error 403 | Reintenta más tarde; el script es reanudable |

**GPU:** el proyecto corre en CPU. JAX en Windows nativo no usa la tarjeta gráfica (haría falta WSL2), y con modelos de 1.000 a 58.000 parámetros la ganancia sería dudosa.

---

## 8. Usar el sistema con una señal

Con el entorno activo (`.venv\Scripts\activate` y `set KERAS_BACKEND=jax`):

```python
import numpy as np
from fusion.sistema import Sistema

datos = dict(np.load("data/dataset_temporal.npz"))
i = list(datos["id_ejemplo"]).index("K00022.01|transito")
entradas = {c: v[i : i + 1] for c, v in datos.items() if len(v) == len(datos["id_ejemplo"])}

sistema = Sistema("modelos")
respuesta = sistema.responder(entradas, particion=2, tasa_base=0.05)

print(respuesta.p)                 # planeta, falso positivo, sin señal
print(respuesta.q_con_tasa_base)   # P(planeta | hay señal) si solo el 5 % fueran planetas
print(respuesta.entropia)          # 0 = el sistema está seguro
```

- `particion` es la partición de la estrella (columna `particion` de `compartido/ejemplos.csv`). Para una estrella que no está en el conjunto, usa `particion=None`: se promedian los cinco sistemas.
- En este ejemplo no se pasa la imagen, así que la rama 3 se da por ausente. La sección 5 del notebook `03` lo hace con imagen y con controles.
- **Rango de validez:** señales de Kepler de 1 a 30 días.

---

## 9. Qué falta

- Correr `scripts\verificar.bat` en tu PC.
- La rúbrica oficial, y con ella el informe y la presentación.
- Reemplazar `00_ESTADO_Y_DECISIONES.md` en el proyecto de Claude con lo decidido.
