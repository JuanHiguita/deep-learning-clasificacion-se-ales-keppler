# Plan de ejecución del curso de Deep Learning

**Versión:** 1.1 · **Fecha:** 19 de noviembre de 2026 · **Entrega:** domingo 6 de diciembre de 2026 (quedan 17 días)
**Título provisional:** *Clasificación calibrada de señales de tránsito de Kepler con una red multirrama y un codificador por tránsito con atención* (pendiente de confirmación de Juan).
**Reemplaza a:** el calendario y el detalle de `roadmap_curso_deep_learning.md`. El alcance y las reglas del roadmap siguen vigentes; donde este plan dice algo distinto, manda este plan, y por encima de ambos manda `00_ESTADO_Y_DECISIONES.md`.
**Nombre en el repositorio:** `docs/planeacion/04_plan_ejecucion_deep_learning.md` (repositorio del curso: `D:\Proyectos Especializacion\Deep Learning\Proyecto_Final`). La arquitectura del repositorio está en `docs/ARQUITECTURA.md` y cada paquete de trabajo tiene su especificación en `specs/`.

---

## 0. Cómo usar este documento (léelo primero si eres un agente)

1. Lee, en este orden: `00_ESTADO_Y_DECISIONES.md`, este plan, `02_contrato_integracion.md` y `CLAUDE.md`.
2. Abre `docs/planeacion/BITACORA_DL.md` (sección 13) para saber qué paquete de trabajo (PT) está en curso. Trabaja **un PT a la vez**, en el orden de la sección 7.
3. Cada PT es una funcionalidad de Spec Kit y **ya tiene** `specs/<NNN-nombre>/spec.md`, `plan.md` y `tasks.md`. Si la spec y este plan difieren en un detalle, manda la spec (es más reciente y más precisa); si difieren en una regla o en el alcance, detente y pregunta.
4. Un PT está terminado solo cuando pasan **todos** sus criterios de aceptación y `python src/comun/verificar_contrato.py`.
5. **Detente y pregunta a Juan** si ocurre algo de la sección 13.3. No improvises en esos casos.
6. Nunca escribas una cifra en un documento si no salió de un script del repositorio en esa sesión. No inventes datos, cifras ni referencias.

Términos usados aquí, explicados una vez:

- **Partición:** uno de los 5 grupos de estrellas de la validación cruzada. Cada ejemplo se predice con un modelo que no vio su estrella.
- **OOF** (*out-of-fold*): predicción de un ejemplo hecha por el modelo de su partición de prueba.
- **Vector o *embedding*:** resumen numérico de tamaño fijo que entrega cada rama.
- **Sesgo inductivo:** supuesto que una arquitectura impone antes de ver los datos.
- **ECE:** error de calibración esperado, la diferencia promedio entre la confianza declarada y el acierto real.
- **NLL** (*log-loss*): penaliza probabilidades malas, no solo decisiones malas.

---

## 1. Estado al 19 de noviembre y supuestos

**Ya hecho (19-nov):** repositorio creado con la estructura de `docs/ARQUITECTURA.md`, constitución, protocolo fijo (`config/protocolo.json`), especificaciones 000–008 y el núcleo del motor (`comun/protocolo.py`, `metricas.py`, `estadistica.py`) con 17 pruebas en verde.

Según `00_ESTADO_Y_DECISIONES.md` (bitácora hasta el 10 de noviembre), existen la demo (`src/demo_dataset.py`, `src/demo_cnn.py`), la selección de 260 KOIs (`data/seleccion_demo.csv`: 130 confirmados y 130 falsos positivos, una estrella por KOI, periodos de 1,0 a 29,9 d) y la planeación. **No hay registro de base común, dataset temporal ni modelos del curso.** Este plan asume que se empieza desde ahí.

Supuestos que el PT0 debe comprobar antes de seguir:

| Supuesto | Si no se cumple |
|---|---|
| El PC de Juan llega a MAST y al Exoplanet Archive | Descargar desde el navegador o en Colab (`CLAUDE.md`) |
| Los TPF de Q5 de la demo están en `data/tpf/` | Descargarlos de nuevo con `lightkurve` (solo hacen falta para la imagen v0 y la línea base de la demo) |
| Keras 3 + JAX en CPU funciona en el PC | Usar Colab en CPU; no cambiar de framework |
| El curso de Visión no ha empezado (confirmado el 19-nov) | — |

---

## 2. Decisiones

### 2.1 Decisiones de diseño tomadas en este plan

| ID | Decisión | Razón |
|---|---|---|
| DD-01 | **Base del curso:** la selección de la demo (260 KOIs) con etiquetas de `q1_q17_dr25_koi`. No se amplía la muestra. | Quedan 17 días. La muestra ya está descargable y balanceada. |
| DD-02 | **Las ventanas por tránsito y las vistas salen de las curvas PDCSAP de Q1–Q17**, no del TPF de Q5. | La comparación de arquitecturas debe usar la misma información de entrada. Con solo Q5 la rama 2 vería un trimestre y la rama 1, diecisiete. |
| DD-03 | **La pregunta 1 se contesta con un diseño controlado:** las tres variantes comparten el mismo codificador por tránsito y las mismas ventanas; solo cambia cómo se combinan los tránsitos (promedio, GRU o atención). La CNN tipo AstroNet sobre las vistas es la línea base profunda externa. | Así una diferencia se atribuye a la arquitectura y no a los datos. |
| DD-04 | **Normalización por forma, no por SNR recortado.** Vistas y ventanas se dividen por una escala común por ejemplo (sección PT2). La profundidad y el SNR entran por la rama de escalares. | El recorte a ±30σ de la demo satura las señales profundas (error documentado de K00022.01, `00_ESTADO` §5) y borraría la diferencia entre eclipses pares e impares, que es la pista central de la rama 2. Es una desviación del roadmap ("unidades de SNR") y no cambia las formas del contrato. |
| DD-05 | **Validación interna por convención:** para la partición de prueba *k*, la validación interna es la partición *(k+1) mod 5*. Se usa para parada temprana, selección entre variantes y *temperature scaling*. | Es determinista y la pueden usar los dos cursos sin columnas nuevas. Evita que un codificador congelado haya entrenado con las estrellas con que se calibra la fusión. |
| DD-06 | **Hiperparámetros fijados de antemano** (sección 4.4). No hay búsqueda. Cualquier cambio se anota como desviación en la bitácora, con fecha y motivo, y solo puede decidirse mirando la validación interna. | Con unos 500 ejemplos, buscar hiperparámetros produce diferencias que son ruido. |
| DD-07 | **Estrategia (b) de fusión limitada a las ramas propias.** La rama de imagen nunca se ajusta en este curso. | Regla de doble entrega. |
| DD-08 | **El experimento de pooling se hace en la CNN 1D de la rama 1**, no en la CNN 2D. | "Sin pooling" en 2D es tema del curso de Visión. |
| DD-09 | **Los escalares de profundidad y SNR se miden con el extractor en la época del ejemplo**, no se copian del catálogo. | El ejemplo "sin señal" comparte catálogo con su gemelo real; con valores de catálogo la rama 4 no podría distinguirlos y la comparación sería artificial. |
| DD-10 | **La ablación principal reentrena la cabeza de fusión sin la rama.** Apagar la rama con la bandera de disponibilidad es solo un análisis secundario. | Una cabeza entrenada con las cuatro ramas nunca vio un vector en ceros; apagarlo en prueba mide otra cosa. |
| DD-11 | No se comparan resultados con el "modelo combinado" del curso de Visión. La referencia es la demo vuelta a medir sobre la base común. | Doble entrega; además los resultados de Visión llegan el mismo día. |
| DD-12 | **Repositorio propio del curso** en `D:\Proyectos Especializacion\Deep Learning\Proyecto_Final`, separado del repositorio de la tesis y de Visión. Se conectan por archivos: `compartido/` (base común, con SHA-256) y `modelos/imagen/v1/`. | Lo pidió Juan el 19-nov. Cambia el "un solo repositorio" de `03_plan_integrado_cursos.md` §6; la base común sigue siendo una sola. |
| DD-13 | **Entregas independientes.** Para la entrega del curso, este proyecto no depende de nada del curso de Visión: construye su propia base y usa como rama de imagen el **codificador v0** (código de la demo, anterior a los cursos), congelado. No hay punto I2. Los dos proyectos se unen **después** de las entregas, en la tesis. | Lo decidió Juan el 19-nov; los dos profesores aceptaron el proyecto. La regla de doble entrega sigue vigente: la imagen se acredita a la demo y no se estudia ni se reporta como aporte de este curso. |

### 2.2 Decisiones abiertas (el agente usa el valor por defecto y lo deja anotado)

| ID | Pregunta para Juan | Valor por defecto |
|---|---|---|
| DA-1 | ¿Los dos profesores aceptaron el proyecto compartido? | **Resuelta (19-nov): sí aceptaron.** |
| DA-2 | ¿El chat de Visión ya fijó una definición del conjunto prerregistrado o construyó parte de la base? | **Resuelta (19-nov): Visión no ha empezado.** Este curso construye su base con la definición del PT1. |
| DA-3 | ¿Fecha de I2 (codificador de imagen v1)? | **Ya no aplica (DD-13):** la entrega usa v0. |
| DA-4 | ¿Formato de entrega: los cuatro (informe, notebook, presentación, repositorio) o uno? | Los cuatro. |
| DA-5 | ¿Enunciado y rúbrica oficiales? | Juan aún no tiene la rúbrica (19-nov). Cuando llegue, revisar la sección 11 y la spec 008 contra ella. |
| DA-6 | Título definitivo. | El provisional de la cabecera. |

---

## 3. Preguntas, hipótesis operativas y criterios de rechazo

Las hipótesis se fijan **antes** de ver resultados. Todas las diferencias se miden con *bootstrap* pareado por estrella (sección 4.3).

| | Pregunta | Hipótesis operativa | Se rechaza si | Experimentos |
|---|---|---|---|---|
| P1a | ¿La lectura tránsito por tránsito supera a la curva plegada? | El codificador por tránsito (mejor agregador según validación interna) supera en AUC planeta vs FP a la CNN tipo AstroNet. | El intervalo del 95 % de la diferencia incluye el 0. | E1, E2 |
| P1b | ¿La atención aporta frente al promedio? | Atención > promedio en AUC y en NLL. | El intervalo incluye el 0. **Es un resultado esperable en periodo corto y se reporta igual.** | E2, E3, E4 |
| P1c | ¿Importa el orden de los tránsitos? | La GRU no supera a la atención, y barajar el orden en prueba no cambia a promedio ni atención y sí a la GRU. | — (descriptivo) | E2, E3 |
| P2 | ¿Cuánto mejora la fusión? | La fusión de 4 ramas supera a la mejor rama sola y a la demo vuelta a medir; quitar cada rama empeora el AUC. | El intervalo de la diferencia incluye el 0 (por rama, se reporta cuál no aporta). | E6, E7 |
| P3a | ¿Salen calibradas? | ECE y NLL después de *temperature scaling* ≤ antes. | El intervalo de la diferencia incluye el 0 (resultado probable; la demo ya salía calibrada). | E8 |
| P3b | ¿El prior explícito funciona? | Con la corrección por tasa base, el ECE ponderado a la tasa objetivo es menor que sin corregir. | No mejora. | E9 |
| P3c | ¿La incertidumbre sirve? | Los errores tienen más incertidumbre que los aciertos, y el ensamble de semillas es mejor que MC Dropout en NLL y en riesgo-cobertura. | No hay diferencia. | E10 |

La meta "mejora clara en estrellas dobles" del documento de divulgación se trata como hipótesis secundaria: diferencia en el acierto sobre FP con bandera de eclipse secundario, con su intervalo y su *n*.

**Tabla de sesgos inductivos (hilo del informe).** Cada fila se contesta con un experimento:

| Decisión de arquitectura | Supuesto que mete | Experimento que lo prueba |
|---|---|---|
| Plegar la curva (AstroNet) | Todos los tránsitos son iguales; solo importa el promedio | E1 frente a E2 |
| Pesos compartidos entre tránsitos | Cada tránsito se lee con el mismo detector | Viene con el diseño de E2 |
| Promedio o atención | El orden de los tránsitos no importa (Zaheer et al., 2017) | E3 (permutación) |
| Atención | Algunos tránsitos pesan más que otros (Ilse et al., 2018) | E2 (atención vs promedio), E4 |
| GRU | El orden importa; memoria secuencial | E3 |
| Max pooling 1D | Da igual dónde cae exactamente la forma | E5 |
| Fusión por concatenación congelada | Las ramas aportan evidencia que se combina tarde | E6 (a) frente a (c) |

---

## 4. Protocolo experimental (obligatorio para todos los PT)

### 4.1 Datos y particiones

- **Ejemplos:** dos por KOI: `transito` (clase 0 o 1) y `sin_senal` (clase 2, época `t0 + 0,25·P`). Se esperan unos 520 ejemplos; el número exacto lo fija PT1.
- **Particiones:** 5, agrupadas por `kepid` y estratificadas por clase del KOI, con un reparto determinista por SHA-256 de `"0:<kepid>"` (spec 001, RF-8). No se usa `StratifiedGroupKFold`, porque su resultado depende de la versión de scikit-learn. Se guardan en `compartido/ejemplos.csv` y **no se regeneran nunca**.
- **Para la partición de prueba *k*:** entrenamiento = las 3 particiones restantes; validación interna = partición *(k+1) mod 5* (DD-05); prueba = *k*.
- **Prerregistrados:** fuera de todo (entrenamiento, validación, prueba, EDA de resultados).

### 4.2 Semillas

- Semillas 0, 1 y 2 para cada configuración de las ramas temporales (15 modelos por configuración).
- Los codificadores congelados v1 que van a la fusión son los de la **semilla 0**.
- `keras.utils.set_random_seed(semilla)` al crear cada modelo; la semilla del muestreo de aumentación deriva de `(semilla, partición)`.

### 4.3 Métricas y estadística

- **Principal:** AUC planeta vs FP, solo en ejemplos `transito`, con `q = p0 / (p0 + p1)`.
- **Secundarias:** acierto y matriz de confusión de 3 clases; NLL y Brier de 3 clases; ECE *top-label* con 15 intervalos de igual ancho (Guo et al., 2017) y ECE binario sobre `q`; acierto en FP por bandera (`fp_ss`, `fp_co`, `fp_nt`); AUC por terciles de MES y por número de tránsitos válidos. Los terciles se calculan sobre `ejemplos.csv` **antes** de entrenar y se guardan en el manifiesto.
- **Intervalos:** *bootstrap* por estrella, 2.000 remuestreos, percentiles 2,5 y 97,5, semilla fija. Se remuestrean estrellas, no ejemplos.
- **Diferencias entre modelos:** *bootstrap* pareado. En cada remuestreo se calcula `Δ = media_semillas(métrica A) − media_semillas(métrica B)` sobre las mismas estrellas.
- **Variabilidad por semilla:** se reporta aparte (media y desviación entre las 3 semillas).
- Todo estrato con menos de 20 ejemplos se reporta con su *n* y sin conclusiones.
- La cifra de ECE de la demo (0,022) **no es comparable** hasta volver a medirla con esta definición: el script de calibración de la demo no está en el proyecto.

### 4.4 Entrenamiento (valores fijos, DD-06)

| Parámetro | Valor |
|---|---|
| Optimizador | AdamW, tasa 1e-3, *weight decay* 1e-4 |
| Lote | 32 |
| Épocas máximas | 150 |
| Parada temprana | NLL de validación interna, paciencia 15, se restauran los mejores pesos |
| Pérdida | Entropía cruzada de 3 clases con pesos por clase (inverso de la frecuencia en entrenamiento) |
| Dropout | 0,3 antes del vector de cada rama (0,2 en escalares) |
| Estandarización | Solo con estadísticas del entrenamiento de cada partición; se guardan junto al codificador |

### 4.5 Pruebas de cordura (se corren una vez por rama, antes de los experimentos)

1. **Etiquetas barajadas:** entrenar con etiquetas permutadas debe dar AUC ≈ 0,5. Si no, hay fuga.
2. **Sobreajuste de un lote:** el modelo debe memorizar 32 ejemplos. Si no, hay un error de implementación.
3. **Solapamiento de estrellas:** ningún `kepid` en dos de {entrenamiento, validación, prueba}.
4. **AUC > 0,99 en cualquier rama:** detenerse y auditar fugas antes de continuar.

---

## 5. Estructura del repositorio y convenciones

```
compartido/                 ejemplos.csv, manifiesto_base.json           (versionados)
specs/NNN-nombre/           spec.md, plan.md, tasks.md                   (Spec Kit)
src/comun/                  descarga, particiones, entrenamiento, métricas, estadística, verificación
src/temporal/               curvas, dataset temporal, modelos 1D, experimentos de las ramas
src/fusion/                 imagen v0, fusión, ablación, calibración, incertidumbre
modelos/<rama>/v<N>/        codificadores por partición, vectores, manifiesto
resultados/dl/<experimento>/  predicciones OOF, métricas JSON, figuras
notebooks/                  eda_temporal.ipynb, informe_dl.ipynb
tests/                      pruebas con curvas sintéticas (solo para probar código)
docs/planeacion/            este plan y BITACORA_DL.md
data/                       FITS y .npz grandes (fuera de git)
```

Convenciones:

- Scripts ejecutables con `argparse`; **todas las rutas por argumento**, ninguna absoluta. Español en nombres, comentarios y documentos.
- Cada script de experimento escribe `manifiesto.json` con: commit, fecha, argumentos, semillas, versiones de librerías y *hash* SHA-256 de sus archivos de entrada.
- Un notebook nunca entrena: solo lee `resultados/dl/` y dibuja.
- Las curvas sintéticas existen **solo en `tests/`**. Nunca entran a un resultado.
- Mensajes de commit: `PT<n>: <qué>`. Etiquetas git: `base-v1`, `ramas-v1`, `curso-dl-v1`.

Revisión de seguridad que aplica a todos los PT:

- Descargas solo por HTTPS desde `exoplanetarchive.ipac.caltech.edu`, `mast.stsci.edu` y `archive.stsci.edu`. `kepid` se valida como entero antes de construir una consulta o una ruta.
- `np.load(..., allow_pickle=False)`; los identificadores se guardan como arreglos Unicode, no como objetos. `keras.saving.load_model(..., safe_mode=True)`.
- Sin `shell=True`, sin `eval`, sin credenciales en el repositorio. Código sobre archivos descargados: `python -I`.
- Versiones fijadas: generar `requirements.lock.txt` con `pip freeze` en PT0.

---

## 6. Paquetes de trabajo

Cada PT indica el modelo sugerido: **Opus** para diseño delicado y auditorías; **Sonnet** para implementar contra una especificación cerrada.

### PT0 · Preparación del entorno (Sonnet · 0,5 día)

**Tareas**

- [ ] T0.1 Crear entorno, instalar `requirements.txt`, generar `requirements.lock.txt`. Comprobar `KERAS_BACKEND=jax` con un modelo de juguete.
- [ ] T0.2 Comprobar acceso: una consulta TAP de 1 fila y `lightkurve.search_lightcurve("KIC 7440748")`.
- [ ] T0.3 Comprobar si existen `data/tpf/demo_kplr*.fits.gz` y `compartido/ejemplos.csv`.
- [ ] T0.4 Crear carpetas de la sección 5, `pytest` mínimo y `docs/planeacion/BITACORA_DL.md`.
- [ ] T0.5 **Lanzar ya la descarga de PT2 (T2.1)** en segundo plano: es lo más lento.

**Aceptación:** `pytest` corre; la bitácora registra el resultado de T0.2 y T0.3.

### PT1 · Base común (`specs/001-base-comun`) (Sonnet implementa, Opus revisa · 1 día)

**Objetivo:** tabla maestra, particiones y líneas base tabulares. Es infraestructura compartida con Visión: si ya existe, se verifica y se reutiliza.

**Salidas:** `compartido/ejemplos.csv`, `compartido/manifiesto_base.json`, `data/koi_dr25.csv`, `resultados/dl/lineas_base_tabulares/metricas.json`.

**Especificación**

1. `src/comun/descargar_koi.py`: consulta TAP a `q1_q17_dr25_koi` con columnas explícitas (`kepid, kepoi_name, koi_disposition, koi_pdisposition, koi_score, koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec, koi_period, koi_time0bk, koi_duration, koi_depth, koi_model_snr, koi_max_mult_ev, koi_num_transits, koi_kepmag, koi_dikco_msky, koi_dicco_msky`). Si una columna no existe en esa tabla, el script falla con un mensaje claro y se anota (no se sustituye en silencio). Guarda consulta, fecha y SHA-256.
2. `src/comun/construir_ejemplos.py`:
   - Cruza `data/seleccion_demo.csv` con DR25 por `kepoi_name`. Entran los KOIs con `koi_disposition` ∈ {CONFIRMED, FALSE POSITIVE} **en DR25**. Reporta cuántos cambian de etiqueta o salen respecto a la demo.
   - Columnas del contrato §3, más columnas de **metadatos solo para reportar**: `fp_nt, fp_ss, fp_co, fp_ec, n_koi_en_estrella`. `tipo_fp`: el nombre de la bandera si hay exactamente una (`ec` → `otro`); `otro` si hay varias.
   - `particion` según 4.1.
   - `prerregistrado` (definición por defecto, DA-2): verdadero si `koi_disposition` difiere entre `q1_q17_dr25_koi` y `cumulative` en la fecha de descarga. Es una aproximación operativa a "cambió de disposición por seguimiento posterior"; la definición de la tesis sigue pendiente (`00_ESTADO` §7). Reportar cuántos ejemplos marca. Si marca más del 10 % de la selección, detenerse.
3. `src/comun/lineas_base_tabulares.py`: S3-B y S3-C con `HistGradientBoostingClassifier` (maneja NaN de forma nativa: no rellenar), en dos modalidades: (i) entrenado solo con la selección; (ii) entrenado con todos los KOIs etiquetados de DR25 menos las estrellas de prueba y los prerregistrados. Además, AUC de `koi_score` como **comparador externo**. Mismas particiones.
4. `src/comun/verificar_contrato.py`: las comprobaciones del contrato §8, más: lista blanca de columnas permitidas en `escalares`; validación interna *(k+1) mod 5* sin solapamiento de estrellas.

**Aceptación**

- [ ] `verificar_contrato.py` pasa sobre `ejemplos.csv`.
- [ ] El manifiesto permite regenerar la tabla byte a byte.
- [ ] Clases balanceadas por partición (diferencia máxima de 3 ejemplos entre particiones).
- [ ] Etiqueta git `base-v1` y aviso a Visión (anexo B).

### PT2 · Dataset temporal y EDA (`specs/002-dataset-temporal`) (Opus diseña y revisa, Sonnet implementa · 2 días)

**Objetivo:** `data/dataset_temporal.npz` con las claves del contrato §4.

**Especificación**

1. **T2.1 Descarga** (`src/temporal/descargar_llc.py`): curvas de cadencia larga de Q1–Q17, `author="Kepler"`, flujo `pdcsap_flux`, `quality_bitmask="default"`. Reanudable, con reintentos y espera creciente. Escribe `data/llc/manifiesto_llc.csv` (kepid, trimestre, archivo, tamaño, SHA-256). Estrellas con menos de 4 trimestres: se registran; su ejemplo queda con la bandera de disponibilidad en 0 para las ramas 1 y 2.
2. **Limpieza por trimestre** (`src/comun/curvas.py`): quitar NaN, normalizar por la mediana, aplanar con Savitzky–Golay (`window_length=401`, como la demo) con **máscara de tránsito de todos los KOIs de la estrella** (1,5 duraciones), recortar atípicos solo hacia arriba (4σ). Ruido `σ` = 1,4826·MAD fuera de tránsito, por trimestre.
3. **Escala por ejemplo** (DD-04): `escala = max(profundidad_medida, 3·σ_local)`, donde `profundidad_medida` es menos la mediana del flujo aplanado dentro de ±0,25 duraciones de la época del ejemplo, y `σ_local` es el error de esa mediana. Todo se expresa como `(flujo − 1) / escala` y se recorta a [−5, 5]. Así un tránsito real vale cerca de −1 y un ejemplo sin señal no se infla.
4. **Rama 1 · vistas** (Shallue y Vanderburg, 2018):
   - `vista_global` (n, 2001, 1): órbita completa plegada, mediana por intervalo.
   - `vista_local` (n, 201, 4): ±4 duraciones; canales: todos los tránsitos, pares, impares y fase 0,5. Los cuatro con la **misma** escala.
   - Intervalos vacíos: interpolación lineal; si falta más de un tercio, el ejemplo se marca no disponible.
5. **Rama 2 · tránsitos:**
   - `transitos` (n, 32, 61): por tránsito, ±2,5 duraciones interpoladas linealmente a 61 puntos, misma escala.
   - Un tránsito es **válido** si tiene ≥ 70 % de las cadencias esperadas en la ventana, ≥ 2 cadencias dentro de ±0,5 duraciones y no se solapa con el tránsito de otro KOI de la estrella.
   - **Selección de 32:** hasta 16 pares y 16 impares, espaciados uniformemente en el tiempo dentro de cada paridad, y luego en orden cronológico. Determinista. `mascara_transitos` (n, 32).
   - Claves internas extra (no cambian el contrato): `paridad_transitos` (n, 32), `n_transitos_validos` (n,), `consistencia` (n, 4) = [dispersión de la profundidad por tránsito, diferencia par−impar en unidades de su error, log10 del número de tránsitos válidos, mayor desviación de un tránsito respecto a la mediana].
6. **Rama 4 · escalares** (DD-09), `d = 7`: log10 del periodo, log10 de la duración, profundidad medida (ppm, con signo), SNR medido de la señal plegada, log10 de tránsitos válidos, `koi_dikco_msky`, `koi_dicco_msky`. `faltantes` (n, 7). En `sin_senal` el centroide va como faltante: el catálogo lo midió en la época real. El informe declara este atajo; no afecta a la métrica principal, que solo usa ejemplos `transito`.
7. **Clase `sin_senal`:** el mismo proceso con `t0 + 0,25·P`. Se cuenta y se reporta cuántos tienen SNR medido > 5 (posible eclipse secundario excéntrico). **No se eliminan.**
8. `id_ejemplo` en cada `.npz`. `src/temporal/construir_dataset_temporal.py <dir_llc> <ejemplos.csv> <koi_dr25.csv> <salida.npz>`.

**EDA** (`notebooks/eda_temporal.ipynb`, solo sobre ejemplos no prerregistrados): tránsitos válidos por clase; diferencia par−impar por bandera de FP; señal en fase 0,5 por bandera; 12 ejemplos al azar con semilla 7, con sus tres representaciones.

**Pruebas** (`tests/`, curvas sintéticas): una caja de profundidad conocida se recupera con error < 10 %; una alternancia par/impar inyectada aparece en `consistencia`; un hueco en los datos invalida el tránsito; `sin_senal` no contiene la caja.

**Aceptación**

- [ ] Formas del contrato; `verificar_contrato.py` pasa.
- [ ] Ninguna columna con fuga; no hay NaN en ningún arreglo.
- [ ] K00022.01 y K03615.01 se inspeccionan a ojo: el primero con tránsitos iguales, el segundo con alternancia (`00_ESTADO` §5). Si no se ve, el extractor está mal.
- [ ] Respaldo si la descarga no termina el domingo 22: construir todo desde el TPF de Q5 (curva de apertura, como la demo), declarar la limitación y seguir.

### PT3 · Motor de entrenamiento, métricas y estadística (`specs/003-motor-experimentos`) (Opus · 1 día)

**Salidas:** `src/comun/entrenamiento.py`, `metricas.py`, `estadistica.py`, con pruebas unitarias.

**Especificación**

- `entrenar_cv(constructor_modelo, entradas, config, semillas) → resultados/dl/<exp>/`: aplica 4.1–4.4. Guarda, por semilla: `pred_semilla<s>.npz` (`id_ejemplo`, `particion`, `p_prueba` (n, 3), `p_validacion` (n, 3)), historial de pérdida, número de parámetros y tiempo por época.
- Exportación al contrato §5: `modelos/<rama>/v1/particion_k.keras` (hasta la capa `embedding`), `vectores_particion_k.npz` para **todos** los ejemplos y `manifiesto.json`.
- `metricas.py`: sección 4.3. `estadistica.py`: *bootstrap* por estrella, simple y pareado.
- Pruebas: AUC contra `sklearn`; ECE de un caso calculado a mano; el *bootstrap* de dos modelos idénticos da Δ = 0; las 4 pruebas de cordura (4.5) como funciones reutilizables.

**Aceptación:** pruebas en verde; una corrida completa de un modelo trivial (regresión logística sobre escalares) produce todos los archivos.

### PT4 · Ramas y comparación de arquitecturas (`specs/004-ramas-temporales`) (Sonnet implementa, Opus audita · 3 días)

**Modelos** (`src/temporal/modelos.py`; todos terminan en `Dense(…, name="embedding")` y una cabeza `Dense(3, softmax)` propia):

| ID | Modelo | Entrada | Arquitectura |
|---|---|---|---|
| M0 | Demo 1D, vuelta a medir | `seq` de la demo (61, 2) | La rama 1D de `demo_cnn.py`, sin cambios, con las particiones comunes (`src/temporal/linea_base_demo.py`). También el modelo completo de la demo, **como referencia de base común**. |
| M1 | CNN tipo AstroNet reducida (rama 1) | vistas | Global: 3 bloques [Conv1D(16/32/64, 5) ×2 + MaxPool(5, 2)] + GlobalMaxPool. Local: 2 bloques [Conv1D(16/32, 5) ×2 + MaxPool(7, 2)] + GlobalMaxPool. Concatenar → Dense(64) → Dropout → Dense(32). |
| M1r | BiGRU sobre la vista local | `vista_local` | BiGRU(32) → Dense(32). **Prioridad B.** |
| φ | Codificador por tránsito (compartido) | (61, 1) | Conv1D(16, 5) → Conv1D(32, 5) → GlobalMaxPool → Dense(32), con `TimeDistributed`. |
| M2p | φ + promedio | tránsitos + máscara | Media enmascarada → concat `consistencia` → Dense(32). |
| M2g | φ + GRU | ídem | BiGRU(32) con máscara, orden cronológico → concat → Dense(32). |
| M2a | φ + atención | ídem | 1 bloque [MultiHeadAttention(2 cabezas, key_dim 16, con máscara) + residual + LayerNorm + Dense(64) + residual] → *attention pooling* con consulta aprendida y softmax enmascarado (se guardan los pesos) → concat → Dense(32). Sin codificación posicional. |
| M4 | Escalares (rama 4) | escalares + faltantes | Faltantes a 0 después de estandarizar, concatenados con su aviso → Dense(32) → Dropout(0,2) → Dense(16). |

**Aumentación** (solo en entrenamiento): ruido gaussiano de 0,05; desplazamiento de ±2 puntos en vistas y ventanas; en la rama 2, conservar cada tránsito con probabilidad 0,8 (mínimo 4).

**Experimentos**

| ID | Qué | Prioridad |
|---|---|---|
| E0 | Pruebas de cordura (4.5) para M1, M2a y M4 | A |
| E1 | M0, M1 y M4: 3 semillas × 5 particiones | A |
| E2 | M2p, M2g y M2a: 3 semillas × 5 particiones | A |
| E3 | Permutación del orden de los tránsitos en prueba (10 permutaciones) para M2p, M2g y M2a | A |
| E4 | Menos tránsitos en prueba: conservar 32, 16, 8, 4 y 2 al azar con semilla; AUC frente a número de tránsitos | B |
| E5 | Pooling en la torre local de M1: máximo, promedio y convolución con paso 2. Además, robustez a desplazar la vista 0, 2, 5 y 10 puntos en prueba | B |
| E2b | M2a sin `consistencia` | B |
| E1b | M1r | B |

**Selección del agregador de la rama 2 v1:** el de menor NLL medio de **validación interna** (5 particiones × 3 semillas). Si la diferencia entre el mejor y otro más simple es menor que un error estándar, gana el más simple (promedio < atención < GRU). Nunca se mira la prueba para elegir.

**Advertencia para el informe:** los pesos de atención se muestran como descripción, en casos elegidos al azar; no son una explicación causal (Jain y Wallace, 2019). E4 simula pocos tránsitos con señales de periodo corto; **no** permite concluir nada sobre el régimen de zona habitable.

**Aceptación**

- [ ] E0 pasa. E1–E3 completos con sus manifiestos.
- [ ] Tabla comparativa con AUC, NLL, ECE, parámetros y tiempo por época, con intervalos.
- [ ] Codificadores v1 (semilla 0) de `vistas`, `transitos` y `escalares` exportados; `verificar_contrato.py` pasa; etiqueta `ramas-v1`.

### PT5 · Fusión y ablación (`specs/005-fusion`) (Sonnet implementa, Opus audita · 2 días)

**Imagen v0** (`src/fusion/codificador_imagen_v0.py`): la rama 2D de `demo_cnn.py` hasta su Dense(32), entrenada por partición con el protocolo 4.1 sobre la imagen que genera `demo_dataset.py` desde los TPF de Q5. Su manifiesto dice: "v0, código de la demo (base común), no es resultado de ningún curso". **La entrega del curso se hace con v0 (DD-13).** Cambiarla por el codificador de Visión es trabajo de la unión posterior: solo cambian los vectores de entrada.

**Experimentos**

| ID | Qué | Prioridad |
|---|---|---|
| E6a | Fusión del contrato §6: concatenación (116) → Dense(64) → Dropout(0,3) → Dense(3), sobre vectores congelados; 5 semillas de cabeza | A |
| E6c | Fusión tardía: promedio de las probabilidades de las 4 cabezas individuales | A |
| E7 | Ablación (DD-10): cabeza reentrenada sin cada rama (4 corridas) y con cada rama sola (4 corridas) | A |
| E7b | Ablación con bandera de disponibilidad, entrenando la cabeza con *dropout* de ramas (p = 0,15) | C |
| E6b | Ajuste fino de las ramas 1, 2 y 4 con tasa 1e-4; la imagen sigue congelada (DD-07) | C |

**Riesgo conocido del apilado:** la cabeza se entrena con vectores de ejemplos que los codificadores ya vieron, y se evalúa con vectores de ejemplos nuevos. **Diagnóstico obligatorio:** comparar el NLL de la cabeza en entrenamiento y en validación interna. Si la validación es más de 2 veces peor, reportarlo y entrenar la cabeza sobre los vectores de la validación interna de las otras particiones (ajuste cruzado), anotándolo como desviación.

**Reporte:** tabla de la sección 3 (P2); ablación por bandera de FP y por tercil de MES; comparación con M0-completo, S3-B, S3-C y `koi_score` (comparador externo). Qué versión de la imagen se usó.

**Aceptación:** E6a, E6c y E7 completos; diagnóstico del apilado en el informe; la sección de arquitectura acredita la rama 3 a la demo (v0).

### PT6 · Calibración e incertidumbre (`specs/006-calibracion`) (Opus · 2 días)

**Experimentos** (todos sobre las predicciones de E6a; no hay entrenamiento nuevo salvo cabezas):

| ID | Qué | Prioridad |
|---|---|---|
| E8 | Diagrama de confiabilidad, ECE, NLL y Brier antes y después de *temperature scaling*. La temperatura se ajusta minimizando el NLL en la **validación interna** de cada partición y se aplica a su prueba | A |
| E9 | Prior explícito (Saerens et al., 2002): `q' = q·r / (q·r + 1 − q)`, con `r = [π/(1−π)] / [π_ent/(1−π_ent)]`. Se evalúa ponderando los ejemplos de prueba para emular la tasa `π`. Se reporta el ECE ponderado con y sin corrección y cuánto cambia el orden de los candidatos (τ de Kendall y cambios en los 20 primeros) | A |
| E10 | MC Dropout (50 pasadas con el Dropout de la cabeza activo) frente a un ensamble de 5 semillas de cabeza. Métricas: NLL, Brier, ECE, AUROC de detección de errores con la entropía predictiva y curva riesgo-cobertura con su área (Geifman y El-Yaniv, 2017) | A |
| E10b | Ensamble con 3 semillas de codificadores propios (cada uno con su cabeza) | C |
| E11 | Análisis de errores: 10 casos al azar (semilla 7) entre los mal clasificados y 10 entre los bien clasificados, con sus entradas y los pesos de atención | A |

**Tasas base para E9** (se calculan con código desde `data/koi_dr25.csv`, nunca a mano): (i) la proporción confirmados / (confirmados + FP) de DR25 en periodos de 1–30 d; (ii) una rejilla ilustrativa π ∈ {0,5; 0,2; 0,1; 0,05; 0,02}. La cifra de 1,8 % de `00_ESTADO` §5 **no se usa como prior**: su denominador incluye candidatos sin resolver.

**Limitaciones que el informe declara:** la corrección supone que solo cambia la tasa base y no el aspecto de las señales, lo cual no se cumple entre periodo corto y zona habitable; E9 demuestra el mecanismo, no su validez en ese régimen. La incertidumbre de E10 solo cubre la cabeza, porque los codificadores están congelados.

**Aceptación:** E8–E11 completos; la temperatura nunca se ajustó con datos de prueba (lo comprueba una prueba unitaria); figuras de confiabilidad y riesgo-cobertura generadas por script.

### PT7 · Consolidación de resultados (`specs/007-reporte`) (Sonnet · 1 día)

- [ ] `src/fusion/consolidar_resultados.py` lee `resultados/dl/*` y escribe `resultados/dl/tablas/*.csv` y las figuras finales. Es el único origen de cifras del informe.
- [ ] `notebooks/informe_dl.ipynb`: reproduce todas las tablas y figuras desde `resultados/dl/` en menos de 5 minutos, sin entrenar.
- [ ] Auditoría final (Opus): lista de la sección 12; `verificar_contrato.py`; búsqueda de cifras en el informe que no estén en las tablas.

### PT8 · Informe, presentación y repositorio (Opus redacta, Sonnet da formato · 3 días)

Estructura en la sección 11. Antes de entregar: comprobar las referencias con la habilidad `citation-check`.

---

## 7. Calendario

| Bloque | Fechas | Paquetes | Hito de cierre |
|---|---|---|---|
| A | vie 20 – dom 22 nov | PT0, PT1, PT2 (la descarga arranca el viernes) | **D1–D3:** `base-v1` y `dataset_temporal.npz` verificados (dom 22) |
| B | lun 23 – jue 26 nov | PT3 (lun), PT4: E0–E3 (mar–jue); imagen v0 en paralelo (obligatoria) | **D4:** tabla comparativa y `ramas-v1` (jue 26) |
| C | vie 27 – dom 29 nov | PT5 con imagen v0: E6a, E6c, E7. PT4 prioridad B: E4, E5 | **D5:** fusión y ablación (dom 29) |
| D | lun 30 nov – mié 2 dic | PT6: E8–E11. PT7 | **D6:** resultados congelados (mié 2). **Desde aquí no se entrena nada.** |
| E | jue 3 – sáb 5 dic | PT8 | **D7:** etiqueta `curso-dl-v1` y entrega subida (sáb 5) |
| — | dom 6 dic | Colchón | Fecha límite |

Regla del calendario: si un bloque no cierra en su fecha, **se recorta alcance (sección 9), no se corre el hito D6**.

## 8. Presupuesto de cómputo (estimado; medir en PT3 y corregir)

| Grupo | Corridas |
|---|---|
| E1 (M0, M1, M4) | 3 × 15 = 45 |
| E2 (M2p, M2g, M2a) | 45 |
| Imagen v0 | 5 |
| Prioridad B (E5: 3 variantes; E2b; E1b) | 5 × 15 = 75 |
| Cabezas de fusión (E6, E7, E10) | unas 70, de segundos cada una |

Son modelos de pocos miles de parámetros sobre unos 310 ejemplos de entrenamiento. PT3 mide el tiempo real de una corrida; si las 95 corridas de prioridad A no caben en una noche de CPU, se baja a 2 semillas y se anota.

## 9. Orden de recortes (si hay retraso, se corta de arriba hacia abajo)

1. E6b, E7b y E10b (prioridad C).
2. E1b (BiGRU sobre vistas) y E2b.
3. E5 (pooling) y la fila correspondiente de la tabla de sesgos.
4. E4 (menos tránsitos).
5. Modalidad (ii) de las líneas base tabulares.
6. Semillas: de 3 a 2.
7. Vista global de 2001 puntos: reducir a 401 (anotar la desviación del contrato y avisar a Visión; no les afecta porque el vector sigue siendo de 32).

**No se recorta nunca:** particiones por estrella, pruebas de cordura, control atención vs promedio (E2), E3, ablación E7, E8–E10, intervalos por *bootstrap* y la acreditación del origen de la rama de imagen.

## 10. Riesgos

| Riesgo | Señal | Respuesta |
|---|---|---|
| La descarga de curvas falla o es lenta | Menos del 80 % el sábado 21 | Colab; respaldo con TPF de Q5 (PT2) |
| Al unir los proyectos, las bases de los dos cursos no coinciden | Visión construye su propia `ejemplos.csv` | Ofrecerle esta base (anexo B). Si usa otra, la unión reentrena los codificadores sobre una base única; las cifras de los dos informes no se comparan entre sí |
| No hay TPF de Q5 para la imagen v0 | Falla T0.7 | Descargarlos con `lightkurve` (260 archivos). Si no se puede, la fusión se entrega con 3 ramas y se declara |
| Diferencias dentro del ruido | Intervalos que incluyen el 0 | Es un resultado; se reporta así. No se buscan hiperparámetros para "ganar" |
| Fuga | AUC > 0,99 o prueba de etiquetas barajadas ≠ 0,5 | Detener y auditar |
| Sobreajuste del apilado | NLL de validación ≫ entrenamiento en la cabeza | Ajuste cruzado (PT5) |
| JAX no instala en Windows | Falla T0.1 | WSL o Colab en CPU |

---

## 11. Entregables y estructura del informe

**Entregables:** informe (PDF), `notebooks/informe_dl.ipynb`, presentación (10–12 diapositivas) y repositorio con la etiqueta `curso-dl-v1` y un `README` con los comandos para reproducir.

**Informe (12–15 páginas):**

1. Resumen, con las tres respuestas y sus intervalos.
2. Problema y trabajo previo: AstroNet, ExoMiner, Robovetter. La novedad se redacta como "no encontramos".
3. Datos y base. Declarar que el problema y los datos son los mismos del proyecto del curso de Visión, y que las entregas son independientes.
4. Arquitectura del sistema: qué ramas son de este curso y que la de imagen es el codificador v0 de la demo, congelado.
5. Protocolo experimental (sección 4 de este plan).
6. P1: comparación de arquitecturas y sesgos inductivos (E1–E5).
7. P2: fusión y ablación (E6–E7).
8. P3: calibración, prior explícito e incertidumbre (E8–E10).
9. Análisis de errores (E11), con casos al azar.
10. Resultados negativos y limitaciones: periodo corto; solo KOIs reales; clase `sin_senal` construida; atajo del centroide faltante; apilado; la incertidumbre solo cubre la cabeza; versión de la imagen.
11. Conexión con la tesis (fases 3, 4b, 5 y 6) y conclusiones.
12. Declaración de uso de IA y de trabajo compartido entre cursos.

Vocabulario obligatorio: "P(en zona habitable y rocoso)", nunca "P(habitable)"; nada sobre atmósferas; "clasificación" o "revisión" de señales, no "detección de exoplanetas".

## 12. Definición de terminado (auditoría final)

- [ ] Ninguna estrella en entrenamiento y prueba, incluidos los codificadores congelados y la imagen.
- [ ] Ninguna columna con fuga en ninguna entrada (`koi_score`, `koi_fpflag_*`, `koi_pdisposition`). Las banderas solo aparecen como estratos de reporte.
- [ ] Temperatura y selección de variantes hechas con la validación interna, nunca con la prueba.
- [ ] El conjunto prerregistrado no se usó.
- [ ] Toda cifra del informe está en `resultados/dl/tablas/` y se regenera con un comando.
- [ ] Casos mostrados elegidos al azar con semilla fija; se muestran errores.
- [ ] Toda comparación trae su intervalo; los resultados negativos están escritos con el mismo detalle.
- [ ] El informe acredita la rama 3 y no reporta resultados de Visión como propios.
- [ ] Desviaciones del plan listadas en un anexo del informe (desde la bitácora).
- [ ] Referencias verificadas.

---

## 13. Protocolo de sesión para agentes

### 13.1 Bitácora (`docs/planeacion/BITACORA_DL.md`)

Una tabla de estado por PT (pendiente, en curso, terminado, bloqueado) y, debajo, una entrada por sesión con: fecha, modelo, PT, qué se hizo, comandos corridos, archivos creados, desviaciones del plan y siguiente paso exacto. **Se actualiza antes de cerrar cada sesión.** Es la única memoria entre sesiones.

### 13.2 Instrucción de arranque (copiar al iniciar una sesión)

> Trabajas en el proyecto del curso de Deep Learning. Lee `00_ESTADO_Y_DECISIONES.md`, `docs/planeacion/04_plan_ejecucion_deep_learning.md`, `02_contrato_integracion.md`, `CLAUDE.md` y `docs/planeacion/BITACORA_DL.md`. Continúa el PT que la bitácora marca como en curso, o empieza el siguiente pendiente según la sección 7. Sigue Spec Kit (spec, plan, tareas, implementación) dentro de `specs/`. Respeta el protocolo de la sección 4 sin cambios. No amplíes el alcance. Al terminar, corre las pruebas y `verificar_contrato.py`, actualiza la bitácora y dime qué criterios de aceptación pasaron y cuáles no, con las cifras tal como salieron del script.

### 13.3 Cuándo detenerse y preguntar

- Hace falta cambiar el contrato, las particiones o la base después de `base-v1`.
- Una prueba de cordura falla o aparece un AUC > 0,99.
- Faltan datos, una columna de la tabla no existe o un archivo descargado no coincide con su *hash*.
- Para avanzar habría que mirar la prueba (elegir un modelo, un umbral o un hiperparámetro).
- Se necesitaría cualquier cosa del curso de Visión (las entregas son independientes).
- El prerregistro marca más del 10 % de la selección.
- Una cifra del informe no se puede regenerar.

### 13.4 Lo que un agente no hace

No rellena NaN con 0 sin su aviso. No cambia hiperparámetros para mejorar la prueba. No borra ejemplos "raros" sin regla escrita antes. No escribe conclusiones sobre zona habitable. No reporta la CNN de imagen como resultado propio. No usa PyTorch. No sube datos grandes a git.

---

## Anexo A · Texto propuesto para `00_ESTADO_Y_DECISIONES.md`

**Añadir al final de la sección 4:**

> - **Calendario real (actualizado el 19-nov-2026):** entrega de los dos cursos el 6 de diciembre de 2026. El plan de 9 semanas se comprimió. Para Deep Learning manda `04_plan_ejecucion_deep_learning.md`: base `base-v1` el 22-nov, ramas propias `ramas-v1` el 26-nov, fusión el 29-nov, resultados congelados el 2-dic.
> - **Entregas independientes (19-nov-2026):** para las entregas de los cursos, los dos proyectos son independientes y se unen después, en la tesis. Deep Learning usa como rama de imagen el codificador v0 de la demo; ya no existe el punto I2 ni la dependencia de Deep Learning respecto a Visión. Los dos profesores aceptaron el proyecto. Esto reemplaza el calendario común y la sección de dependencias de `03_plan_integrado_cursos.md`. Deep Learning trabaja en un repositorio propio (`D:\Proyectos Especializacion\Deep Learning\Proyecto_Final`).
> - **Decisiones del curso de Deep Learning (19-nov-2026):** base = selección de la demo (260 KOIs) con etiquetas DR25; vistas y ventanas por tránsito desde PDCSAP Q1–Q17; normalización por forma con escala común por ejemplo (no SNR recortado); validación interna de la partición *k* = partición *(k+1) mod 5*; hiperparámetros fijados de antemano; la pregunta 1 se contesta con un codificador por tránsito compartido y tres agregadores (promedio, GRU, atención).

**Añadir a la sección 9, nivel 4:** `04_plan_ejecucion_deep_learning.md` (en el repositorio, `docs/planeacion/04_plan_ejecucion_deep_learning.md`); prevalece sobre `roadmap_curso_deep_learning.md` en calendario y detalle.

**Añadir a la sección 10:**

> - 2026-11-19: plan de ejecución del curso de Deep Learning (`04_plan_ejecucion_deep_learning.md`). Los profesores aceptaron; entregas independientes. Pendiente: rúbrica oficial y definición final del conjunto prerregistrado.

## Anexo B · Lo que debe saber el chat de Visión

1. **Entregas independientes (decisión de Juan, 19-nov).** Cada curso entrega su proyecto sin depender del otro. Deep Learning no espera el codificador de imagen: usa el v0 de la demo. No hay punto I2 ni calendario común. Los dos profesores aceptaron.
2. **Base disponible, no obligatoria.** Deep Learning construye `compartido/ejemplos.csv` (selección de la demo, etiquetas DR25, 5 particiones por `kepid`, prerregistro) en su repositorio `D:\Proyectos Especializacion\Deep Learning\Proyecto_Final`. Visión puede copiarla tal cual, y conviene: así la unión posterior no exige reentrenar. Si Visión construye otra, no pasa nada para las entregas.
3. **Para que la unión sea barata**, conviene que Visión respete el contrato v1.0 (vector de 32 con capa `embedding`, un codificador por partición, `id_ejemplo`) y la convención de validación interna *(k+1) mod 5*.
4. **Doble entrega:** Deep Learning no hace experimentos sobre la CNN 2D ni reporta su desempeño como aporte propio; la usa congelada y la acredita a la demo. Visión no reporta fusión de 4 ramas, calibración ni comparación de arquitecturas temporales.
5. La unión (codificador de Visión dentro de la fusión, base única, reentrenamiento) es trabajo de la tesis, después del 6 de diciembre.

## Anexo C · Referencias (verificar con `citation-check` antes de entregar)

- Shallue, C. J. y Vanderburg, A. (2018). *Identifying Exoplanets with Deep Learning.* AJ 155, 94.
- Valizadegan, H. et al. (2022). *ExoMiner.* ApJ 926, 120.
- Thompson, S. E. et al. (2018). *Planetary Candidates Observed by Kepler. VIII.* ApJS 235, 38.
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
- Zaheer, M. et al. (2017). *Deep Sets.* NeurIPS.
- Ilse, M., Tomczak, J. y Welling, M. (2018). *Attention-based Deep Multiple Instance Learning.* ICML.
- Jain, S. y Wallace, B. C. (2019). *Attention is not Explanation.* NAACL.
- Guo, C. et al. (2017). *On Calibration of Modern Neural Networks.* ICML.
- Saerens, M., Latinne, P. y Decaestecker, C. (2002). *Adjusting the Outputs of a Classifier to New a Priori Probabilities.* Neural Computation 14(1), 21–41.
- Gal, Y. y Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation.* ICML.
- Lakshminarayanan, B., Pritzel, A. y Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS.
- Geifman, Y. y El-Yaniv, R. (2017). *Selective Classification for Deep Neural Networks.* NeurIPS.
