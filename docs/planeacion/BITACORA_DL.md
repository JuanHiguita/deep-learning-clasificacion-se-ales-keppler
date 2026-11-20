# Bitácora del curso de Deep Learning

Única memoria entre sesiones. **Actualizar antes de cerrar cada sesión.**

## Estado por paquete de trabajo

| PT | Spec | Estado | Hito | Fecha límite |
|---|---|---|---|---|
| PT0 Entorno | `specs/000-entorno` | Hecho en el PC (Python 3.13, Keras 3.15.1 con JAX) | — | vie 20-nov |
| PT1 Base común | `specs/001-base-comun` | **Hecho con datos reales** (falta etiqueta `base-v1`) | D1–D2, `base-v1` | sáb 21-nov |
| PT2 Dataset temporal | `specs/002-dataset-temporal` | **Dataset construido** (95,3 % disponible); faltan figura de inspección y EDA | D3 | dom 22-nov |
| PT3 Motor | `specs/003-motor-experimentos` | **Hecho** | — | lun 23-nov |
| PT4 Ramas | `specs/004-ramas-temporales` | En curso: M4 entrenada y exportada | D4, `ramas-v1` | jue 26-nov |
| PT5 Fusión | `specs/005-fusion` | Pendiente | D5 (con imagen v0) | dom 29-nov |
| PT6 Calibración | `specs/006-calibracion-incertidumbre` | Pendiente | D6 | mié 2-dic |
| PT7 Consolidación | `specs/007-consolidacion` | Pendiente | D6 | mié 2-dic |
| PT8 Informe | `specs/008-informe` | Pendiente | D7, `curso-dl-v1` | sáb 5-dic |

## Decisiones abiertas (responde Juan)

| ID | Pregunta | Por defecto | Respuesta |
|---|---|---|---|
| DA-1 | ¿Los dos profesores aceptaron el proyecto compartido? | Seguir | **Sí (19-nov)** |
| DA-2 | ¿Visión ya construyó la base o definió el prerregistro? | Definición de la spec 001 | **Visión no ha empezado (19-nov): se construye aquí** |
| DA-3 | ¿Fecha de I2? | vie 27-nov | **No aplica: entregas independientes, imagen v0 (DD-13)** |
| DA-4 | ¿Formato de entrega: los cuatro o uno? | Los cuatro | |
| DA-5 | Enunciado y rúbrica oficiales | No están | Juan aún no la tiene (19-nov) |
| DA-6 | Título definitivo | El provisional | |

## Desviaciones del plan

| Fecha | Qué cambió | Por qué | Efecto |
|---|---|---|---|
| 2026-11-19 | Repositorio propio del curso, separado del de la tesis (DD-12) | Lo pidió Juan | La base común se copia entre repositorios y se comprueba por SHA-256 |
| 2026-11-20 | Particiones con reparto propio por SHA-256 en lugar de `StratifiedGroupKFold` | En la máquina de pruebas, otra versión de scikit-learn dio un reparto distinto y desbalanceado | Particiones idénticas en cualquier máquina; balance de ±1 KOI por clase |
| 2026-11-19 | Entregas independientes: sin I2; la rama de imagen de la entrega es el codificador v0 de la demo (DD-13) | Lo decidió Juan; Visión no ha empezado | La spec 005 usa v0; la unión con Visión queda para la tesis |

## Sesiones

### 2026-11-19 · Claude Fable 5.2 · planeación

- **Hecho:** plan de ejecución; arquitectura del repositorio; constitución; `config/protocolo.json`; especificaciones 000–008 (spec, plan y tareas); núcleo del motor (`comun/protocolo.py`, `comun/metricas.py`, `comun/estadistica.py`) con 17 pruebas en verde (`tests/test_motor.py`); `data/seleccion_demo.csv` (260 KOIs: 130 confirmados y 130 falsos positivos).
- **Comprobado:** `pytest -q` → 17 pruebas pasan (Python 3.10, numpy y scikit-learn; sin Keras).
- **No comprobado:** instalación de Keras + JAX en el PC; acceso a MAST y al Exoplanet Archive; existencia de los TPF de la demo; que las columnas de la spec 001 RF-1 existan todas en `q1_q17_dr25_koi`.
- **Pendiente de traer:** scripts de la demo a `src/demo/` y copias de `00_ESTADO_Y_DECISIONES.md`, `02_contrato_integracion.md` y `03_plan_integrado_cursos.md` a `docs/planeacion/` (tareas T0.5 y T0.6).
- **Siguiente paso exacto:** `specs/000-entorno/tasks.md`, tarea T0.1.

### 2026-11-19/20 · Claude Fable 5.2 · PT1 y PT2 (código)

- **Hecho:** `comun/utilidades.py`, `descargar_koi.py`, `construir_ejemplos.py`, `verificar_contrato.py`, `lineas_base_tabulares.py`; `temporal/descargar_mast.py` (curvas Q1–Q17 y TPF de Q5, directo de `archive.stsci.edu`), `temporal/curvas.py` (extractor), `temporal/lectura.py`, `temporal/construir_dataset_temporal.py`; `scripts/descargar_recursos.bat`; pruebas `test_base_comun.py`, `test_extractor.py`, `sinteticos.py`.
- **Resultados (tal como salieron):** `pytest -q` → 59 pruebas pasan, incluida una corrida de punta a punta con FITS sintéticos (lectura → extractor → `.npz` → `verificar_contrato`).
- **Comprobado contra los servidores reales (desde el navegador):** las 19 columnas de la spec 001 existen en `q1_q17_dr25_koi`; el listado de `archive.stsci.edu/missions/kepler/lightcurves/0074/007440748/` tiene 17 archivos `_llc.fits` con el formato que espera `descargar_mast.py`; el TPF de Q5 se llama `kplr<kepid>-2010174085026_lpd-targ.fits.gz`.
- **NO hecho: ninguna descarga de datos.** El entorno donde corre el asistente no tiene salida a `exoplanetarchive.ipac.caltech.edu` ni a `archive.stsci.edu` (el proxy responde 403). Las descargas se hacen en el PC con `scripts\descargar_recursos.bat`.
- **Sin probar con datos reales:** todo el extractor. La primera corrida real debe pasar por la inspección de K00022.01 y K03615.01 (spec 002, T2.9).
- **Desviaciones:** (1) descarga directa por HTTPS en lugar de `lightkurve`; (2) calidad `SAP_QUALITY = 0` en lugar de la máscara `default`; (3) los estadísticos de consistencia 1, 2 y 4 van en escala `sign·log10(1+|x|)`. Todas anotadas en la spec 002.
- **Siguiente paso exacto:** ejecutar `scripts\descargar_recursos.bat`; luego spec 002 T2.8–T2.11 y spec 003 T3.4.

### 2026-11-20 · Claude Fable 5.2 · PT1 y PT2 con datos reales

- **Descargas (en el PC, `scripts\\descargar_recursos.bat`):** DR25 (8.054 filas) y acumulada (9.564 filas); 3.792 curvas `_llc.fits` (1,6 GB); 224 TPF de Q5 (525 MB; 21 estrellas no tienen TPF en Q5). El primer intento dio HTTP 403 por una tilde en la cabecera `User-Agent`; corregido.
- **Base (`base-v1` pendiente de etiquetar):** 245 KOIs, 490 ejemplos (130 planetas, 115 FP, 245 sin señal); 26 planetas y 23 FP por partición. Salieron 15 KOIs de la demo, todos FP (8 sin fila en DR25, 7 sin etiqueta definitiva). 0 prerregistrados. Terciles de MES: 35,6 y 100,6.
- **Líneas base (AUC, IC 95 %):** S3-B selección 0,857 [0,804–0,904]; S3-B tabla completa 0,920 [0,882–0,952]; S3-C selección 0,979 [0,962–0,992]; S3-C tabla completa 0,989 [0,978–0,996]; `koi_score` 0,996 [0,986–1,000].
- **Dataset temporal:** primera corrida con 89,6 % de disponibilidad (< 95 %). Causa: la regla que descartaba ventanas que tocaran a otro KOI anulaba los ejemplos sin señal con duración/periodo > 1/13 y las estrellas con compañeros de periodo muy corto (K00191.01). Corregido (spec 002, RF-12, RF-12b y RF-19). Segunda corrida: vistas 467/490, tránsitos 471/490, ambas 467/490 = 95,3 %. No disponibles: 14 ejemplos de 7 estrellas con menos de 4 trimestres y 9 ejemplos sin señal con duración/periodo > 0,22. 10 ejemplos sin señal con SNR > 5 (se conservan).
- **Inspección T2.9:** K00022.01 (planeta): par−impar = −0,16 (escala log), vista local −1,01 / −1,01 / −1,00. K03615.01 (estrella doble): par−impar = 2,52, pares −0,90 e impares −1,59. La alternancia se recupera. Falta la figura.
- **Pruebas:** 61 pasan.
- **Decisión de Juan (20-nov):** se siguen los planes establecidos. La rama 4 conserva el centroide (7 escalares, spec 002 RF-16 y `ROADMAP_PROYECTO` 4.4). No se agregan la variante sin centroide ni la prueba de circularidad al alcance del curso. El informe declara que S3-C ya da 0,98–0,99 en esta muestra y que hay circularidad parcial con las etiquetas (`00_ESTADO` §5).
- **Siguiente paso exacto:** etiqueta `base-v1`; spec 003 T3.4 (motor de entrenamiento).

### 2026-11-20 · Claude Fable 5.2 · PT3 (motor) y primera rama

- **Hecho:** `comun/entrenamiento.py` (validación cruzada, parada temprana, estandarización por partición, exportación de codificadores, pruebas de cordura, métricas con intervalos); `temporal/modelos.py` (M4); `temporal/exp_ramas.py`; `tests/test_entrenamiento.py`, `tests/conftest.py`.
- **Pruebas:** 63 rápidas y 2 lentas pasan (entorno de desarrollo: Keras 3.12.4 con JAX, Python 3.10; en el PC hay Keras 3.15.1 y Python 3.13: **volver a correr `pytest -q` y `pytest -q -m lenta` allí**).
- **E0 de M4 (cordura):** AUC con etiquetas barajadas 0,503; memoriza un lote con exactitud 1,000. Pasa.
- **E1 de M4 (rama 4, escalares con centroide), tal como salió:** AUC planeta vs FP 0,9573 / 0,9415 / 0,9449 en las semillas 0, 1 y 2 (media 0,948; desviación 0,007). Semilla 0: AUC 0,957 [0,927–0,982], NLL 0,200 [0,153–0,256], ECE 0,017, exactitud de 3 clases 0,933. Matriz de confusión (filas reales): [[124, 6, 0], [11, 98, 6], [0, 10, 235]]. 1.059 parámetros; 0,007 s por época; el experimento completo (15 corridas y exportación) tarda 29 s.
- **Presupuesto de cómputo:** confirmado como holgado; no hace falta recortar semillas.
- **Exportado:** `modelos/escalares/v1/` (contrato verificado).
- **Siguiente paso exacto:** spec 004 T4.1–T4.2 (M1 y los tres agregadores de la rama 2), E0 y E1–E2.

<!-- Plantilla de entrada
### AAAA-MM-DD · <modelo> · PT<n>
- **Hecho:**
- **Comandos:**
- **Archivos creados o cambiados:**
- **Resultados (tal como salieron):**
- **Desviaciones:**
- **Siguiente paso exacto:**
-->
