# Bitácora del curso de Deep Learning

Única memoria entre sesiones. **Actualizar antes de cerrar cada sesión.**

## Estado por paquete de trabajo

| PT | Spec | Estado | Hito | Fecha límite |
|---|---|---|---|---|
| PT0 Entorno | `specs/000-entorno` | **Siguiente: ejecutar `scripts\descargar_recursos.bat` en el PC** | — | vie 20-nov |
| PT1 Base común | `specs/001-base-comun` | Código y pruebas listos; falta ejecutarlo con datos reales | D1–D2, `base-v1` | sáb 21-nov |
| PT2 Dataset temporal | `specs/002-dataset-temporal` | Extractor y pruebas listos; faltan la descarga y la corrida real | D3 | dom 22-nov |
| PT3 Motor | `specs/003-motor-experimentos` | En curso: T3.1–T3.3 hechas | — | lun 23-nov |
| PT4 Ramas | `specs/004-ramas-temporales` | Pendiente | D4, `ramas-v1` | jue 26-nov |
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

<!-- Plantilla de entrada
### AAAA-MM-DD · <modelo> · PT<n>
- **Hecho:**
- **Comandos:**
- **Archivos creados o cambiados:**
- **Resultados (tal como salieron):**
- **Desviaciones:**
- **Siguiente paso exacto:**
-->
