# Bitácora del curso de Deep Learning

Única memoria entre sesiones. **Actualizar antes de cerrar cada sesión.**

## Estado por paquete de trabajo

| PT | Spec | Estado | Hito | Fecha límite |
|---|---|---|---|---|
| PT0 Entorno | `specs/000-entorno` | **Siguiente** | — | vie 20-nov |
| PT1 Base común | `specs/001-base-comun` | Pendiente | D1–D2, `base-v1` | sáb 21-nov |
| PT2 Dataset temporal | `specs/002-dataset-temporal` | Pendiente (la descarga arranca en PT0) | D3 | dom 22-nov |
| PT3 Motor | `specs/003-motor-experimentos` | En curso: T3.1–T3.3 hechas | — | lun 23-nov |
| PT4 Ramas | `specs/004-ramas-temporales` | Pendiente | D4, `ramas-v1` | jue 26-nov |
| PT5 Fusión | `specs/005-fusion` | Pendiente | D5 (I2: vie 27-nov) | dom 29-nov |
| PT6 Calibración | `specs/006-calibracion-incertidumbre` | Pendiente | D6 | mié 2-dic |
| PT7 Consolidación | `specs/007-consolidacion` | Pendiente | D6 | mié 2-dic |
| PT8 Informe | `specs/008-informe` | Pendiente | D7, `curso-dl-v1` | sáb 5-dic |

## Decisiones abiertas (responde Juan)

| ID | Pregunta | Por defecto | Respuesta |
|---|---|---|---|
| DA-1 | ¿Los dos profesores aceptaron el proyecto compartido? | Seguir | |
| DA-2 | ¿Visión ya construyó la base o definió el prerregistro? | Definición de la spec 001 | |
| DA-3 | ¿Fecha de I2? | vie 27-nov | |
| DA-4 | ¿Formato de entrega: los cuatro o uno? | Los cuatro | |
| DA-5 | Enunciado y rúbrica oficiales | No están | |
| DA-6 | Título definitivo | El provisional | |

## Desviaciones del plan

| Fecha | Qué cambió | Por qué | Efecto |
|---|---|---|---|
| 2026-11-19 | Repositorio propio del curso, separado del de la tesis (DD-12) | Lo pidió Juan | La base común se copia entre repositorios y se comprueba por SHA-256 |

## Sesiones

### 2026-11-19 · Claude Fable 5.2 · planeación

- **Hecho:** plan de ejecución; arquitectura del repositorio; constitución; `config/protocolo.json`; especificaciones 000–008 (spec, plan y tareas); núcleo del motor (`comun/protocolo.py`, `comun/metricas.py`, `comun/estadistica.py`) con 17 pruebas en verde (`tests/test_motor.py`); `data/seleccion_demo.csv` (260 KOIs: 130 confirmados y 130 falsos positivos).
- **Comprobado:** `pytest -q` → 17 pruebas pasan (Python 3.10, numpy y scikit-learn; sin Keras).
- **No comprobado:** instalación de Keras + JAX en el PC; acceso a MAST y al Exoplanet Archive; existencia de los TPF de la demo; que las columnas de la spec 001 RF-1 existan todas en `q1_q17_dr25_koi`.
- **Pendiente de traer:** scripts de la demo a `src/demo/` y copias de `00_ESTADO_Y_DECISIONES.md`, `02_contrato_integracion.md` y `03_plan_integrado_cursos.md` a `docs/planeacion/` (tareas T0.5 y T0.6).
- **Siguiente paso exacto:** `specs/000-entorno/tasks.md`, tarea T0.1.

<!-- Plantilla de entrada
### AAAA-MM-DD · <modelo> · PT<n>
- **Hecho:**
- **Comandos:**
- **Archivos creados o cambiados:**
- **Resultados (tal como salieron):**
- **Desviaciones:**
- **Siguiente paso exacto:**
-->
