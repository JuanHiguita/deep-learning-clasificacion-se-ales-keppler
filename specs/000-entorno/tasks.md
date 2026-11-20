# 000 · Entorno (PT0) — tareas

Modelo sugerido: Sonnet. Duración: medio día. Sin spec ni plan propios: es preparación.

- [ ] T0.1 Crear entorno virtual (Python ≥ 3.10). `pip install -r requirements.txt` y `pip install -e .`. Generar `requirements.lock.txt` con `pip freeze`.
- [ ] T0.2 Comprobar Keras + JAX: `KERAS_BACKEND=jax python -c "import keras; print(keras.backend.backend())"` debe imprimir `jax`. Si JAX no instala en Windows: usar WSL o Colab en CPU y anotarlo.
- [ ] T0.3 `pytest -q` debe pasar (17 pruebas del motor).
- [ ] T0.4 Comprobar acceso a datos: una consulta TAP de 1 fila a `q1_q17_dr25_koi` y `lightkurve.search_lightcurve("KIC 7440748", author="Kepler", cadence="long")`. Anotar el resultado en la bitácora.
- [ ] T0.5 Copiar los scripts de la demo a `src/demo/` y llenar `src/demo/ORIGEN.md`.
- [ ] T0.6 Copiar a `docs/planeacion/` los documentos `00_ESTADO_Y_DECISIONES.md`, `02_contrato_integracion.md` y `03_plan_integrado_cursos.md` desde el proyecto de Claude o el repositorio de la tesis. Son copias de lectura: la fuente de verdad es el proyecto de Claude.
- [ ] T0.7 Comprobar si existen los TPF de Q5 de la demo (`demo_kplr*.fits.gz`) y si el curso de Visión ya tiene `compartido/ejemplos.csv`. Si existe, copiarlo aquí con su manifiesto y saltar a la verificación de la spec 001.
- [ ] T0.8 `git init` si no existe, primer commit `PT0: entorno`.
- [ ] T0.9 **Lanzar la descarga de curvas** en cuanto exista `compartido/ejemplos.csv` (spec 002, T2.1). Es la tarea más lenta del proyecto.

**Aceptación:** T0.2 y T0.3 en verde; bitácora actualizada con T0.4 y T0.7.
