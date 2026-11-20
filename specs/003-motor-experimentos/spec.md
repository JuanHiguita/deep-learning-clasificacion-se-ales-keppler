# 003 · Motor de experimentos — especificación

**PT3 del plan · Capa 2 · Dueño: Deep Learning (reutilizable por la tesis)**

## Para qué

Que todos los modelos se entrenen, evalúen y exporten con el mismo código, de modo que las diferencias entre ellos vengan de la arquitectura y no del procedimiento.

## Estado

Ya implementado y probado (17 pruebas en `tests/test_motor.py`):

- `comun/protocolo.py`: `cargar_protocolo`, `separar`, `comprobar_sin_solape`.
- `comun/metricas.py`: `q_planeta`, `auc`, `auc_planeta_fp`, `exactitud`, `matriz_confusion`, `nll`, `brier`, `ece_top`, `ece_binario`, `pesos_para_tasa_base`, `corregir_prior`.
- `comun/estadistica.py`: `bootstrap_por_estrella`, `diferencia_media_semillas`.

Falta `comun/entrenamiento.py`, que es el objeto de esta spec.

## Requisitos funcionales

- **RF-1** `entrenar_cv` recibe un constructor de modelo, las entradas alineadas con `ejemplos.csv`, el protocolo y una lista de semillas. Para cada semilla y partición *k*: separa con `protocolo.separar`, comprueba el solape, estandariza lo que haga falta con estadísticas del entrenamiento, entrena y predice.
- **RF-2** Entrenamiento según `config/protocolo.json`: AdamW, lote 32, hasta 150 épocas, parada temprana por NLL de validación interna con paciencia 15 y restauración de los mejores pesos, pérdida con pesos por clase.
- **RF-3** La aumentación es una función opcional `aumentar(entradas, rng) → entradas`, aplicada solo al entrenamiento, una vez por época, con un generador sembrado por `(semilla, k)`.
- **RF-4** Los ejemplos con `disponible_* = 0` para la rama que se entrena se excluyen del entrenamiento y de la validación de esa rama; en la prueba reciben la probabilidad a priori de las clases del entrenamiento, y se cuenta cuántos son.
- **RF-5** Salida por experimento en `resultados/dl/<experimento>/`:
  - `pred_semilla<s>.npz`: `id_ejemplo`, `particion`, `p_prueba` (n, 3), `p_validacion` (n, 3) y `particion_modelo_validacion` (n,): la predicción de cada ejemplo hecha por el modelo de la partición en la que ese ejemplo es validación interna (ver plan);
  - `historial_semilla<s>.json`: pérdida de entrenamiento y validación por época y partición, época elegida;
  - `metricas.json`: todas las métricas de la sección 4.3 del plan, con intervalos, más número de parámetros y tiempo medio por época;
  - `manifiesto.json`.
- **RF-6** `exportar_codificador`: guarda, para la semilla congelada, `modelos/<rama>/v<N>/particion_k.keras` (submodelo hasta la capa `embedding`), `vectores_particion_k.npz` (`id_ejemplo`, `z`) para **todos** los ejemplos no prerregistrados, las estadísticas de estandarización y `manifiesto.json`.
- **RF-7** Pruebas de cordura como funciones:
  - `prueba_etiquetas_barajadas`: entrena con etiquetas permutadas dentro del entrenamiento; exige AUC planeta vs FP entre 0,40 y 0,60.
  - `prueba_sobreajuste_lote`: 32 ejemplos, sin regularización, 300 épocas; exige exactitud de entrenamiento ≥ 0,95.
- **RF-8** `evaluar_predicciones(p, ejemplos, protocolo) → dict`: la única función que calcula métricas de un experimento; la usan todas las specs posteriores.

## Requisitos no funcionales

- `KERAS_BACKEND=jax` se fija con `os.environ.setdefault` antes de importar Keras.
- Reproducible: la misma semilla produce las mismas predicciones en la misma máquina (tolerancia 1e-5).
- Ningún acceso a `p_prueba` dentro del bucle de entrenamiento.

## Criterios de aceptación

- [ ] Prueba de integración (`-m lenta`): 40 estrellas sintéticas, modelo denso, 3 épocas; produce todos los archivos de RF-5 y RF-6 y `verificar_contrato.py` los acepta.
- [ ] Prueba: los índices de prueba nunca aparecen en los lotes de entrenamiento (se inyecta un marcador en una entrada y se comprueba).
- [ ] Prueba: la estandarización usa solo el entrenamiento (cambiar los valores de prueba no cambia las estadísticas).
- [ ] Medir y anotar en la bitácora el tiempo de una corrida real de M4 (escalares): con eso se confirma el presupuesto de cómputo del plan §8.
