# 001 · Base común — especificación

**PT1 del plan · Capa 1–2 · Dueño: ambos cursos (infraestructura compartida)**

## Para qué

Tener una sola tabla maestra de ejemplos, con etiquetas DR25, particiones por estrella y prerregistro, que usen los dos cursos. Y medir el listón tabular que cualquier red debe superar.

## Historias

1. Como autor, quiero regenerar `compartido/ejemplos.csv` con un comando y obtener exactamente el mismo archivo, para que toda cifra sea reproducible.
2. Como autor, quiero que sea imposible entrenar con una estrella de prueba o con un ejemplo prerregistrado, para no inflar resultados.
3. Como lector del informe, quiero saber cuánto logra un modelo tabular sin columnas con fuga, para juzgar si la red aporta.

## Requisitos funcionales

- **RF-1** `descargar_koi.py` consulta `q1_q17_dr25_koi` por TAP con columnas explícitas: `kepid, kepoi_name, koi_disposition, koi_pdisposition, koi_score, koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec, koi_period, koi_time0bk, koi_duration, koi_depth, koi_model_snr, koi_max_mult_ev, koi_num_transits, koi_kepmag, koi_dikco_msky, koi_dicco_msky`. Con `--tabla cumulative` descarga la tabla acumulada (para el prerregistro).
- **RF-2** Si una columna no existe, el script termina con error y nombra la columna. No la sustituye.
- **RF-3** El manifiesto guarda: consulta exacta, URL, fecha UTC, SHA-256 del CSV, número de filas y versiones de librerías.
- **RF-4** `construir_ejemplos.py` cruza `data/seleccion_demo.csv` con DR25 por `kepoi_name` y conserva los KOIs con `koi_disposition` ∈ {CONFIRMED, FALSE POSITIVE} en DR25. Imprime y guarda cuántos KOIs cambian de etiqueta o se excluyen frente a la selección.
- **RF-5** Genera dos filas por KOI: `<kepoi_name>|transito` (clase 0 o 1) y `<kepoi_name>|sin_senal` (clase 2).
- **RF-6** Columnas: las del contrato §3 (`id_ejemplo, kepid, kepoi_name, clase, tipo_fp, periodo_d, mes, particion, prerregistrado`) más metadatos de reporte `fp_nt, fp_ss, fp_co, fp_ec, n_koi_en_estrella, t0_bkjd, duracion_h`.
- **RF-7** `tipo_fp`: `no_transito`, `eclipse_secundario` o `centroide` si hay exactamente una bandera (`nt`, `ss`, `co`); `otro` si la única es `ec` o hay varias; vacío si no es FP.
- **RF-8** `particion`: 5 particiones agrupadas por `kepid` y estratificadas por clase, con un reparto propio y determinista: dentro de cada clase las estrellas se ordenan por SHA-256 de `"<semilla>:<kepid>"` (semilla 0) y se reparten por turnos. No se usa `StratifiedGroupKFold`: su resultado cambia con la versión de scikit-learn y puede quedar desbalanceado (visto el 20-nov: 9 planetas y 1 falso positivo en una partición de prueba). Las dos filas de un KOI comparten partición.
- **RF-9** `prerregistrado`: verdadero si `koi_disposition` difiere entre DR25 y `cumulative` en la fecha de descarga. Si marca más del 10 % de la selección, el script termina con error (hay que consultar a Juan).
- **RF-10** `lineas_base_tabulares.py` mide, con las mismas particiones y solo ejemplos `transito` no prerregistrados:
  - S3-B: `HistGradientBoostingClassifier` sobre `koi_period, koi_duration, koi_depth, koi_model_snr, koi_num_transits, koi_kepmag`.
  - S3-C: S3-B + `koi_dikco_msky, koi_dicco_msky`.
  - Dos modalidades: (i) entrenado con la selección; (ii) entrenado con todos los KOIs etiquetados de DR25, sin las estrellas de prueba ni los prerregistrados.
  - `koi_score` como comparador externo (AUC directo, sin entrenar).
  - Salida: `metricas.json` con AUC e intervalo por *bootstrap* por estrella.
- **RF-11** `verificar_contrato.py --raiz .` comprueba todo lo que exista y termina con código distinto de 0 si algo falla:
  1. `id_ejemplo` único y con el formato del contrato §1.
  2. Para cada *k*: sin estrellas repetidas entre entrenamiento, validación interna y prueba (`comun.protocolo`).
  3. Ningún prerregistrado en archivos de vectores ni de predicciones.
  4. Formas de `dataset_temporal.npz` según `config/protocolo.json`; sin NaN; `id_ejemplo` ⊆ `ejemplos.csv`.
  5. Nombres de escalares (clave `nombres_escalares` del `.npz`) dentro de `escalares_permitidos`; ninguno en `columnas_prohibidas_como_entrada`.
  6. Para cada rama en `modelos/`: 5 codificadores, 5 archivos de vectores, dimensión correcta, cobertura de todos los ejemplos no prerregistrados.
  7. SHA-256 de `ejemplos.csv` igual al del manifiesto.

## Requisitos no funcionales

- NaN se dejan como NaN en las líneas base (el *boosting* los maneja); no se rellenan.
- Solo HTTPS al host `exoplanetarchive.ipac.caltech.edu`; tiempo de espera de 120 s; 3 reintentos.
- Determinismo: dos corridas con los mismos CSV de entrada producen el mismo `ejemplos.csv` byte a byte (orden por `id_ejemplo`, `lineterminator="\n"`).

## Criterios de aceptación

- [ ] `verificar_contrato.py` pasa.
- [ ] Diferencia máxima de 1 KOI por clase entre particiones.
- [ ] `metricas.json` de líneas base generado, con intervalos.
- [ ] Pruebas unitarias de RF-7, RF-8, RF-9 y RF-11 (casos que deben fallar) en verde.
- [ ] Etiqueta git `base-v1`. Aviso a Visión (plan, anexo B).

## Fuera de alcance

Gaia, periodos largos, particiones por módulo o estación, ampliar la muestra.

## Riesgos

- La definición del prerregistro es una aproximación (`00_ESTADO` §7 la tiene pendiente). Se declara así en el informe.
- Si Visión ya construyó la base, **no se ejecuta `construir_ejemplos.py`**: se copia y se verifica.
