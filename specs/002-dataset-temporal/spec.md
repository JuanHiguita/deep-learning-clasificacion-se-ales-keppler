# 002 · Dataset temporal — especificación

**PT2 del plan · Capa 1 · Dueño: Deep Learning**

## Para qué

Convertir las curvas PDCSAP de Q1–Q17 de cada estrella en las entradas de las ramas 1 (vistas), 2 (tránsitos uno por uno) y 4 (escalares), con las formas del contrato §4.

## Historias

1. Como autor, quiero que las tres arquitecturas de la pregunta 1 lean exactamente las mismas ventanas, para que la comparación sea justa.
2. Como autor, quiero que la alternancia de profundidad entre tránsitos pares e impares sobreviva al preprocesamiento, porque es la pista que justifica la rama 2.
3. Como autor, quiero saber qué estrellas y qué tránsitos se descartaron y por qué.

## Requisitos funcionales

### Descarga

- **RF-1** `descargar_mast.py --que llc` baja, por `kepid` de `ejemplos.csv`, todas las curvas de cadencia larga (`*_llc.fits`) directamente de `archive.stsci.edu`, sin `lightkurve`. Con `--que tpf` baja el TPF de Q5 de cada estrella. Reanudable: si el archivo existe y su tamaño es > 0, no lo baja de nuevo. Tres reintentos con espera creciente. Pausa entre estrellas para no saturar MAST.
- **RF-2** Escribe `data/llc/manifiesto_llc.csv`: `kepid, trimestre, archivo, bytes, sha256`. Al final imprime cuántas estrellas tienen menos de 4 trimestres.

### Limpieza (por trimestre)

- **RF-3** Flujo `PDCSAP_FLUX`, solo cadencias con `SAP_QUALITY = 0` (más estricto que la máscara `default` de `lightkurve`), sin NaN, normalizado por su mediana.
- **RF-4** Aplanado con Savitzky–Golay de 401 cadencias, enmascarando los tránsitos de **todos** los KOIs de la estrella (±0,75 duraciones a cada lado, es decir, 1,5 duraciones en total). Para el ejemplo `sin_senal` la máscara es la misma (la de las señales reales).
- **RF-5** Recorte de atípicos solo hacia arriba (4σ). Ruido por trimestre: `σ = 1,4826 · MAD` de los puntos fuera de tránsito.

### Escala por ejemplo (decisión DD-04)

- **RF-6** `profundidad_medida` = −mediana de `(flujo − 1)` de los puntos con `|fase| < 0,25 · duración` respecto a la época del ejemplo. `σ_local` = `σ_global / √n_puntos`. `escala = max(profundidad_medida, 3 · σ_local)`.
- **RF-7** Toda serie se expresa como `(flujo − 1) / escala` y se recorta a [−5, 5].

### Rama 1 · vistas

- **RF-8** `vista_global` (2001, 1): fase en [−0,5; 0,5), mediana por intervalo de igual ancho.
- **RF-9** `vista_local` (201, 4): fase en ±4 duraciones. Canal 0: todos los tránsitos. Canal 1: tránsitos pares. Canal 2: impares. Canal 3: la misma ventana centrada en fase 0,5.
- **RF-10** Intervalos vacíos: interpolación lineal desde los vecinos. Si más de un tercio de los intervalos del canal 0 están vacíos, `disponible_vistas = 0` y los arreglos van en ceros.

### Rama 2 · tránsitos

- **RF-11** Para cada tránsito *n* con centro `t0 + n·P` dentro de los datos: ventana de ±2,5 duraciones, interpolada linealmente a 61 puntos equiespaciados.
- **RF-12** El tránsito es válido si: (a) tiene ≥ 70 % de las cadencias esperadas en la ventana; (b) tiene ≥ 2 cadencias con `|t − centro| < 0,5 · duración`; (c) su ventana no se solapa con la máscara de otro KOI de la estrella.
- **RF-13** Selección determinista de hasta 32: hasta 16 pares y 16 impares, eligiendo en cada paridad índices equiespaciados sobre la lista de válidos ordenada por tiempo; si una paridad tiene menos de 16, la otra completa hasta 32. Resultado en orden cronológico.
- **RF-14** `transitos` (32, 61), `mascara_transitos` (32,), `paridad_transitos` (32,; 0 par, 1 impar, −1 vacío), `n_transitos_validos` (total antes de seleccionar).
- **RF-15** `consistencia` (4,), calculada con **todos** los tránsitos válidos (no solo los 32):
  1. desviación estándar robusta de la profundidad por tránsito, dividida por su error esperado;
  2. (profundidad media impar − par) dividida por el error de esa diferencia;
  3. log10(1 + número de tránsitos válidos);
  4. mayor desviación absoluta de un tránsito respecto a la mediana, en unidades de su error.
  Los estadísticos 1, 2 y 4 se comprimen con `sign(x)·log10(1 + |x|)`, porque en estrellas dobles brillantes valen miles de sigmas (así, 3σ ≈ 0,6 y 9σ ≈ 1,0).
  Con menos de 4 tránsitos válidos: `disponible_transitos = 0`.

### Rama 4 · escalares (decisión DD-09)

- **RF-16** `escalares` (7,) en este orden, con `nombres_escalares` guardado en el `.npz`: `log10_periodo_d, log10_duracion_h, profundidad_medida_ppm, snr_medido, log10_n_transitos_validos, koi_dikco_msky, koi_dicco_msky`. `snr_medido` = `profundidad_medida / σ_local`.
- **RF-17** `faltantes` (7,): 1 donde el valor es NaN. En el arreglo `escalares` el faltante se guarda como 0 **junto con** su aviso; la estandarización ocurre después, en el entrenamiento, con estadísticas de la parte de entrenamiento calculadas solo sobre los no faltantes.
- **RF-18** En ejemplos `sin_senal`, los dos escalares de centroide van como faltantes.

### Clase `sin_senal`

- **RF-19** Mismo proceso con época `t0 + 0,25 · P`. Se guarda `aviso_sin_senal` = 1 si su `snr_medido > 5`. No se eliminan; el conteo va al manifiesto y al informe.

### Salida

- **RF-20** `data/dataset_temporal.npz` con: `id_ejemplo, vista_global, vista_local, transitos, mascara_transitos, paridad_transitos, n_transitos_validos, consistencia, escalares, faltantes, nombres_escalares, disponible_vistas, disponible_transitos, aviso_sin_senal`. `float32`, sin NaN, sin objetos.
- **RF-21** `data/dataset_temporal_registro.csv`: por ejemplo, trimestres usados, tránsitos totales, válidos y descartados por cada motivo (a, b, c).
- **RF-22** No se generan ejemplos prerregistrados.

### EDA

- **RF-23** `notebooks/eda_temporal.ipynb`: tránsitos válidos por clase; `consistencia[1]` (par−impar) por bandera de FP; profundidad en fase 0,5 por bandera; 12 ejemplos al azar (semilla 7) con sus tres representaciones; tabla de descartes. Usa etiquetas, **no** predicciones.

## Criterios de aceptación

- [ ] `verificar_contrato.py` pasa (formas, NaN, identificadores, lista blanca).
- [ ] Pruebas sintéticas en verde (ver plan).
- [ ] Inspección visual: K00022.01 (planeta) muestra tránsitos iguales y `consistencia[1]` cerca de 0; K03615.01 (estrella doble) muestra alternancia y `|consistencia[1]|` grande (`00_ESTADO` §5). Guardar la figura en `resultados/dl/eda/`.
- [ ] Al menos el 95 % de los ejemplos tiene `disponible_vistas = 1` y `disponible_transitos = 1`. Si no, detenerse.

## Respaldo

Si la descarga no está completa el domingo 22 de noviembre: construir las mismas claves desde el TPF de Q5 (curva de apertura, como `src/demo/demo_dataset.py`), declararlo como limitación y seguir.

## Fuera de alcance

Corrección con CBV propia, curvas sintéticas, periodos largos, registro entre trimestres (no hace falta en 1D).
