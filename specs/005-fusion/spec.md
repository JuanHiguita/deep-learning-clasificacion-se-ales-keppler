# 005 · Fusión y ablación — especificación

**PT5 del plan · Capa 4 · Dueño: Deep Learning · Responde la pregunta 2**

## Para qué

Unir los vectores de las cuatro ramas en una sola decisión y medir cuánto aporta cada rama.

## Entradas

`modelos/{vistas,transitos,escalares}/v1/` (spec 004) y `modelos/imagen/v0/`.

**Decisión DD-13 (19-nov): entregas independientes.** La entrega del curso usa como rama de imagen el **codificador v0**: la rama 2D de la demo, código anterior a los cursos, entrenada aquí por partición y congelada. No se espera nada del curso de Visión. Sustituirla por el codificador de Visión es trabajo de la unión posterior (tesis).

## Requisitos funcionales

### Imagen v0

- **RF-1** `codificador_imagen_v0.py` usa la rama 2D de `src/demo/demo_cnn.py` (3 × Conv2D 3×3 sin pooling → Flatten → Dropout(0,4) → Dense(32, name="embedding")) con una cabeza `Dense(3)`, entrenada con el motor (spec 003), aumentación por rotaciones de 90° y ruido 0,05, semilla 0.
- **RF-2** Fuente de imágenes: `img` de `data/dataset_demo.npz` (spec 004, T4.6), construido con `src/demo/demo_dataset.py` desde los TPF de Q5. Si no hay TPF y no se pueden descargar: la fusión se entrega con 3 ramas y se declara.
- **RF-3** Su `manifiesto.json` incluye: `"origen": "v0 · código de la demo (base común) · no es resultado de ningún curso"`.
- **RF-4** Este curso no modifica la arquitectura de la imagen ni reporta su desempeño como resultado propio. La rama sola aparece en la tabla de ablación únicamente como referencia, con su origen.

### Fusión

- **RF-5** Para cada partición *k* se cargan los `vectores_particion_k.npz` de las 4 ramas y se unen por `id_ejemplo`. Bandera de disponibilidad por rama: 0 si el ejemplo no tiene vector o si `disponible_*` = 0 (entonces el vector va en ceros).
- **RF-6** **E6a:** cabeza del contrato §6: entrada 116 → Dense(64, relu) → Dropout(0,3) → Dense(3, softmax). Se entrena con el motor (mismas particiones, validación interna y parada temprana), 5 semillas de cabeza. Los vectores se estandarizan por partición con estadísticas del entrenamiento.
- **RF-7** **E6c:** fusión tardía: promedio de las probabilidades de prueba de las 4 cabezas individuales (semilla 0).
- **RF-8** **Diagnóstico del apilado:** por partición, NLL de la cabeza en entrenamiento y en validación interna. Si `NLL_val > 2 × NLL_ent` en la media de las particiones: se reporta y se activa el **ajuste cruzado**: la cabeza de la partición *k* se entrena solo con los vectores de la validación interna de *k* (que los codificadores no vieron), con validación por 4 submuestras por estrella. Se anota como desviación.
- **RF-9** **E7 (ablación principal):** cabeza reentrenada con 3 ramas (quitando cada una: 4 configuraciones) y con 1 rama (4 configuraciones), 5 semillas cada una. Entrada de dimensión reducida, misma arquitectura de cabeza.
- **RF-10** **E7b (C):** cabeza entrenada con *dropout* de ramas (cada rama se apaga con p = 0,15 poniendo vector y bandera en 0) y evaluada apagando una rama en prueba.
- **RF-11** **E6b (C):** ajuste fino conjunto de las ramas 1, 2 y 4 con tasa 1e-4 a partir de v1; la imagen entra como vector fijo.

### Reporte

- **RF-12** `resultados/dl/tablas/p2_fusion.csv`: fusión E6a, E6c, cada ablación y cada rama sola; AUC, NLL, ECE, exactitud, con intervalos; diferencias pareadas: E6a − mejor rama sola, E6a − M0 completo, E6a − E6c, E6a − (sin cada rama).
- **RF-13** Las mismas diferencias por bandera de FP (`fp_ss`, `fp_co`, `fp_nt`) y por tercil de MES, con `n`.
- **RF-14** Comparación con S3-B, S3-C y `koi_score` (este último marcado como comparador externo que contiene la decisión del Robovetter).
- **RF-15** `resultados/dl/fusion/manifiesto.json` registra la versión de la imagen (v0) y el SHA-256 de sus vectores.

## Criterios de aceptación

- [ ] E6a, E6c y E7 completos.
- [ ] Diagnóstico del apilado escrito en `metricas.json` y en la bitácora.
- [ ] `verificar_contrato.py` pasa con `modelos/imagen/`.
- [ ] La predicción final del sistema (`resultados/dl/fusion/e6a/pred_semilla*.npz`) existe con `p_prueba` y `p_validacion`: es la entrada de la spec 006.

## Fuera de alcance

Cualquier cambio en la rama de imagen. Cualquier comparación con resultados del informe de Visión.
