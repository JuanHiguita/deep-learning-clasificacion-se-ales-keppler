# Qué se hizo, cómo funciona y qué resultados salieron

**Proyecto del curso de Deep Learning** · Juan Sebastian Higuita Riascos · Especialización en Inteligencia Artificial, Universidad Nacional de Colombia · 20 de noviembre de 2026

*Clasificación calibrada de señales de tránsito de Kepler con una red multirrama y un codificador por tránsito con atención* (título provisional)

> Todas las cifras de este documento salen de `resultados/dl/tablas/` y se regeneran con los comandos de la sección 8. Los intervalos entre corchetes son del 95 %, calculados remuestreando estrellas (*bootstrap*). "No se distingue" significa que el intervalo de la diferencia incluye el 0.

---

## 1. La idea en un minuto

El telescopio Kepler midió el brillo de miles de estrellas cada 30 minutos durante 4 años. Cuando un planeta pasa por delante de su estrella, el brillo baja un poco y vuelve: es un **tránsito**, y se repite con un ritmo fijo. El problema es que hay **impostores** con el mismo ritmo: estrellas dobles que se eclipsan, estrellas vecinas cuya luz se mezcla y fallas del instrumento.

Este proyecto construye una red que mira cada señal de **cuatro formas** y las combina:

| Rama | Qué mira | Quién la construyó |
|---|---|---|
| 1 · Vistas | La curva "plegada": todos los tránsitos apilados y promediados | Este curso |
| 2 · Tránsitos | Cada tránsito por separado, hasta 32 | Este curso |
| 3 · Imagen | Los píxeles del telescopio (dónde cae la señal) | **Demo previa a los cursos (v0), congelada. No es un aporte de este curso** |
| 4 · Escalares | Periodo, duración, profundidad, señal/ruido y desplazamiento de centroide | Este curso |

Y responde tres preguntas:

1. ¿Qué arquitectura lee mejor la señal de tránsito?
2. ¿Cuánto mejora la decisión al fusionar las cuatro ramas?
3. ¿Se puede confiar en las probabilidades que entrega?

## 2. Respuestas cortas

| Pregunta | Respuesta | Evidencia |
|---|---|---|
| 1. Arquitectura | La atención entre tránsitos fue la mejor rama temporal, pero **su ventaja no se distingue de cero** frente al promedio simple ni frente a la CNN de curva plegada. Solo supera de forma distinguible a la GRU. | AUC 0,883 (atención), 0,863 (promedio), 0,862 (GRU), 0,864 (AstroNet) |
| 2. Fusión | **Sí mejora** frente a cualquier rama sola y frente a la demo. Queda **al nivel** del mejor modelo tabular, sin superarlo de forma demostrable. | Fusión 0,985 [0,974–0,994]; mejor rama sola 0,954; tabular con centroide 0,979–0,989 |
| 3. Calibración | El sistema salió algo sobreconfiado. La temperatura lo corrige en la dirección esperada, sin mejora distinguible. **El prior explícito sí funciona** cuando los planetas son escasos. La incertidumbre **señala los errores**. | ECE 0,033 → 0,022; con tasa base 0,05, ECE 0,108 → 0,021 |

Varios resultados son **negativos** (no se encontró diferencia). Se reportan con el mismo detalle que los positivos: con 245 señales de periodo corto, eso es lo que los datos permiten decir.

## 3. Los datos

- **Fuente:** catálogo DR25 de NASA (`q1_q17_dr25_koi`, 8.054 señales) para las etiquetas; curvas de luz PDCSAP de los 17 trimestres (3.792 archivos, 1,6 GB) y archivos de píxeles del trimestre 5 (224 archivos), descargados de MAST.
- **Muestra:** 245 señales (KOIs) de periodo entre 1 y 30 días, una por estrella: **130 planetas confirmados y 115 falsos positivos**. Son las de la demo, menos 15 que DR25 no etiqueta de forma definitiva.
- **Tres clases, 490 ejemplos:** cada señal da un ejemplo real (*planeta* o *falso positivo*) y otro de la misma estrella en una época desplazada un cuarto de órbita, con la señal real oculta (*sin señal*).
- **Particiones:** 5 grupos de estrellas (26 planetas y 23 falsos positivos cada uno). Cada ejemplo se predice con un modelo que **nunca vio su estrella**.
- **Disponibilidad:** 467 de 490 ejemplos (95,3 %) sirven para las ramas temporales. Faltan 14 de 7 estrellas con menos de 4 trimestres, y 9 ejemplos *sin señal* de binarias cuya duración supera un quinto del periodo (no existe una época limpia). 224 de las 245 estrellas tienen imagen.

### Cómo se preparan las entradas

1. Cada trimestre se normaliza y se **aplana** con un filtro suave que ignora los tránsitos (para no absorberlos).
2. Las cadencias que caen en el tránsito de **otro planeta de la misma estrella se ocultan**.
3. Todo se expresa en **unidades de profundidad**: se divide por la profundidad medida, de modo que un tránsito real vale −1. Así las señales profundas no se saturan y la diferencia entre eclipses pares e impares se conserva.
4. De ahí salen: la vista global (2001 puntos), la vista local (201 puntos × 4 canales: todos, pares, impares y fase 0,5), hasta 32 ventanas de tránsito de 61 puntos, 4 estadísticos de consistencia y 7 escalares.

**Comprobación con casos conocidos.** En `K00022.01` (planeta) pares e impares coinciden (−1,01 y −1,00). En `K03615.01` (estrella doble) los impares son 1,8 veces más profundos que los pares (−1,59 frente a −0,90), la misma razón que ya se conocía del catálogo.

## 4. Cómo funciona el sistema

```
vista global + vista local ──► Rama 1: CNN 1D tipo AstroNet ──────────► vector de 32 ─┐
32 tránsitos ──► mismo codificador para cada uno ──► atención ────────► vector de 32 ─┤
imagen de píxeles ──► Rama 3: CNN 2D de la demo (congelada) ──────────► vector de 32 ─┼─► fusión ─► temperatura ─► prior ─► probabilidades
7 escalares + avisos de dato faltante ──► Rama 4: red densa ──────────► vector de 16 ─┘            (planeta / falso positivo / sin señal)
```

- **Cada rama se entrena sola**, con su propia salida de 3 clases. Después se **congela** y solo se usa su vector.
- **La fusión** concatena los cuatro vectores más 4 banderas ("esta rama tiene dato o no") y decide con una capa de 64 neuronas.
- **La rama 2 es la pieza nueva.** Un mismo codificador (CNN 1D pequeña) lee cada tránsito; luego un *agregador* los combina. Se probaron tres agregadores sobre exactamente las mismas ventanas:
  - **Promedio:** todos pesan igual. Equivale a plegar la curva.
  - **GRU:** los lee en orden cronológico.
  - **Atención:** aprende cuánto peso darle a cada tránsito.

### Reglas que se respetaron

- Ninguna columna con fuga (`koi_score`, `koi_fpflag_*`, `koi_pdisposition`) entra a ningún modelo. Lo comprueba un script automático.
- Particiones por estrella, también para los codificadores congelados (uno por partición).
- **La prueba no decide nada:** la parada temprana, la elección del agregador y la temperatura se fijan con una validación interna distinta de la prueba.
- Hiperparámetros **fijados de antemano** (`config/protocolo.json`); no hubo búsqueda.
- Pruebas de cordura antes de entrenar: con las etiquetas barajadas el AUC queda en 0,50; el modelo puede memorizar un lote.

## 5. Resultados

### 5.1 El listón

| Modelo | AUC planeta vs FP | IC 95 % |
|---|---|---|
| Tabular sin centroide (S3-B), entrenado con la muestra | 0,857 | 0,804–0,904 |
| Tabular sin centroide (S3-B), entrenado con todo DR25 | 0,920 | 0,882–0,952 |
| Tabular con centroide (S3-C), entrenado con la muestra | 0,979 | 0,962–0,992 |
| Tabular con centroide (S3-C), entrenado con todo DR25 | 0,989 | 0,978–0,996 |
| `koi_score` del Robovetter (comparador externo, nunca entrada) | 0,996 | 0,986–1,000 |
| Demo vuelta a medir: solo curva | 0,821 | 0,769–0,871 |
| Demo vuelta a medir: imagen + curva | 0,939 | 0,909–0,965 |

El listón tabular con centroide es muy alto. Dos advertencias: en periodo corto la revisión de Kepler ya funciona bien, y parte de las etiquetas de falso positivo se asignaron usando el centroide (circularidad parcial).

### 5.2 Pregunta 1 — Arquitecturas

| Modelo | AUC (media de 3 semillas) | IC 95 % | Parámetros |
|---|---|---|---|
| Rama 1 · CNN tipo AstroNet (curva plegada) | 0,864 | 0,823–0,901 | 57.747 |
| Rama 2 · tránsitos + promedio | 0,863 | 0,817–0,905 | 5.027 |
| Rama 2 · tránsitos + GRU | 0,862 | 0,817–0,900 | 18.723 |
| **Rama 2 · tránsitos + atención** | **0,883** | 0,842–0,921 | 13.604 |
| Rama 4 · escalares (con centroide) | 0,948 | 0,914–0,976 | 1.059 |

| Diferencia de AUC (pareada) | Valor | IC 95 % | ¿Se distingue de 0? |
|---|---|---|---|
| Atención − AstroNet | +0,019 | −0,025 a +0,062 | No |
| Atención − promedio | +0,020 | −0,014 a +0,055 | No |
| GRU − atención | −0,022 | −0,046 a −0,000 | Sí (por poco) |
| GRU − promedio | −0,001 | −0,027 a +0,023 | No |

- **Sesgos inductivos.** Al barajar el orden de los tránsitos en la prueba, promedio y atención no cambian (cambio 0,000) y la GRU sí (su probabilidad cambia 0,048 en promedio y su AUC baja 0,005). La GRU usa el orden, pero eso no le dio ventaja.
- **Elección para la fusión.** La atención tuvo la menor pérdida en validación interna (0,457 frente a 0,506 del promedio y 0,475 de la GRU), por más de un error estándar: pasa a la fusión. La decisión no usó datos de prueba.
- **Interpretación.** Con decenas o cientos de tránsitos por estrella, promediar ya recoge casi toda la información. Que la atención ayude con *pocos* tránsitos es la hipótesis de la tesis para la zona habitable; **estos datos no permiten evaluarla**.

### 5.3 Pregunta 2 — Fusión y ablación

| Sistema | AUC (media de 5 semillas de cabeza) | IC 95 % |
|---|---|---|
| **Fusión de 4 ramas** | **0,985** | 0,974–0,994 |
| Fusión tardía (promedio de probabilidades) | 0,976 | 0,958–0,991 |
| Sin rama 1 (vistas) | 0,978 | 0,963–0,989 |
| Sin rama 2 (tránsitos) | 0,984 | 0,971–0,993 |
| Sin rama 3 (imagen v0) | 0,981 | 0,968–0,992 |
| Sin rama 4 (escalares) | 0,969 | 0,950–0,985 |
| Solo rama 1 / 2 / 3 / 4 | 0,913 / 0,894 / 0,909 / 0,954 | — |

- **La fusión supera a la mejor rama sola** (escalares): +0,032 [+0,010 a +0,057]. Y a la demo vuelta a medir: +0,052 [+0,027 a +0,079].
- **Qué ramas aportan de forma distinguible:** vistas (+0,007 [+0,002 a +0,014]) y escalares (+0,016 [+0,005 a +0,031]). Quitar la rama de tránsitos (+0,002) o la de imagen (+0,004) **no se distingue de cero**: lo que aportan ya lo cubren las otras en este régimen.
- **Frente al listón tabular (S3-C: 0,979–0,989)** la fusión está en el mismo rango. No se puede afirmar que lo supere.
- Exactitud de 3 clases: 0,918. Matriz de confusión (filas = real): planeta [119, 7, 4], falso positivo [8, 101, 6], sin señal [0, 15, 230].
- **El error típico de la demo se corrigió:** `K00022.01`, que la demo llamaba falso positivo con 0,73, ahora sale planeta con 0,95.
- **Alerta de apilado.** La cabeza de fusión se entrena con vectores de ejemplos que los codificadores ya conocían; su pérdida en validación es 3,6 veces la de entrenamiento. Entrenarla solo con vectores "no vistos" quita la alerta (razón 1,4) y da un AUC de 0,975, no distinguible. La consecuencia práctica es sobreconfianza, que se mide y corrige en la pregunta 3.

### 5.4 Pregunta 3 — Calibración e incertidumbre

**Temperature scaling** (temperaturas por partición: 1,06 · 1,30 · 1,18 · 1,93 · 1,14; todas mayores que 1 = sobreconfianza):

| Medida | Antes | Después | Diferencia [IC 95 %] |
|---|---|---|---|
| ECE | 0,033 | 0,022 | −0,011 [−0,018 a +0,005] |
| NLL | 0,206 | 0,188 | −0,018 [−0,046 a +0,001] |
| AUC | 0,985 | 0,985 | no cambia |

Mejora en la dirección esperada; **no se distingue de cero** con este tamaño de muestra.

**Prior explícito.** La tasa de planetas en entrenamiento es 0,53; la medida en DR25 para periodos de 1–30 d es 0,453 (2.060 confirmados y 2.487 falsos positivos).

| Tasa base declarada | ECE sin corregir | ECE corregido | ¿Mejora distinguible? |
|---|---|---|---|
| 0,5 | 0,048 | 0,052 | No |
| 0,453 (DR25) | 0,051 | 0,050 | No |
| 0,2 | 0,082 | 0,041 | Sí |
| 0,1 | 0,099 | 0,030 | Sí |
| 0,05 | 0,108 | 0,021 | Sí |
| 0,02 | 0,114 | 0,010 | Sí |

Con tasa 0,05, las señales con probabilidad de planeta mayor que 0,5 pasan de 132 a 94. El orden de los candidatos casi no cambia (τ de Kendall 0,98). **Límite:** la corrección supone que solo cambia la tasa base y no el aspecto de las señales; aquí se demuestra el mecanismo, no su validez en zona habitable.

**Incertidumbre.**

| Método | NLL | ECE | Detección de errores (AUROC) |
|---|---|---|---|
| Una sola cabeza | 0,206 | 0,033 | 0,925 |
| MC Dropout (50 pasadas) | 0,186 | 0,026 | 0,942 |
| Ensamble de 5 semillas | 0,189 | 0,019 | 0,889 |

Los errores tienen mucha más entropía que los aciertos (0,60 frente a 0,10 en el ensamble). Entre MC Dropout y ensamble **no hay diferencia distinguible** (NLL: +0,003 [−0,019 a +0,034]). La incertidumbre solo cubre la cabeza de fusión.

## 6. Qué significa y qué no

**Lo que el trabajo muestra**

- Un sistema de cuatro ramas, entrenado y evaluado sin fugas, que alcanza AUC 0,985 y mejora de forma medible a sus partes y a la demo.
- Un diseño controlado para comparar arquitecturas, con su respuesta honesta: en periodo corto, la atención no gana de forma demostrable.
- Que declarar la tasa base explícitamente corrige la calibración cuando los planetas son escasos.
- Que la incertidumbre del modelo sirve para saber cuándo no confiar en él.

**Lo que no se puede afirmar**

- Que la fusión supere al modelo tabular con centroide.
- Que la atención sea mejor que promediar.
- Nada sobre el régimen de zona habitable: todas las señales son de periodo corto, con muchos tránsitos.

**Limitaciones declaradas**

1. Periodo corto (1–30 d) y solo 245 señales: intervalos anchos.
2. Solo señales reales; sin inyecciones ni falsas alarmas reconstruidas.
3. La clase *sin señal* es construida, y en ella el centroide siempre falta (pista fácil para la rama 4; no afecta al AUC planeta vs FP).
4. Circularidad parcial entre el centroide y las etiquetas de falso positivo.
5. Apilado: cabeza sobreconfiada, medida y corregida con calibración.
6. La rama de imagen es la v0 de la demo y cubre 224 de 245 estrellas.
7. La incertidumbre solo cubre la cabeza.
8. El prerregistro usó una definición operativa (disposición distinta entre DR25 y la tabla acumulada) y marcó 0 señales.

## 7. Qué se construyó

| Parte | Dónde |
|---|---|
| Plan, decisiones y protocolo | `docs/planeacion/04_plan_ejecucion_deep_learning.md` |
| Arquitectura del repositorio | `docs/ARQUITECTURA.md` |
| Especificaciones por paquete de trabajo | `specs/000` a `specs/008` |
| Base de datos y verificación del contrato | `src/comun/` |
| Extractor temporal y ramas 1, 2 y 4 | `src/temporal/` |
| Imagen v0, fusión, calibración, incertidumbre | `src/fusion/` |
| Scripts originales de la demo, sin modificar | `src/demo/` |
| 73 pruebas automáticas (71 rápidas y 2 que entrenan) | `tests/` |
| Notebook ilustrativo e interactivo | `notebooks/informe_dl.ipynb` |
| Tablas con todas las cifras | `resultados/dl/tablas/` |
| Registro de cada sesión y de cada desviación del plan | `docs/planeacion/BITACORA_DL.md` |

### Desviaciones del plan (todas anotadas en la bitácora)

- Particiones con un reparto propio por SHA-256, porque `StratifiedGroupKFold` cambiaba según la versión de scikit-learn.
- Las cadencias de otros KOIs se ocultan en vez de descartar la ventana entera (la regla original dejaba sin tránsitos a 37 ejemplos).
- Las ventanas individuales se escalan por el mayor entre la profundidad y el ruido por punto.
- Descarga directa de MAST por HTTPS (sin `lightkurve`) y calidad `SAP_QUALITY = 0`.
- No se hicieron los experimentos de prioridad B y C: menos tránsitos (E4), pooling (E5), BiGRU sobre vistas, rama 2 sin estadísticos de consistencia, ajuste fino conjunto y ensamble de codificadores.

## 8. Cómo abrir el notebook y reproducir

```bat
scripts\abrir_notebook.bat          :: instala lo necesario y abre notebooks\informe_dl.ipynb
scripts\reproducir_todo.bat         :: vuelve a correr todo, desde las tablas hasta el notebook
```

El notebook ya trae las figuras incrustadas. Para usar el **explorador interactivo** (sección 7 del notebook) hay que ejecutarlo: "Run All".

## 9. Lo que sigue

- Confirmar en el PC que `pytest -q` y `pytest -q -m lenta` pasan (allí hay Keras 3.15 y Python 3.13; se desarrolló con Keras 3.12 y Python 3.10).
- Revisar este documento contra la rúbrica oficial cuando llegue.
- Redactar el informe y la presentación (spec 008) a partir de estas tablas.
- Después de las entregas: unir con el curso de Visión (cambiar la imagen v0 por su codificador) y pasar al régimen de periodo largo, donde la atención y el prior explícito tienen su prueba real.
