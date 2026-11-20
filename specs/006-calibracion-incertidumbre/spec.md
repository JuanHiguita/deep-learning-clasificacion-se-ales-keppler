# 006 · Calibración e incertidumbre — especificación

**PT6 del plan · Capa 4 · Dueño: Deep Learning · Responde la pregunta 3**

## Para qué

Saber si las probabilidades del sistema se pueden tomar al pie de la letra, cómo cambian al declarar una tasa base distinta y si la incertidumbre avisa de los errores.

## Entradas

`resultados/dl/fusion/e6a/pred_semilla<s>.npz` (5 semillas de cabeza) con `p_prueba` y `p_validacion`. El "modelo del sistema" para E8 y E9 es la **semilla 0**.

## Requisitos funcionales

### E8 · Temperature scaling

- **RF-1** Para cada partición *k*, la temperatura `T_k` minimiza el NLL de `p_validacion` de los ejemplos cuya validación interna corresponde al modelo de *k*. Se optimiza `log T` con `scipy.optimize.minimize_scalar` acotado a `T ∈ [0,05; 20]`. Se aplica a `p_prueba` de la partición *k*: `softmax(log p / T)`.
- **RF-2** Se reporta antes y después: NLL, Brier, ECE *top-label* (15 intervalos), ECE binario sobre `q`, exactitud y AUC (no deben cambiar), con intervalos; y la diferencia pareada después − antes.
- **RF-3** Figura de confiabilidad antes y después (frecuencia observada frente a confianza, con el número de ejemplos por intervalo). Las 5 temperaturas en `metricas.json`.

### E9 · Prior explícito

- **RF-4** Se trabaja sobre `q` calibrado, solo en ejemplos `transito`. `π_ent` = fracción de planetas entre los ejemplos `transito` de entrenamiento de cada partición.
- **RF-5** Tasas objetivo: (i) `π_dr25`: confirmados / (confirmados + FP) en DR25 con periodo entre 1 y 30 d, **calculada por el script** desde `data/koi_dr25.csv`; (ii) rejilla `[0,5; 0,2; 0,1; 0,05; 0,02]`.
- **RF-6** Para cada tasa: `q' = corregir_prior(q, π_ent, π)`; la prueba se pondera con `pesos_para_tasa_base`. Se reporta ECE binario ponderado y NLL binario ponderado, **con y sin** corrección, y su diferencia con intervalo (el *bootstrap* recalcula los pesos en cada remuestreo).
- **RF-7** Efecto sobre el orden: `q'` es monótona en `q`, así que el orden por `q` no cambia. Lo que cambia es (a) cuántos candidatos superan un umbral de 0,5 y de 0,9, y (b) el orden por el producto con otra probabilidad. Se reporta (a) por tasa. Para (b) se usa el orden de los 3 tipos de salida: `p0` de 3 clases frente a `q'·(1 − p2)`; τ de Kendall y cambios en los 20 primeros.
- **RF-8** El informe declara: la corrección supone que solo cambia la tasa base; no se valida en zona habitable; la cifra de 1,8 % de `00_ESTADO` §5 no se usa como prior porque su denominador incluye candidatos.

### E10 · Incertidumbre

- **RF-9** **MC Dropout:** 50 pasadas de la cabeza de la semilla 0 con el Dropout activo (`training=True`); probabilidad = media. **Ensamble:** media de las 5 semillas de cabeza.
- **RF-10** Para la cabeza determinista (semilla 0), MC Dropout y ensamble: NLL, Brier, ECE; entropía predictiva; AUROC de detección de errores (¿la entropía separa aciertos de errores?); curva riesgo-cobertura (error entre los ejemplos retenidos al ordenar por confianza) y su área (AURC). Diferencias pareadas ensamble − MC Dropout.
- **RF-11** Para MC Dropout y ensamble, además, la información mutua (desacuerdo entre pasadas o miembros), como medida de incertidumbre del modelo.
- **RF-12** El informe declara que la incertidumbre solo cubre la cabeza: los codificadores están congelados.
- **RF-13** **E10b (C):** ensamble de 3 semillas de codificadores propios, cada uno con su cabeza.

### E11 · Análisis de errores

- **RF-14** Con semilla 7: 10 ejemplos al azar entre los mal clasificados y 10 entre los bien clasificados (estratificado por clase cuando haya suficientes). Para cada uno: sus entradas (vista local de 4 canales, ventanas por tránsito con pesos de atención, escalares), las probabilidades de cada rama sola y de la fusión, y su incertidumbre.
- **RF-15** Se informa explícitamente qué pasó con K00022.01 (el error típico de la demo), sea cual sea el resultado. No se elige ningún otro caso a mano.

## Criterios de aceptación

- [ ] Prueba unitaria: `ajustar_temperatura` recupera T = 2 en datos sintéticos generados con esa temperatura (tolerancia 5 %).
- [ ] Prueba unitaria: la función que ajusta la temperatura no recibe ni puede leer `p_prueba` (firma sin ese argumento).
- [ ] Prueba unitaria: MC Dropout con Dropout = 0 da varianza 0.
- [ ] `resultados/dl/tablas/p3_calibracion.csv` y `p3_incertidumbre.csv`, figuras de confiabilidad y de riesgo-cobertura.
- [ ] Ninguna cifra de este módulo se escribe a mano.
