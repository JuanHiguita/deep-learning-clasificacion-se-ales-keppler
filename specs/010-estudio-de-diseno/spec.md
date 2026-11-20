# Spec 010 · Estudio de decisiones de diseño y experimentos pendientes

**Estado:** hecha (20-nov-2026) · **Origen:** pedido de Juan (20-nov-2026): ¿se probaron otras funciones de activación, normalizadores, optimizadores, número de capas, preprocesamiento y funciones de pérdida?

## Problema

Hasta la spec 009 solo se variaron hiperparámetros numéricos (tasa de aprendizaje, decaimiento, lote, dropout, ancho, unidades, cabezas, pooling). Las decisiones **de diseño** (ReLU, sin normalización, AdamW, profundidad, entropía cruzada con pesos, normalización por profundidad, aumentación) se fijaron por criterio y nunca se contrastaron.

## Requisitos funcionales

- **RF-1 Una cosa a la vez.** Cada variante cambia un solo factor respecto al protocolo. La referencia es el experimento publicado (semilla 0), copiado, no reentrenado.
- **RF-2 Variantes declaradas antes de entrenar** en `src/temporal/estudio_diseno.py`:

| Factor | Variantes |
|---|---|
| Activación | GELU, tanh (protocolo: ReLU) |
| Normalización | por lote (BatchNorm), por capa (LayerNorm) (protocolo: ninguna) |
| Optimizador | Adam sin decaimiento, SGD con momento de Nesterov y tasa ×10 (protocolo: AdamW) |
| Profundidad | una capa o bloque menos, una capa o bloque más |
| Pérdida | sin pesos por clase, suavizado de etiquetas 0,1, focal γ = 2 (protocolo: entropía cruzada con pesos por clase) |
| Preprocesamiento | sin aumentación; cada serie en *z-score* en vez de unidades de profundidad (ramas 1 y 2); sin estandarizar (rama 4) |
| Pooling (E5 del plan) | promedio, convolución con paso 2 (solo rama 1; protocolo: máximo) |
| Entradas (E2b del plan) | rama 2 sin los 4 estadísticos de consistencia (los tres agregadores) |

- **RF-3 Modelos:** rama 1 (AstroNet), rama 2 con atención y rama 4. Para promedio y GRU solo E2b.
- **RF-4 Mismo motor**, mismas particiones, semilla 0, parada temprana por validación interna.
- **RF-5 Se reporta por variante:** NLL de validación, AUC, NLL y ECE de prueba, y la diferencia pareada de AUC con la referencia con su intervalo del 95 %.
- **RF-6 E4 del plan (menos tránsitos):** con los modelos de E2b, conservar 32, 16, 8, 4 y 2 tránsitos al azar en prueba (semilla fija, 5 repeticiones) y medir AUC por agregador y la diferencia atención − promedio y atención − GRU.
- **RF-7 No se rellena un dato faltante con 0 sin aviso** en ninguna variante (regla del proyecto): no existe la variante "sin banderas de faltante".

## Cómo se interpreta (fijado antes de ver resultados)

- Son **~15 comparaciones por modelo con una semilla**. Con ese número, que una o dos "se distingan de cero" es esperable por azar. No se cambia el protocolo por una sola variante favorable.
- Una variante solo motivaría cambiar el protocolo si mejora de forma distinguible **y** se confirma con las 3 semillas. Esa confirmación sería una spec nueva.
- El resultado principal del proyecto no cambia con este estudio.

## Fuera de alcance

- E1b (BiGRU sobre las vistas), E6b (ajuste fino conjunto), E7b (dropout de ramas) y E10b (ensamble de codificadores): no se hacen por costo de cómputo. Se declaran como no realizados.
- La segunda mitad de E5 (robustez a desplazar la vista en prueba).
- Combinaciones de factores (interacciones).

## Criterios de aceptación

- `resultados/dl/diseno/<modelo>/variantes.csv` para las ramas 1, 2 (atención) y 4; `resultados/dl/e4/e4_menos_transitos.csv`.
- Los valores por defecto de los modelos siguen dando los mismos conteos de parámetros y las mismas predicciones publicadas.
- Tabla consolidada `resultados/dl/tablas/diseno.csv` y notebook `03_estudio_de_diseno.ipynb`.
