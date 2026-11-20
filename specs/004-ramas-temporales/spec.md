# 004 · Ramas y comparación de arquitecturas — especificación

**PT4 del plan · Capa 3 · Dueño: Deep Learning · Responde la pregunta 1**

## Para qué

Entrenar las ramas 1, 2 y 4, y contestar con un diseño controlado qué forma de leer la señal de tránsito funciona mejor y qué supuesto mete cada una.

## Modelos

Todos terminan en `Dropout` → `Dense(dim, activation="relu", name="embedding")` → `Dense(3, softmax)`.

| ID | Experimento (carpeta) | Entradas | Arquitectura |
|---|---|---|---|
| M0 | `m0_demo_curva`, `m0_demo_completo` | `seq` (61, 2) e `img` de la demo | Ramas de `src/demo/demo_cnn.py` sin cambios de arquitectura, con el motor y las particiones comunes |
| M1 | `m1_astronet` | `vista_global`, `vista_local` | Global: 3 × [Conv1D(f, 5, same, relu) ×2 → MaxPool(5, 2)] con f = 16, 32, 64 → GlobalMaxPool. Local: 2 × [Conv1D(f, 5) ×2 → MaxPool(7, 2)] con f = 16, 32 → GlobalMaxPool. Concat → Dense(64) → embedding(32) |
| M1r | `m1r_bigru` (prioridad B) | `vista_local` | Bidirectional(GRU(32)) → embedding(32) |
| M2p | `m2_promedio` | `transitos`, `mascara_transitos`, `consistencia` | φ → media enmascarada → concat consistencia → embedding(32) |
| M2g | `m2_gru` | ídem | φ → Bidirectional(GRU(32)) con máscara → concat → embedding(32) |
| M2a | `m2_atencion` | ídem | φ → bloque de autoatención → *attention pooling* → concat → embedding(32) |
| M4 | `m4_escalares` | `escalares`, `faltantes` | concat → Dense(32) → Dropout(0,2) → embedding(16) |

**φ (codificador por tránsito, pesos compartidos):** `TimeDistributed` de [Conv1D(16, 5, same, relu) → Conv1D(32, 5, same, relu) → GlobalMaxPool → Dense(32, relu)] sobre `(32, 61, 1)`.

**Bloque de autoatención:** `MultiHeadAttention(num_heads=2, key_dim=16)` con máscara de tránsitos → suma residual → `LayerNormalization` → Dense(64, relu) → Dense(32) → suma residual → `LayerNormalization`. Sin codificación posicional (decisión deliberada: invariancia al orden).

**Attention pooling:** puntuación `Dense(1)` por tránsito, −∞ donde la máscara es 0, softmax, suma ponderada. Los pesos se exponen como segunda salida de un submodelo `pesos_atencion`.

## Experimentos

| ID | Qué | Prioridad | Responde |
|---|---|---|---|
| E0 | Pruebas de cordura (spec 003) para M1, M2a y M4 | A | — |
| E1 | M0, M1 y M4 · 3 semillas × 5 particiones | A | P1a |
| E2 | M2p, M2g y M2a · 3 semillas × 5 particiones | A | P1a, P1b, P1c |
| E3 | Permutación del orden de los tránsitos en prueba: 10 permutaciones por ejemplo, sobre los modelos de E2 | A | P1c |
| E4 | Menos tránsitos en prueba: 32, 16, 8, 4, 2 (al azar, semilla fija, balanceando paridad), sobre los modelos de E2 | B | P1b |
| E5 | Pooling en la torre local de M1: máximo / promedio / Conv1D con paso 2. Y robustez: desplazar `vista_local` 0, 2, 5 y 10 puntos en prueba | B | sesgos |
| E2b | M2a sin `consistencia` | B | sesgos |
| E1b | M1r | B | P1 (misma entrada, otra arquitectura) |

## Requisitos funcionales

- **RF-1** Aumentación (solo entrenamiento): ruido gaussiano σ = 0,05; desplazamiento circular de ±2 puntos en `vista_local` y en cada ventana; en la rama 2, conservar cada tránsito con probabilidad 0,8 (mínimo 4) actualizando la máscara. La GRU no recibe permutaciones.
- **RF-2** En E3 se reporta, por modelo: cambio medio absoluto de `q` por ejemplo y cambio de AUC. Para M2p y M2a debe ser 0 hasta la precisión numérica (1e-5); si no, es un error de implementación.
- **RF-3** En E4 la profundidad de `consistencia` se recalcula con los tránsitos conservados.
- **RF-4** **Selección del agregador de la rama 2 v1:** menor NLL medio de `p_validacion` (15 valores: 5 particiones × 3 semillas). Si otro más simple queda a menos de un error estándar, gana el más simple (promedio < atención < GRU). Se escribe `resultados/dl/seleccion_rama2.json` con los 3 valores, sus errores estándar y la decisión. **No se abre `p_prueba` para decidir.**
- **RF-5** Exportación v1 (semilla 0) de `vistas` (M1), `transitos` (el elegido) y `escalares` (M4) a `modelos/`.
- **RF-6** Tabla comparativa `resultados/dl/tablas/p1_arquitecturas.csv`: por modelo, AUC, NLL, ECE, exactitud, parámetros, tiempo por época, con intervalos; y diferencias pareadas: M2* − M1, M2a − M2p, M2g − M2a, M1 − M0.
- **RF-7** Para 12 ejemplos al azar (semilla 7, 4 por clase) se guardan los pesos de atención de M2a con las ventanas, para el informe.

## Criterios de aceptación

- [ ] E0 pasa en las tres ramas.
- [ ] E1, E2 y E3 completos, con manifiestos.
- [ ] RF-2 se cumple para M2p y M2a.
- [ ] `verificar_contrato.py` pasa con `modelos/{vistas,transitos,escalares}/v1`.
- [ ] Etiqueta git `ramas-v1`.
- [ ] Si algún AUC supera 0,99: detenerse y auditar.

## Advertencias para el informe

- Los pesos de atención describen, no explican (Jain y Wallace, 2019).
- E4 no permite concluir nada sobre zona habitable: son señales de periodo corto con tránsitos quitados.
- Si la atención no supera al promedio, se reporta así. Es el resultado esperable con decenas de tránsitos por estrella.
