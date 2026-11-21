# Spec 013 · Copia experimental: sistema con la rama 1 mejorada (v2)

**Estado:** hecha (20-nov-2026) · **Origen:** decisión de Juan: adoptar una variante de la rama 1, pero como copia experimental.

## Qué es

Una segunda versión del sistema en la que **solo cambia la rama 1**. Todo lo demás (ramas 2, 3 y 4, cabeza, protocolo de entrenamiento, particiones, semillas) es idéntico. Vive aparte:

- `modelos/vistas/v2/` — codificador de la rama 1 con la variante elegida.
- `resultados/dl/experimental_v2/` — fusión, calibración, incertidumbre y comparación.

**El resultado principal no se toca:** `modelos/vistas/v1`, `resultados/dl/fusion`, `calibracion`, `incertidumbre` y `tablas` quedan como estaban.

## Cómo se eligió la variante

Regla fijada antes de correr: la variante de la rama 1 con **menor NLL de validación interna** (semilla 0) entre las 15 del estudio de diseño. Es **tanh en vez de ReLU** (0,275 frente a 0,385 del protocolo).

Advertencia: cuando se eligió ya se conocían sus resultados de prueba (spec 010). La regla usa solo la validación, pero la decisión de hacer este experimento no fue ciega. Por eso es una copia experimental y no el resultado principal.

## Requisitos

- **RF-1** El codificador v2 se exporta con el mismo motor y debe reproducir las predicciones de la variante ya entrenada (diferencia 0).
- **RF-2** Fusión de 4 ramas con 5 semillas de cabeza, temperatura por partición y MC Dropout frente a ensamble, igual que el principal.
- **RF-3** Comparación pareada v2 − v1 (AUC, NLL, ECE, exactitud) con intervalo del 95 % por *bootstrap* de estrellas: `comparacion.csv`.
- **RF-4** Después de correr, `git diff` de los resultados principales debe estar vacío.

## Fuera de alcance

Ablación, ajuste cruzado, prior explícito y empaquetado del sistema v2 para inferencia.
