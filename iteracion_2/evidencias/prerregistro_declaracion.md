# Evaluación única de las señales prerregistradas — declaración previa

**Fecha:** 22 de noviembre de 2026. **Escrita y versionada antes de ejecutar la evaluación.**
Autores: Thomas Monnier Granda, Esteban Felipe Genes Noya, Juan Sebastian Higuita Riascos,
Alexandra Arias Mazo y Eliel Segura Perez (curso de Deep Learning).

## Qué son

76 señales (KOIs) de la selección de la iteración 2 cuya disposición difiere entre el catálogo DR25
y la tabla acumulada del NASA Exoplanet Archive. Se apartaron antes de entrenar: no entraron al
entrenamiento, a la validación, a la búsqueda de hiperparámetros ni a ninguna cifra reportada.

Al abrirlas (solo sus etiquetas, sin ninguna predicción) se encontró que **las 76 son FALSE POSITIVE
en DR25 y CANDIDATE en la tabla acumulada**. Ninguna está confirmada.

## Qué se puede medir y qué no

No hay respuesta correcta: «candidato» significa que la señal se reabrió, no que sea un planeta.
Por eso **no se calcula AUC de planeta frente a falso positivo ni aciertos** sobre estas señales, y
el resultado no se presentará como validación del clasificador ni como prueba de calibración.

Lo que sí se puede preguntar: el sistema se entrenó con etiquetas DR25, donde estas señales serían
falsos positivos. ¿Les asigna, aun así, más probabilidad de planeta que a los falsos positivos que
nadie ha reabierto?

## Hipótesis y estadístico (fijados ahora)

- **Modelo:** la fusión principal de tres ramas (`tres_ramas`), con los codificadores de
  `iteracion_2/modelos_finales/` y las cabezas de la semilla 0. Cada señal la predice el sistema de
  **su** partición (asignada por estrella), que nunca vio esa estrella.
- **Puntaje:** q = P(planeta) / (P(planeta) + P(falso positivo)), del ejemplo con tránsito.
- **Grupo de comparación:** los falsos positivos DR25 no prerregistrados, con sus predicciones fuera
  de muestra de la misma fusión y la misma semilla.
- **Estadístico:** A = probabilidad de que una señal prerregistrada reciba un q mayor que un falso
  positivo normal (área bajo la curva ROC entre los dos grupos), con intervalo del 95 % por
  bootstrap por estrella (2.000 remuestreos, semilla del protocolo).
- **Hipótesis:** A > 0,5. Se considera respaldada si el intervalo queda entero por encima de 0,5;
  refutada en ese sentido si queda entero por debajo; no concluyente si contiene 0,5.
- **Descriptivos que se reportan en cualquier caso:** mediana de q en cada grupo, fracción con
  q > 0,5 en cada grupo, y lo mismo por tipo de falso positivo DR25. Para los ejemplos «sin señal»
  de las 76, la fracción clasificada como sin señal (comprobación, no resultado).

## Reglas

1. Antes de tocar las prerregistradas, el script comprueba que reproduce las predicciones ya
   publicadas de ejemplos normales de prueba. Si no las reproduce, se detiene sin evaluar.
2. Se ejecuta **una sola vez**. Si el archivo de resultados existe, el script se niega a repetir.
3. Se reporta salga lo que salga, con el mismo rigor.
4. Una señal sin datos suficientes en alguna rama se evalúa con las ramas que tenga (su bandera de
   disponibilidad en cero), igual que cualquier otro ejemplo; se informa cuántas son.
