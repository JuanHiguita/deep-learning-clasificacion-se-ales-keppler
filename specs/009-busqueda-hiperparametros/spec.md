# Spec 009 · Búsqueda de hiperparámetros y notebooks exploratorios

**Estado:** hecha (20-nov-2026) · **Origen:** pedido de Juan (20-nov-2026): argumentar el porqué de cada arquitectura y mostrar el proceso de experimentación.

## Problema

El plan fijó los hiperparámetros de antemano (DD del plan, `config/protocolo.json`) para comparar arquitecturas con el mismo presupuesto. Eso deja dos preguntas sin responder:

1. ¿Cuánto mejoraría cada modelo si se ajustaran sus hiperparámetros?
2. ¿Las conclusiones de la pregunta 1 (atención frente a promedio, GRU y AstroNet) dependen de la configuración elegida?

## Requisitos funcionales

- **RF-1 Espacio declarado antes de entrenar.** Los espacios viven en `src/temporal/busqueda_ramas.py` y `src/fusion/busqueda_cabeza.py`. No se modifican después de ver resultados.
- **RF-2 Mismo presupuesto por arquitectura.** 8 configuraciones cada una: la del protocolo (configuración 0) y 7 sorteadas con semilla fija (`20261120`).
- **RF-3 La prueba no decide nada.** Selección **anidada**: el modelo de la partición *k* elige su configuración con su validación interna (partición *k + 1*). La partición de prueba *k* no interviene.
- **RF-4 Reanudable.** Cada partición entrenada se guarda; una corrida interrumpida continúa donde quedó.
- **RF-5 La configuración 0 es el experimento publicado.** Se copia de `resultados/dl/<modelo>` (se comprobó que reentrenarla da predicciones idénticas).
- **RF-6 Se reportan las dos versiones:** protocolo fijo y con búsqueda, con intervalo del 95 % por *bootstrap* de estrellas y su diferencia pareada.
- **RF-7 Notebooks exploratorios.**
  - `notebooks/01_exploracion_datos.ipynb`: qué muestran los datos y por qué motivan cada rama.
  - `notebooks/02_experimentacion_hiperparametros.ipynb`: el proceso de búsqueda, con una celda que entrena de verdad (reutiliza lo ya entrenado).
- **RF-8 Embeddings sin cambio.** La dimensión de los vectores (32/32/32/16) la fija el contrato y no se busca.

## Fuera de alcance

- Volver a exportar los codificadores con la configuración encontrada y rehacer la fusión, la calibración y la incertidumbre. Se decide después de ver si la búsqueda mejora de forma distinguible (ver `plan.md`, decisión D-3).
- Búsqueda con más de una semilla.

## Riesgos y cómo se declaran

- **Optimismo de selección:** la validación interna también decide la parada temprana. La NLL de validación "con búsqueda" es optimista; la comparación válida es la de prueba.
- **Una semilla:** la variación entre semillas (0,007–0,015 de AUC en las ramas) es del orden de las diferencias esperadas. Un resultado no distinguible no prueba que no haya mejora.
- **Espacio pequeño (7 sorteos):** es una exploración, no una optimización exhaustiva. Así se redacta.
- **Excepción a la regla "los notebooks no entrenan":** vale solo para el notebook 02, que es de experimentación. `informe_dl.ipynb` sigue sin entrenar.

## Criterios de aceptación

- `pytest -q` pasa, con pruebas del muestreo y de la selección anidada.
- `resultados/dl/busqueda/<arquitectura>/{configuraciones.csv,resumen.json}` existen para las 5 ramas y la cabeza.
- Los dos notebooks se ejecutan de principio a fin sin errores.
- Ninguna cifra de los notebooks está escrita a mano.
