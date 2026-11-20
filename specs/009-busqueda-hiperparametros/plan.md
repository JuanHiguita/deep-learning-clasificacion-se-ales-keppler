# Plan · Spec 009

## Diseño

| Pieza | Dónde | Qué hace |
|---|---|---|
| Núcleo | `src/comun/busqueda.py` | Muestreo reproducible, protocolo con hiperparámetros cambiados, selección anidada, resumen con intervalos |
| Ramas | `src/temporal/busqueda_ramas.py` | Espacios de las ramas 1, 2 y 4; ejecuta con `entrenar_cv` (semilla 0) |
| Cabeza | `src/fusion/busqueda_cabeza.py` | Espacio de la cabeza de fusión; usa los vectores congelados v1/v0 |
| Modelos | `src/temporal/modelos.py`, `src/fusion/fusionar.py` | Parámetros nuevos (`ancho`, `unidades_gru`, `cabezas`, `oculta`) con valores por defecto iguales al protocolo |
| Figuras | `src/informe/exploracion.py`, `src/informe/experimentacion.py` | Una función por figura de los notebooks |

## Espacios de búsqueda

Comunes: tasa de aprendizaje log-uniforme en [3e-4, 3e-3]; decaimiento de pesos ∈ {0, 1e-5, 1e-4, 1e-3}; lote ∈ {16, 32, 64}; dropout ∈ {0,1, 0,3, 0,5}.

| Modelo | Además |
|---|---|
| Rama 1 · AstroNet | `ancho` ∈ {0,5, 1}; pooling ∈ {máximo, promedio}; lote ∈ {32, 64} (límite de cómputo: una partición debe caber en una corrida) |
| Rama 2 · promedio | `ancho` ∈ {0,5, 1, 2} (filtros del codificador por tránsito) |
| Rama 2 · GRU | `ancho`; unidades de la GRU ∈ {16, 32, 64} |
| Rama 2 · atención | `ancho`; cabezas de atención ∈ {1, 2, 4} |
| Rama 4 · escalares | dropout ∈ {0,1, 0,2, 0,4}; neuronas ocultas ∈ {16, 32, 64} |
| Cabeza de fusión | neuronas ocultas ∈ {32, 64, 128} |

## Decisiones

- **D-1 Búsqueda aleatoria**, no rejilla: con 5–6 dimensiones y 7 sorteos cubre más valores distintos de cada hiperparámetro (Bergstra y Bengio, 2012).
- **D-2 Selección anidada por partición**, no una configuración global: una elección global usaría, a través de otra partición, las etiquetas de la prueba de cada modelo.
- **D-3 No se reexporta nada automáticamente.** Si la búsqueda no mejora de forma distinguible, el protocolo fijo se mantiene como resultado principal y la búsqueda se reporta como análisis de sensibilidad. Si mejora, se abre una spec para rehacer fusión y calibración.

## Seguridad

Sin entradas externas nuevas. Las rutas llegan por argumento. `np.load(..., allow_pickle=False)`.
