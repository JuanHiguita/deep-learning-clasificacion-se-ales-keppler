# Spec 011 · Sistema conjunto, prueba de extremo a extremo y auditoría

**Estado:** hecha (20-nov-2026) · **Origen:** pedido de Juan (20-nov-2026).

## Requisitos funcionales

- **RF-1 Respuesta conjunta.** `fusion/sistema.py` encadena los 4 codificadores congelados, la cabeza de fusión, la temperatura y el prior explícito, y entrega para cada señal: probabilidades calibradas de las 3 clases, P(planeta | hay señal), su versión con la tasa base declarada, la entropía y qué ramas tuvieron dato.
- **RF-2 Sin datos de entrenamiento en la inferencia.** `empaquetar` guarda en `modelos/sistema/` la estandarización de la entrada de la cabeza, la temperatura y la tasa de entrenamiento de cada partición, y las cabezas, con huellas SHA-256.
- **RF-3 Sin fuga en el uso.** Una estrella del conjunto se responde con el sistema de **su** partición. Una estrella nueva, con el promedio de los cinco.
- **RF-4 Prueba de extremo a extremo** (`scripts/prueba_e2e.py`): desde los FITS de Kepler hasta la probabilidad calibrada, comparando con lo publicado en tres puntos (entradas, vectores, probabilidad), para estrellas elegidas al azar con semilla fija.
- **RF-5 Auditoría** (`src/fusion/auditoria.py`): comprobaciones automáticas de fugas, particiones, reproducibilidad y coherencia entre documentos y tablas; informe en `docs/AUDITORIA.md`.

## Límites declarados

- **La imagen (rama 3) no se vuelve a derivar** en la prueba: se toma de `data/dataset_demo.npz`, que produce el script de la demo sin modificar.
- **Lo que la tesis añade y aquí no existe:** inyecciones, falsas alarmas reconstruidas, y la combinación con P(en zona habitable y rocoso) calculada con Gaia. El sistema de este curso llega hasta la probabilidad calibrada con prior explícito, en periodo corto.
- **Rango de validez:** señales de Kepler de 1 a 30 días.

## Seguridad

- Los modelos se cargan con `safe_mode=True` y solo si su huella coincide con el manifiesto.
- Rutas por argumento; `np.load(..., allow_pickle=False)`.
