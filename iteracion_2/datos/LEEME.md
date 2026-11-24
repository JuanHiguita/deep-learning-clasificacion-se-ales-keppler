# Datos de la iteración 2

| Archivo | Dónde está |
|---|---|
| `seleccion.csv`, `ejemplos.csv`, `manifiesto_base.json` | En el repositorio |
| `dataset_temporal_registro.csv`, `dataset_temporal.manifiesto.json` | En el repositorio |
| `muestra_dataset.npz` (60 ejemplos al azar, semilla fija) | En el repositorio; lo usa el cuaderno 01 si falta el dataset completo |
| `dataset_temporal.npz` (148 MB, 8.942 ejemplos) | Adjunto de la release [`iteracion-2-datos`](https://github.com/JuanHiguita/deep-learning-clasificacion-se-ales-keppler/releases/tag/iteracion-2-datos). Descárgalo y ponlo en esta carpeta. Su SHA-256 está en `dataset_temporal.manifiesto.json` |
| `llc/` (61.714 curvas de luz de Kepler) | No se versiona: se descarga de MAST con `iteracion_2/1_preparar_gpu_y_descargar.bat` |

Los modelos de cada configuración de la búsqueda de hiperparámetros (1,1 GB) tampoco se versionan; los
que quedaron elegidos están en `iteracion_2/modelos_finales/`, y las predicciones y métricas de todas las
configuraciones sí están en `iteracion_2/evidencias/resultados/busqueda/`.
