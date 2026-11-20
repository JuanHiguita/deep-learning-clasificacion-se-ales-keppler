# 003 · Motor de experimentos — tareas

Modelo sugerido: Opus. Duración: 1 día.

- [x] T3.1 `comun/protocolo.py` con pruebas.
- [x] T3.2 `comun/metricas.py` con pruebas (contra `sklearn` y casos a mano).
- [x] T3.3 `comun/estadistica.py` con pruebas.
- [x] T3.4 `estadisticas_estandarizacion` y `aplicar_estandarizacion` con pruebas (faltantes fuera de las estadísticas; prueba de no fuga).
- [x] T3.5 `entrenar_una` (bucle de épocas, parada temprana, restauración de pesos). Prueba con un modelo denso sobre datos separables.
- [x] T3.6 `entrenar_cv` y escritura de `pred_semilla<s>.npz`, historial y manifiesto. Prueba del marcador (ningún ejemplo de prueba en el entrenamiento).
- [x] T3.7 `evaluar_predicciones` con estratos. Prueba: estrato pequeño devuelve `null`.
- [x] T3.8 `exportar_codificador`. Prueba: el vector exportado coincide con la activación de la capa `embedding`.
- [x] T3.9 `prueba_etiquetas_barajadas` y `prueba_sobreajuste_lote`.
- [x] T3.10 `tests/test_integracion.py` (marcada `lenta`).
- [x] T3.11 Corrida real de M4 (escalares) sobre `dataset_temporal.npz` como ensayo general. Anotar tiempos en la bitácora y actualizar el presupuesto del plan §8.
- [ ] T3.12 Commit `PT3: motor de experimentos`.
