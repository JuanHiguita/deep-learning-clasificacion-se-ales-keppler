# 002 · Dataset temporal — tareas

Modelo sugerido: Opus escribe `curvas.py` y revisa; Sonnet hace descarga, orquestación, pruebas y notebook.

- [x] T2.1 `descargar_mast.py`. Ejecutado el 20-nov: 3.792 curvas (1,6 GB) y 224 TPF de Q5 (525 MB); 7 estrellas con menos de 4 trimestres. Revisar el avance cada pocas horas. Corte: domingo 22-nov; si no llega al 80 %, activar el respaldo de la spec.
- [x] T2.2 [P] `tests/sinteticos.py`: generador de curva con tránsitos de caja, ruido gaussiano, tendencia lenta, huecos entre trimestres, opción de alternancia par/impar, eclipse secundario y segundo planeta.
- [x] T2.3 `curvas.py`: `mascara_transitos`, `aplanar`, `ruido_robusto`, `fase`, `medir_profundidad`, `escala`. Pruebas de profundidad y escala.
- [x] T2.4 `vista_plegada` y las 4 vistas. Pruebas de formas y de eclipse secundario.
- [x] T2.5 `ventanas_por_transito`, `seleccionar_transitos`, `estadisticos_consistencia`. Pruebas de par/impar, huecos, otro KOI y selección.
- [x] T2.6 `construir_ejemplo` para `transito` y `sin_senal`. Prueba de sin señal.
- [x] T2.7 `lectura.py` y `construir_dataset_temporal.py` **(escritos; probados de punta a punta con FITS sintéticos)**. Falta probar con 5 estrellas reales antes de correr todo.
- [x] T2.8 Corrida completa. Guardar `.npz`, registro y manifiesto. `verificar_contrato.py`.
- [x] T2.9 (numérica; falta la figura) Inspección de K00022.01 y K03615.01 (figura en `resultados/dl/eda/`). **Si la alternancia de K03615.01 no se ve, detenerse:** el extractor está mal.
- [ ] T2.10 `notebooks/eda_temporal.ipynb`.
- [ ] T2.11 Revisión (Opus): leer el registro de descartes; confirmar ≥ 95 % de disponibilidad; confirmar que ningún escalar viene de una columna prohibida.
- [ ] T2.12 Commit `PT2: dataset temporal`, bitácora (con el conteo de `aviso_sin_senal`).

**Si hay retraso:** reducir `vista_global` a 401 puntos (anotar desviación) y dejar el notebook de EDA para el bloque E.
