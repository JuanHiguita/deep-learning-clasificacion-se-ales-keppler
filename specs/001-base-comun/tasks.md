# 001 · Base común — tareas

Modelo sugerido: Sonnet implementa; Opus revisa T1.9. `[P]` = se puede hacer en paralelo con la anterior.

- [x] T1.0 ¿Existe ya una base construida por Visión? **No (confirmado el 19-nov): se construye aquí.**
- [x] T1.1 `src/comun/utilidades.py` con pruebas (hash, `validar_kepid` con valores inválidos, `ruta_dentro_de` con `..`).
- [ ] T1.2 `descargar_koi.py` **(escrito y probado sin red)**. Falta descargar DR25 y `cumulative` a `data/` (lo hace `scripts\descargar_recursos.bat`).
- [x] T1.3 Comprobar que todas las columnas de RF-1 existen. **Comprobado el 19-nov con una consulta TAP de 2 filas a `q1_q17_dr25_koi`: las 19 columnas existen.** El script lo vuelve a comprobar al descargar.
- [x] T1.4 `construir_ejemplos.py`: `tipo_fp`, `marcar_prerregistrados`, `asignar_particiones`, `construir`. Pruebas con una tabla de 20 KOIs en memoria.
- [ ] T1.5 Ejecutar y escribir `compartido/ejemplos.csv` + `compartido/manifiesto_base.json`. Anotar en la bitácora: filas, clases, cambios de etiqueta frente a la demo, número de prerregistrados.
- [ ] T1.6 Ejecutar dos veces y comparar SHA-256 (determinismo).
- [x] T1.7 `verificar_contrato.py` (comprobaciones 1, 2, 3 y 7 ahora; 4–6 se activan solas cuando existan los archivos). Pruebas: cada comprobación con un caso que debe fallar.
- [ ] T1.8 [P] `lineas_base_tabulares.py` **(escrito y probado con tablas sintéticas)**. Falta ejecutarlo con la tabla real y guardar `resultados/dl/lineas_base_tabulares/metricas.json`.
- [ ] T1.9 Revisión (Opus): leer `ejemplos.csv` y el manifiesto; confirmar que ninguna columna prohibida entra a S3-B/S3-C; confirmar el balance por partición.
- [x] T1.10 Commit `PT1: base común`, etiqueta `base-v1`, bitácora. Escribir en la bitácora el texto de aviso para el chat de Visión.

**Si hay retraso:** recortar la modalidad (ii) de T1.8.
