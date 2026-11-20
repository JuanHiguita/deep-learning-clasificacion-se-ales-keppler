# 007 · Consolidación de resultados — especificación, plan y tareas

**PT7 del plan · Capa 4–5 · Modelo sugerido: Sonnet; auditoría final: Opus · 1 día (mié 2-dic)**

## Para qué

Que cada cifra del informe tenga un único origen y se pueda regenerar con un comando.

## Requisitos

- **RF-1** `src/fusion/consolidar_resultados.py --resultados resultados/dl --salida resultados/dl/tablas` lee los `metricas.json` de todos los experimentos y escribe:
  - `p1_arquitecturas.csv`, `p1_sesgos.csv` (E3, E4, E5);
  - `p2_fusion.csv`, `p2_ablacion_estratos.csv`;
  - `p3_calibracion.csv`, `p3_incertidumbre.csv`;
  - `lineas_base.csv`;
  - `resumen_hipotesis.csv`: una fila por hipótesis de la sección 3 del plan, con la diferencia, su intervalo y el veredicto mecánico (`se sostiene` si el intervalo excluye el 0 en el sentido previsto; `no se sostiene` en otro caso).
- **RF-2** Formato largo: `experimento, modelo, estrato, metrica, estimacion, ic_inf, ic_sup, n, semillas`.
- **RF-3** Figuras finales en `resultados/dl/figuras/`, generadas solo desde las tablas y las predicciones: comparación de arquitecturas con intervalos; E3 y E4; ablación; confiabilidad; riesgo-cobertura; efecto del prior; fichas de casos.
- **RF-4** `notebooks/informe_dl.ipynb`: carga las tablas y figuras, las muestra con el texto mínimo de cada sección del informe. No importa Keras. Corre de principio a fin en menos de 5 minutos.
- **RF-5** `resultados/dl/desviaciones.md`: lista de desviaciones del plan, extraída de la bitácora.
- **RF-6** `src/fusion/auditar_informe.py --informe <ruta.md> --tablas resultados/dl/tablas`: extrae los números del texto del informe y avisa de los que no aparecen (con el redondeo usado) en ninguna tabla.

## Tareas

- [ ] T7.1 `consolidar_resultados.py` y tablas.
- [ ] T7.2 `resumen_hipotesis.csv` (veredicto mecánico; el texto lo redacta una persona o un modelo después, sin cambiar el veredicto).
- [ ] T7.3 Figuras finales.
- [ ] T7.4 `informe_dl.ipynb`; ejecutarlo completo desde cero.
- [ ] T7.5 `desviaciones.md`.
- [ ] T7.6 `auditar_informe.py`.
- [ ] T7.7 **Auditoría final (Opus):** recorrer la lista de la sección 12 del plan, punto por punto, con evidencia (comando y salida) de cada uno. Guardarla en `resultados/dl/auditoria_final.md`.
- [ ] T7.8 Commit `PT7: consolidación`.

## Aceptación

- [ ] Borrar `resultados/dl/tablas/` y regenerarlo da archivos idénticos.
- [ ] La auditoría final no tiene puntos abiertos, o los que tiene están declarados en el informe.
