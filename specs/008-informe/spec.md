# 008 · Informe, presentación y repositorio — especificación y tareas

**PT8 del plan · Capa 5 · Modelo sugerido: Opus redacta, Sonnet da formato · jue 3 a sáb 5-dic**

> Pendiente: contrastar esta estructura con el enunciado y la rúbrica oficiales cuando Juan los suba (DA-5).

## Entregables

| Entregable | Archivo | Notas |
|---|---|---|
| Informe | `docs/informe/informe_dl.md` → PDF | 12–15 páginas |
| Notebook | `notebooks/informe_dl.ipynb` | Reproduce tablas y figuras (spec 007) |
| Presentación | 10–12 diapositivas | Mismo diagrama del sistema que Visión, resaltando lo de este curso |
| Repositorio | etiqueta `curso-dl-v1` | `README.md` con los comandos de `docs/ARQUITECTURA.md` §5 |

## Estructura del informe

| # | Sección | Contenido | Fuente de las cifras |
|---|---|---|---|
| 1 | Resumen | Las tres respuestas, con intervalo | `resumen_hipotesis.csv` |
| 2 | Problema y trabajo previo | Tránsitos, impostores, Robovetter, AstroNet, ExoMiner. Novedad como "no encontramos" | Literatura (anexo C del plan) |
| 3 | Datos y base común | Selección, DR25, tres clases, particiones, prerregistro. **La base se comparte con el curso de Visión** | `manifiesto_base.json`, EDA |
| 4 | Arquitectura del sistema | Diagrama de 4 ramas. Qué es de este curso y qué viene de Visión o de la demo | `docs/ARQUITECTURA.md` §6 |
| 5 | Protocolo | Validación, semillas, métricas, *bootstrap*, hiperparámetros fijos | `config/protocolo.json` |
| 6 | P1 · ¿Qué arquitectura lee mejor la señal? | Tabla comparativa; sesgos inductivos con E3, E4, E5 | `p1_*.csv` |
| 7 | P2 · ¿Cuánto mejora la fusión? | Fusión, fusión tardía, ablación por rama y por estrato; diagnóstico del apilado | `p2_*.csv` |
| 8 | P3 · ¿Probabilidades confiables? | Temperatura, prior explícito, MC Dropout frente a ensamble | `p3_*.csv` |
| 9 | Análisis de errores | Casos al azar; K00022.01 | E11 |
| 10 | Resultados negativos y limitaciones | Ver lista abajo | — |
| 11 | Conexión con la tesis y conclusiones | Fases 3, 4b, 5 y 6 | `00_ESTADO` |
| 12 | Declaraciones | Uso de IA; trabajo compartido entre cursos; versión de la imagen | — |
| A | Anexo de desviaciones | | `desviaciones.md` |

**Limitaciones obligatorias (sección 10):** periodo corto (1–30 d) con decenas de tránsitos; solo KOIs reales, sin inyecciones ni falsas alarmas reconstruidas; clase `sin_senal` construida y el atajo del centroide faltante; unos 520 ejemplos e intervalos anchos; apilado; incertidumbre solo de la cabeza; la corrección de prior supone solo cambio de tasa base; prerregistro aproximado; versión de la imagen usada.

## Reglas de redacción

- Cada afirmación comparativa lleva su intervalo. "Mejor" solo si el intervalo de la diferencia excluye el 0; si no, "no se distingue".
- Un resultado negativo ocupa el mismo espacio que tendría el positivo.
- Vocabulario de la constitución (VIII).
- Ninguna cifra de la demo antigua (0,83; 0,95; 0,022) como resultado: solo como motivación histórica, citando `00_ESTADO` §5, y junto a la cifra vuelta a medir.
- Referencias verificadas con la habilidad `citation-check`.

## Tareas

- [ ] T8.1 Esqueleto del informe con las tablas y figuras insertadas (sin texto de interpretación).
- [ ] T8.2 Secciones 3, 4 y 5 (no dependen de los resultados; se pueden adelantar al bloque C).
- [ ] T8.3 Secciones 6, 7 y 8 a partir de `resumen_hipotesis.csv`.
- [ ] T8.4 Secciones 9 y 10.
- [ ] T8.5 Secciones 1, 2, 11 y 12.
- [ ] T8.6 `auditar_informe.py` sobre el texto; corregir toda cifra huérfana.
- [ ] T8.7 `citation-check` sobre las referencias.
- [ ] T8.8 Presentación.
- [ ] T8.9 `README.md` final; clonar en una carpeta limpia y seguir sus pasos hasta `pytest` y el notebook.
- [ ] T8.10 Etiqueta `curso-dl-v1`. Texto actualizado para `00_ESTADO_Y_DECISIONES.md` (resultados y bitácora) para que Juan lo reemplace en el proyecto.
