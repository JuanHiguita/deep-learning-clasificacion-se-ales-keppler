# Auditoría del proyecto del curso de Deep Learning

**Fecha:** 20 de noviembre de 2026 · **Alcance:** repositorio `Proyecto_Final` hasta las specs 009–011 · **Tarea del plan:** T7.7

Tiene dos partes: comprobaciones automáticas (`python -I src/fusion/auditoria.py --raiz .`, que se pueden repetir en cualquier momento) y una revisión manual de lo que exige criterio.

## 1. Comprobaciones automáticas: 22 de 22 correctas

| Grupo | Comprobación | Resultado | Detalle |
|---|---|---|---|
| fugas | los escalares de entrada están en la lista permitida del protocolo | ✔ | log10_periodo_d, log10_duracion_h, profundidad_medida_ppm, snr_medido, log10_n_transitos_validos, koi_dikco_msky, koi_dicco_msky |
| fugas | ninguna columna prohibida es un escalar de entrada | ✔ |  |
| fugas | las columnas prohibidas solo se nombran en los archivos autorizados | ✔ | autorizados: comun/lineas_base_tabulares.py; comun/construir_ejemplos.py; comun/descargar_koi.py; comun/verificar_contrato.py; fusion/auditoria.py; informe/secciones.py; cuaderno_demo/flujo.py |
| particiones | cada estrella está en una sola partición | ✔ | 245 estrellas |
| particiones | ningún ejemplo prerregistrado entra a los datasets | ✔ | 0 prerregistrados |
| particiones | todas las predicciones usan la partición de compartido/ejemplos.csv | ✔ | 180 archivos revisados |
| datos | las entradas no contienen NaN ni infinitos | ✔ |  |
| datos | todo escalar que falta vale 0 Y tiene su bandera de faltante | ✔ | 597 valores faltantes marcados |
| datos | compartido/ejemplos.csv coincide con la huella de su manifiesto | ✔ | deb694da52428d07… |
| reproducibilidad | la prueba de extremo a extremo reproduce lo publicado | ✔ | 10 estrellas, tolerancia 0.0001 |
| reproducibilidad | los modelos guardados de cabeza_fusion coinciden con su huella | ✔ |  |
| reproducibilidad | los modelos guardados de m1_astronet coinciden con su huella | ✔ |  |
| reproducibilidad | los modelos guardados de m2_atencion coinciden con su huella | ✔ |  |
| reproducibilidad | los modelos guardados de m2_gru coinciden con su huella | ✔ |  |
| reproducibilidad | los modelos guardados de m2_promedio coinciden con su huella | ✔ |  |
| reproducibilidad | los modelos guardados de m4_escalares coinciden con su huella | ✔ |  |
| cifras | las 25 cifras clave de las tablas aparecen en docs/EXPLICACION_RESULTADOS.md | ✔ |  |
| redacción | no aparece «P\(habitable\)» (la salida se llama P(en zona habitable y rocoso)) | ✔ |  |
| redacción | no aparece «\bno existe (ningún|ninguna|otro|otra) (trabajo|método|estudio|modelo)» (la novedad se redacta como «no encontramos») | ✔ |  |
| redacción | no aparece «\bes el primer (modelo|trabajo|método)» (la novedad se redacta como «no encontramos») | ✔ |  |
| redacción | el documento de resultados declara que la rama de imagen no es un aporte del curso | ✔ |  |
| redacción | el documento de resultados declara que no hay conclusiones sobre zona habitable | ✔ |  |

Además: 111 pruebas rápidas, 2 que entrenan y 1 de extremo a extremo pasan; `verificar_contrato.py` responde "correcto"; `ruff check` sin hallazgos.

La primera pasada de esta auditoría encontró tres fallos, los tres de la propia comprobación y no del proyecto: (1) dos archivos nombran `koi_score` solo como texto de una etiqueta, y se añadieron a la lista de autorizados con su motivo; (2) el AUC de la rama 1 aparece como 0,864 en el documento y como 0,8645 en la tabla, que son el mismo valor con distinto redondeo (media de semillas 0,86437); (3) el plan nombra la expresión vetada para la salida solo para prohibirla.

## 2. Hallazgos de la revisión manual

| # | Hallazgo | Gravedad | Estado |
|---|---|---|---|
| 1 | **La rama 1 del protocolo no era la mejor opción.** Tres variantes (tanh, suavizado de etiquetas, *z-score*) la mejoran de forma distinguible y confirmada con 3 semillas (spec 010). El resultado principal no las usa. | Alta: afecta a qué se reporta como "el modelo" | **Abierto.** Decide Juan si se adopta una variante y se rehace fusión, calibración e incertidumbre |
| 2 | Las variantes de la rama 1 que se confirmaron se eligieron mirando la prueba de la semilla 0. | Media | Declarado. La confirmación sin la semilla 0 también se sostiene, pero usa las mismas estrellas. Si se adopta una, debe elegirse con la validación |
| 3 | MC Dropout no era regenerable (la semilla se fijaba después de cargar el modelo). | Media | **Corregido** en la refactorización; cambió cifras, no el veredicto |
| 4 | El entrenamiento solo es reproducible bit a bit con la misma máquina y configuración de hilos. Un cambio de hilos movió un intervalo que rozaba el 0 (spec 009). | Media | Declarado. Por eso los pesos se guardan y se versionan |
| 5 | La regla del conjunto prerregistrado no se ejercitó: la definición operativa marcó 0 señales. | Media | Declarado en limitaciones. Para la tesis hace falta una definición que marque señales |
| 6 | Circularidad parcial: parte de las etiquetas de falso positivo se asignaron con el centroide, que es una entrada de la rama 4. | Media | Declarado. Se mantiene por decisión de Juan (seguir el plan) |
| 7 | En la clase *sin señal* el centroide siempre falta: la bandera de faltante delata la clase. | Baja para la métrica principal | Declarado. No afecta al AUC planeta vs FP |
| 8 | Las búsquedas y el estudio de diseño usan una sola semilla (salvo la confirmación de la rama 1). | Media | Declarado en cada tabla |
| 9 | `docs/ARQUITECTURA.md` §5 lista comandos y módulos del plan que no existen con ese nombre (`descargar_llc.py`, `exp_sesgos.py`, `linea_base_demo.py`, `fusion.py`). | Baja | **Abierto.** `scripts/reproducir_todo.bat` sí tiene los comandos reales |
| 10 | No se hicieron E1b, E6b, E7b, E10b ni la robustez a desplazamientos de E5. | Baja | Declarado como no realizado |
| 11 | La imagen de píxeles no se vuelve a derivar en la prueba de extremo a extremo. | Baja | Declarado. La rama 3 es código de la demo |
| 12 | Nada se ha ejecutado en Windows (Python 3.13, Keras 3.15): todo corrió en Linux con Python 3.10 y Keras 3.12. | Media | **Abierto.** Falta `pytest -q` y abrir los notebooks en el PC |
| 13 | No hay rúbrica oficial; el título es provisional; el correo de git es `juan@local`. | Media | **Abierto** |
| 14 | Hay archivos de otra sesión de trabajo en el repositorio (`notebooks/00_demo_replicable.ipynb`, `src/cuaderno_demo/`, `scripts/construir_notebook_demo.py`, `specs/012-notebook-demostrativo/`). | — | **Fuera de esta auditoría:** no se revisaron ni se confirmaron en git desde aquí |

## 3. Reglas del proyecto

| Regla | Evidencia |
|---|---|
| Sin fugas (`koi_score`, `koi_fpflag_*`, `koi_pdisposition` nunca son entradas) | Comprobaciones automáticas de fugas; lista blanca de escalares en el protocolo |
| Particiones por estrella; un codificador por partición | Comprobaciones de particiones; el sistema usa el codificador de la partición de cada estrella |
| La prueba no decide nada | Parada temprana, agregador, temperatura y búsqueda de hiperparámetros usan validación interna. **Excepción declarada:** hallazgo 2 |
| Casos al azar con semilla fija; los errores también se reportan | `casos_e11.csv` (semilla 7), pesos de atención y estrellas de la prueba de extremo a extremo |
| Toda cifra se regenera con código | Tablas en `resultados/dl/tablas/`; comprobación automática de cifras del documento de resultados |
| No rellenar un faltante con 0 sin aviso | Comprobación automática; ninguna variante del estudio quita las banderas (hay una prueba) |
| Resultados negativos con el mismo rigor | P1, P3a, P3c, búsqueda de hiperparámetros y E4 se reportan con sus intervalos |
| Sin doble entrega | La rama de imagen es la v0 de la demo y se declara en cada documento; no hay resultados del curso de Visión |
| Nada sobre zona habitable ni atmósferas | Comprobación automática de redacción; límite declarado en cada sección |

## 4. Cómo repetirla

```bash
python -I src/fusion/sistema.py --modelos modelos --dataset data/dataset_temporal.npz --ejemplos compartido/ejemplos.csv --resultados resultados/dl
python scripts/prueba_e2e.py --estrellas-por-particion 2
python -I src/fusion/auditoria.py --raiz .
pytest -q && pytest -q -m lenta && pytest -q -m e2e
```
