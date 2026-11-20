# CLAUDE.md

Guía para agentes de código (Opus, Sonnet u otro) que trabajen en este repositorio.

## Qué es

Proyecto del curso de Deep Learning de Juan Sebastian Higuita Riascos (Especialización en IA, Universidad Nacional de Colombia). Entrega: **6 de diciembre de 2026**. Clasifica señales de tránsito de Kepler (planeta, falso positivo, sin señal) con una red de cuatro ramas, y estudia tres preguntas: qué arquitectura lee mejor la señal, cuánto aporta la fusión y si las probabilidades salen calibradas.

Es una parte de un proyecto de tesis más grande. La entrega de este curso es **independiente** de la del curso de Visión por Computador: la rama de imagen que se usa aquí es el codificador v0 de la demo, congelado. Los dos proyectos se unen después de las entregas.

## Qué leer antes de tocar nada

1. `.specify/memory/constitution.md`: reglas que no se negocian.
2. `docs/planeacion/BITACORA_DL.md`: qué está hecho y qué sigue.
3. `docs/planeacion/04_plan_ejecucion_deep_learning.md`: plan, decisiones, protocolo, calendario.
4. `docs/ARQUITECTURA.md`: dónde va cada cosa.
5. `specs/<NNN>/spec.md`, `plan.md`, `tasks.md` del paquete en curso.

## Cómo trabajar

- Un paquete de trabajo a la vez, en el orden del plan §7. Dentro de él, las tareas de `tasks.md` en orden; marca `[x]` al terminar cada una.
- Flujo Spec Kit: la spec y el plan ya existen. Si necesitas cambiar una spec, edítala **antes** de programar y anota el cambio en la bitácora.
- Escribe la prueba junto con el código. Una tarea no está terminada sin `pytest -q` en verde.
- Antes de cerrar la sesión: `pytest -q`, `python -I src/comun/verificar_contrato.py --raiz .` (cuando exista), commit `PT<n>: <qué>` y entrada en la bitácora.
- Reporta las cifras tal como salieron del script. No redondees hacia el lado favorable ni completes una tabla con valores esperados.

## Comandos

```bash
pip install -r requirements.txt && pip install -e .
pytest -q                      # pruebas rápidas
pytest -q -m lenta             # pruebas que entrenan
ruff check src tests
```

En Windows (PowerShell): `$env:KERAS_BACKEND="jax"`. La secuencia completa de scripts está en `docs/ARQUITECTURA.md` §5.

## Cuándo detenerte y preguntar a Juan

- Hace falta cambiar el contrato, las particiones o la base después de la etiqueta `base-v1`.
- Una prueba de cordura falla, o aparece un AUC > 0,99.
- Faltan datos, una columna no existe o un *hash* no coincide.
- Para avanzar habría que mirar la prueba para elegir algo.
- Necesitarías cualquier cosa del curso de Visión (las entregas son independientes).
- El prerregistro marca más del 10 % de la selección.
- La inspección de K03615.01 no muestra alternancia par/impar.

## Lo que no se hace

- Usar `koi_score`, `koi_fpflag_*` o `koi_pdisposition` como entrada.
- Rellenar NaN con 0 sin su columna de aviso.
- Cambiar hiperparámetros de `config/protocolo.json` para mejorar un resultado.
- Borrar ejemplos "raros" sin una regla escrita antes en la spec.
- Reportar la CNN de imagen como resultado propio, o comparar contra resultados de Visión.
- Escribir conclusiones sobre zona habitable. Decir "P(habitable)".
- Usar PyTorch. Subir datos grandes a git. Poner rutas absolutas.
- Usar curvas sintéticas fuera de `tests/`.

## Errores típicos de este dominio

- En los TPF, filtrar cadencias con `np.isfinite(flux).any(axis=(1, 2))`, no `.all()`.
- La posición del objetivo sale de `tpf.wcs.world_to_pixel_values(tpf.ra, tpf.dec)`.
- `koi_score` es NaN fuera de DR25.
- Recortar el SNR a ±30 satura las señales profundas (caso K00022.01): por eso este proyecto normaliza por forma (plan, DD-04).
- Un desplazamiento de centroide igual a 0 "parece planeta": nunca imputar 0 sin aviso.
- Con `StratifiedGroupKFold`, los dos ejemplos de un KOI deben compartir partición.
- Keras con JAX: fijar `KERAS_BACKEND` antes de `import keras`. Capas propias con `keras.ops` y registradas para serialización.

## Estado

Ver la tabla al inicio de `docs/planeacion/BITACORA_DL.md`.
