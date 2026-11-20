# Constitución del proyecto

Principios que ninguna especificación, plan o tarea puede violar. Si una tarea choca con uno de ellos, la tarea está mal.

## I. Sin fugas

`koi_score`, `koi_fpflag_*` y `koi_pdisposition` nunca son entradas de un modelo. Solo pueden aparecer como comparador externo (`koi_score`) o como estratos de reporte (banderas).

## II. Particiones por estrella

Toda separación entre entrenamiento, validación interna y prueba se hace por `kepid`. Los codificadores congelados son uno por partición. Las particiones de `compartido/ejemplos.csv` no se regeneran.

## III. La prueba no decide nada

Ninguna elección (arquitectura, hiperparámetro, temperatura, umbral, variante que pasa a la fusión) se toma mirando datos de prueba. Se decide con la validación interna: la partición *(k+1) mod 5*.

## IV. El conjunto prerregistrado no se toca

No entra a entrenamiento, validación, prueba ni análisis exploratorio de resultados.

## V. Toda cifra se regenera

Cada número de un informe sale de `resultados/dl/tablas/` y se produce con un comando documentado. Lo que viene de la literatura lleva su cita. No se inventan cifras, datos ni referencias; lo que no se puede verificar se marca como pendiente.

## VI. Honestidad del reporte

Casos mostrados elegidos al azar con semilla fija; los errores también se muestran. Toda comparación lleva intervalo de confianza por *bootstrap* por estrella. Los resultados negativos se escriben con el mismo detalle que los positivos. La novedad se redacta como "no encontramos".

## VII. Sin doble entrega

La rama de imagen se usa congelada y se acredita al curso de Visión (o a la demo, si es v0). Este repositorio no reporta resultados de Visión como propios ni comparte experimentos con ese curso.

## VIII. Vocabulario

"P(en zona habitable y rocoso)", nunca "P(habitable)". Nada sobre atmósferas. "Clasificación" o "revisión" de señales, no "detección de exoplanetas". Ninguna conclusión sobre el régimen de zona habitable a partir de datos de periodo corto.

## IX. Datos faltantes

No se rellena NaN con 0 sin una columna que avise que el dato falta.

## X. Ingeniería

Spec Kit para cada funcionalidad (spec → plan → tareas → implementación). Keras 3 con JAX en CPU; sin PyTorch. Scripts con `argparse`, rutas por argumento, ninguna ruta absoluta. Revisión de seguridad según `docs/ARQUITECTURA.md` §9. Pruebas antes de dar una tarea por terminada.

## XI. Desviaciones

Cualquier cambio respecto al plan de ejecución se anota en `docs/planeacion/BITACORA_DL.md` con fecha, motivo y efecto, y aparece en el anexo de desviaciones del informe.
