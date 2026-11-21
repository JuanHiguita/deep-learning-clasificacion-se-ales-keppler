| Grupo | Comprobación | Resultado | Detalle |
|---|---|---|---|
| fugas | los escalares de entrada están en la lista permitida del protocolo | ✔ | log10_periodo_d, log10_duracion_h, profundidad_medida_ppm, snr_medido, log10_n_transitos_validos, koi_dikco_msky, koi_dicco_msky |
| fugas | ninguna columna prohibida es un escalar de entrada | ✔ |  |
| fugas | las columnas prohibidas solo se nombran en los archivos autorizados | ✔ | autorizados: comun/lineas_base_tabulares.py; comun/construir_ejemplos.py; comun/descargar_koi.py; comun/verificar_contrato.py; fusion/auditoria.py; informe/secciones.py; cuaderno_demo/flujo.py |
| particiones | cada estrella está en una sola partición | ✔ | 245 estrellas |
| particiones | ningún ejemplo prerregistrado entra a los datasets | ✔ | 0 prerregistrados |
| particiones | todas las predicciones usan la partición de compartido/ejemplos.csv | ✔ | 191 archivos revisados |
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
