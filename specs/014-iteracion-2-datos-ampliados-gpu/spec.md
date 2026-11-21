# Spec 014 · Iteración 2: más datos, GPU, redes más profundas y búsqueda amplia

**Estado:** en ejecución (fase de datos) · **Origen:** decisión del equipo (20-nov-2026).

**Autores del proyecto del curso:** Thomas Monnier Granda, Esteban Felipe Genes Noya, Juan Sebastian Higuita Riascos, Alexandra Arias Mazo y Eliel Segura Perez.

## Por qué

La iteración 1 (245 señales, CPU, protocolo fijo) iguala a un modelo tabular con centroide, pero no lo supera. Tres límites que esta iteración ataca: pocos datos para una red profunda, modelos pequeños y una búsqueda de hiperparámetros corta.

## Qué cambia y qué no

| | Iteración 1 | Iteración 2 |
|---|---|---|
| Señales | 245 (selección de la demo) | **4.547**: todas las de DR25 con etiqueta definitiva y periodo de 1 a 30 días (4.007 estrellas) |
| Ejemplos | 490 | 9.094 (señal real + misma estrella sin señal) |
| Prerregistrados | 0 | 76 señales apartadas antes de entrenar |
| Motor | Keras sobre JAX, CPU | **Keras sobre PyTorch, GPU** (RTX 3060 Ti). El código de los modelos es el mismo |
| Ramas | 1, 2, 4 + imagen v0 | **1, 2 y 4** (todas de este curso). Sin imagen |
| Redes | pequeñas, protocolo fijo | más anchas y profundas; búsqueda amplia de hiperparámetros |

**No se borra nada de la iteración 1.** La iteración 2 vive en `iteracion_2/` y solo se presenta como resultado si mejora de forma distinguible.

## Decisiones (20-nov-2026)

- **D-1 Motor:** Keras sobre PyTorch en Windows nativo con CUDA. Rompe la regla "sin PyTorch" solo en esta iteración, porque JAX no usa GPU en Windows nativo. Decidido por Juan.
- **D-2 Datos:** periodo 1–30 d, el mismo régimen que la iteración 1, para que la comparación sea limpia.
- **D-3 Imagen:** no se entrena en esta iteración (es del curso de Visión). **No se obstaculiza la unión futura:** se conservan el contrato (`id_ejemplo`, vectores de 32/32/16, un codificador por partición) y la cabeza con banderas de disponibilidad, de modo que añadir `modelos/imagen/<versión>` después solo exige reentrenar la cabeza.
- **D-4 Lanzamiento:** los procesos largos corren en Windows nativo (`iteracion_2\*.bat`) y dejan registros en `iteracion_2/evidencias/registros/`.

## Reglas que se mantienen

Sin columnas prohibidas como entrada; particiones por estrella; validación interna para toda decisión; prerregistrados apartados y evaluados una sola vez al final; toda cifra sale de un script; resultados negativos reportados igual.

## Fases y notebooks (`iteracion_2/notebooks/`)

| Fase | Notebook | Contenido |
|---|---|---|
| 1 | `01_datos_y_preprocesamiento` | Selección, descarga, extractor, control de calidad |
| 2 | `02_arquitecturas` | Las tres ramas y la fusión, con sus variantes más profundas |
| 3 | `03_busqueda_de_hiperparametros` | Espacio declarado, búsqueda en GPU, selección por validación |
| 4 | `04_entrenamiento_final_y_fusion` | Mejores modelos, fusión, calibración; modelos en `iteracion_2/modelos_finales/` |
| 5 | `05_analisis_de_resultados` | Comparación con el tabular (diferencias pareadas), con la iteración 1 y prerregistrados |

Cada notebook prepara su entorno en la primera celda y usa solo rutas dentro del repositorio.

## Criterio de éxito (fijado antes de entrenar)

La iteración 2 "mejora" si la fusión de las ramas 1, 2 y 4 supera de forma distinguible (intervalo del 95 % de la diferencia pareada por encima de 0) al modelo tabular **sin centroide** entrenado con las mismas señales. Se reporta además frente al tabular con centroide, sea cual sea el resultado.

## Riesgos

- Reproducibilidad en GPU: PyTorch con CUDA no es determinista bit a bit; se guardan los pesos.
- 26 GB de descarga: reanudable por estrella.
- Con miles de señales el tabular también mejora; puede que la conclusión no cambie.
