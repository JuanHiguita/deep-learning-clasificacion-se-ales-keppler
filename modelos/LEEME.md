# Modelos guardados

Todos los pesos de esta carpeta se versionan en git (son pequeños) para poder volver a usarlos.

| Carpeta | Qué contiene | Para qué |
|---|---|---|
| `vistas/v1`, `transitos/v1`, `escalares/v1` | Codificador por partición (hasta la capa `embedding`), sus vectores y su estandarización | Ramas del **resultado principal** (protocolo fijo) |
| `imagen/v0` | Codificador de imagen de la demo, por partición | Rama 3 (código anterior a los cursos; no es un aporte de este curso) |
| `../resultados/dl/fusion/e6a_cuatro_ramas/cabezas/` | Cabeza de fusión por partición (semilla 0) | Decisión final del resultado principal |
| `mejores/<arquitectura>/` | Modelo **completo** que eligió cada partición en la búsqueda de hiperparámetros, con estandarización y manifiesto | Análisis de sensibilidad (spec 009) |

## Volver a usar un modelo de `mejores/`

```python
from temporal import despliegue

guardado = despliegue.cargar("modelos/mejores", "m2_atencion", particion=0)   # verifica la huella SHA-256
probabilidades = guardado.predecir(entradas)   # entradas con el formato de data/dataset_temporal.npz
```

`probabilidades` tiene una fila por ejemplo y tres columnas: planeta, falso positivo y sin señal.

## Qué hay que saber antes de usarlos

- **Hay un modelo por partición.** El de la partición *k* se entrenó sin las estrellas de esa partición: sobre ellas su predicción es honesta. Sobre estrellas de otras particiones, las vio al entrenar.
- **Para una estrella nueva** se promedian los cinco modelos. Eso todavía no está automatizado de punta a punta (falta el paso que va de la curva de luz descargada a las entradas y encadena ramas, fusión y temperatura).
- **Rango de validez:** señales de Kepler de periodo entre 1 y 30 días. Fuera de ese rango no hay evidencia de que funcionen.
- Las capas propias (`temporal.modelos`) deben estar importadas para cargar los `.keras`; `despliegue.cargar` ya lo hace y usa `safe_mode=True`.
- Los pesos son reproducibles en la misma máquina y configuración de hilos. En otra máquina, reentrenar puede dar pesos ligeramente distintos; por eso se guardan.
