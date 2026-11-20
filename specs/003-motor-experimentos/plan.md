# 003 · Motor de experimentos — plan técnico

## Archivos

`src/comun/entrenamiento.py`, `tests/test_entrenamiento.py`, `tests/test_integracion.py` (marcada `lenta`).

## Firmas

```python
Entradas = dict[str, np.ndarray]     # nombre de entrada Keras → arreglo (n, ...)

@dataclass(frozen=True)
class Experimento:
    nombre: str                                   # carpeta bajo resultados/dl/
    constructor: Callable[[int], keras.Model]     # semilla → modelo sin compilar, con capa "embedding"
    claves_entrada: tuple[str, ...]
    clave_disponible: str | None = None           # p. ej. "disponible_transitos"
    aumentar: Callable[[Entradas, np.random.Generator], Entradas] | None = None
    estandarizar: tuple[str, ...] = ()            # claves que se estandarizan por partición
    clave_faltantes: str | None = None            # máscara que acompaña a la clave estandarizada

def entrenar_cv(exp: Experimento, entradas: Entradas, ejemplos: pd.DataFrame, protocolo: dict,
                semillas: Sequence[int], salida: Path) -> dict
def entrenar_una(modelo, x_ent, y_ent, x_val, y_val, cfg, aumentar, rng) -> tuple[keras.Model, dict]
def estadisticas_estandarizacion(x, faltantes=None) -> tuple[np.ndarray, np.ndarray]
def aplicar_estandarizacion(x, media, desviacion, faltantes=None) -> np.ndarray     # faltantes → 0
def exportar_codificador(exp, entradas, ejemplos, protocolo, semilla, destino: Path, version: str) -> None
def evaluar_predicciones(p: np.ndarray, ejemplos: pd.DataFrame, protocolo: dict) -> dict
def prueba_etiquetas_barajadas(exp, entradas, ejemplos, protocolo) -> float
def prueba_sobreajuste_lote(exp, entradas, ejemplos) -> float
```

## Decisiones técnicas

- **Bucle propio de épocas** (como la demo): `modelo.fit(..., epochs=1)` dentro de un `for`, para aplicar la aumentación en cada época y controlar la parada temprana. La validación se evalúa con `modelo.evaluate` sin pesos por clase.
- **`p_validacion`:** con la convención *(k+1) mod 5*, cada ejemplo es validación interna de exactamente una partición (la *k* = su partición − 1). Se guarda la predicción del modelo de esa partición. Así `p_validacion` está completa y sirve para ajustar la temperatura sin tocar la prueba. Se guarda además `particion_modelo_validacion` para dejarlo explícito.
- **Pesos por clase:** `n / (3 · n_clase)` sobre el entrenamiento de la partición.
- **Submodelo de exportación:** `keras.Model(modelo.inputs, modelo.get_layer("embedding").output)`.
- **Tiempo por época:** `time.perf_counter` alrededor de `fit`.
- **`evaluar_predicciones`** llama a `comun.metricas` y `comun.estadistica`; estratos: global, por bandera de FP, por tercil de MES (terciles guardados en el manifiesto de la base) y por tercil de `n_transitos_validos`. Estratos con menos de 20 ejemplos: se reporta `n` y `null` en el intervalo.

## Riesgos

- Fuga por estandarización: cubierta por una prueba.
- JAX recompila en cada modelo nuevo: aceptable con modelos pequeños; si pesa, reutilizar el modelo reinicializando pesos.
