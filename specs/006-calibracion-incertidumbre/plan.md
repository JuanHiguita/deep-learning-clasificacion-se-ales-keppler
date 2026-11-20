# 006 · Calibración e incertidumbre — plan técnico

## Archivos

`src/fusion/calibracion.py`, `src/fusion/incertidumbre.py`, `src/fusion/errores.py`, `tests/test_calibracion.py`.

## Firmas

```python
# calibracion.py
def aplicar_temperatura(p: np.ndarray, T: float) -> np.ndarray
def ajustar_temperatura(p_validacion: np.ndarray, y_validacion: np.ndarray) -> float
def calibrar_por_particion(pred: dict, ejemplos: pd.DataFrame) -> tuple[np.ndarray, list[float]]
def tasa_base_dr25(koi: pd.DataFrame, periodo_min: float = 1.0, periodo_max: float = 30.0) -> tuple[float, int, int]
def evaluar_prior(q, es_planeta, kepid, pi_ent_por_ejemplo, tasa_objetivo, protocolo) -> dict
def figura_confiabilidad(p_antes, p_despues, y, ruta: Path) -> None

# incertidumbre.py
def predecir_mc_dropout(cabeza: keras.Model, x: np.ndarray, pasadas: int, semilla: int) -> np.ndarray   # (pasadas, n, 3)
def entropia(p: np.ndarray) -> np.ndarray
def informacion_mutua(p_miembros: np.ndarray) -> np.ndarray          # (m, n, 3) → (n,)
def riesgo_cobertura(confianza: np.ndarray, acierto: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]   # cobertura, riesgo, AURC
def auroc_deteccion_errores(incertidumbre: np.ndarray, acierto: np.ndarray) -> float

# errores.py
def elegir_casos(pred, ejemplos, semilla: int = 7, n: int = 10) -> pd.DataFrame
def ficha_de_caso(id_ejemplo, ...) -> Figure
```

## Decisiones técnicas

- `aplicar_temperatura` trabaja con `log(clip(p))`, resta el máximo y normaliza (estable).
- La temperatura de la partición *k* se ajusta con las filas donde `particion_modelo_validacion == k` (spec 003).
- MC Dropout: `cabeza(x, training=True)` en un bucle con `keras.utils.set_random_seed(semilla + i)`. Requiere guardar las cabezas `.keras` y los vectores de entrada por partición (lo hace la spec 005).
- `riesgo_cobertura`: ordenar por confianza descendente; riesgo acumulado = errores acumulados / retenidos; AURC = media del riesgo sobre todas las coberturas.
- `scipy` ya viene con `scikit-learn`; no es una dependencia nueva.

## Pruebas

Las de los criterios de aceptación, más: `entropia` de una distribución uniforme de 3 clases = ln 3; `informacion_mutua` = 0 con miembros idénticos; AURC de un clasificador perfecto = 0.
