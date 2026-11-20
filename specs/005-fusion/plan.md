# 005 · Fusión y ablación — plan técnico

## Archivos

`src/fusion/codificador_imagen_v0.py`, `src/fusion/vectores.py`, `src/fusion/fusion.py`, `tests/test_fusion.py`.

## Firmas

```python
# vectores.py
RAMAS = ("vistas", "transitos", "imagen", "escalares")
def cargar_vectores(modelos: Path, versiones: dict[str, str], k: int, ids: np.ndarray, dataset_temporal: Path | None
                    ) -> tuple[dict[str, np.ndarray], np.ndarray]
    # → vectores por rama alineados con ids (ceros donde falta) y disponible (n, 4)
def armar_entrada(vectores: dict, disponible: np.ndarray, ramas: tuple[str, ...]) -> np.ndarray

# fusion.py
def cabeza(dim_entrada: int, semilla: int, dropout: float = 0.3) -> keras.Model
def entrenar_fusion(ramas: tuple[str, ...], ..., semillas, salida: Path) -> dict       # una configuración de ramas
def fusion_tardia(carpetas_ramas: dict[str, Path]) -> np.ndarray
def diagnostico_apilado(historiales: list[dict]) -> dict
def ajuste_cruzado(...) -> dict                                                         # solo si el diagnóstico lo pide
```

## Decisiones técnicas

- La cabeza no es un `Experimento` normal porque sus entradas cambian con la partición (los vectores de la partición *k* vienen del codificador *k*). `entrenar_fusion` repite el bucle de particiones del motor llamando a `entrenar_una` y a `evaluar_predicciones`; no duplica la lógica de separación (`protocolo.separar`).
- Sin aumentación en la cabeza, salvo E7b.
- Una configuración de ramas = una carpeta: `resultados/dl/fusion/e6a/`, `e7_sin_<rama>/`, `e7_solo_<rama>/`.
- Para "rama sola" en la tabla se usa la cabeza reentrenada sobre el vector congelado (misma receta para las 4), no la cabeza con que se entrenó la rama.

## Pruebas

- `cargar_vectores` une por identificador aunque los archivos vengan en otro orden.
- Un ejemplo sin vector de imagen queda con bandera 0 y ceros.
- `armar_entrada` con 3 ramas tiene la dimensión esperada.
- La fusión tardía de 4 predicciones idénticas devuelve esa predicción.

## Riesgo principal

Sobreajuste del apilado (RF-8). Segundo riesgo: que Visión use otra validación interna; entonces los vectores de imagen de la validación son "vistos". No afecta a la prueba; se declara.
