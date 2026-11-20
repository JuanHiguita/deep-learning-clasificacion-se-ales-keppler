# 001 · Base común — plan técnico

## Archivos

| Archivo | Contenido |
|---|---|
| `src/comun/descargar_koi.py` | CLI de descarga TAP |
| `src/comun/construir_ejemplos.py` | CLI que escribe `compartido/ejemplos.csv` |
| `src/comun/lineas_base_tabulares.py` | CLI de S3-B, S3-C y `koi_score` |
| `src/comun/verificar_contrato.py` | CLI de verificación |
| `src/comun/utilidades.py` | `sha256_archivo`, `escribir_manifiesto`, `validar_kepid`, `ruta_dentro_de` |
| `tests/test_base_comun.py` | Pruebas con tablas pequeñas construidas en memoria |

## Firmas

```python
# utilidades.py
def sha256_archivo(ruta: Path) -> str
def escribir_manifiesto(ruta: Path, argumentos: dict, entradas: list[Path], extra: dict | None = None) -> None
    # agrega fecha UTC, commit git (o "sin-git"), versiones de numpy/pandas/sklearn/keras/lightkurve
def validar_kepid(valor) -> int          # entero entre 1 y 99_999_999; si no, ValueError
def ruta_dentro_de(base: Path, ruta: Path) -> Path   # resuelve y exige que quede bajo base

# descargar_koi.py
COLUMNAS: tuple[str, ...]
def construir_consulta(tabla: str, columnas: tuple[str, ...]) -> str     # tabla ∈ {"q1_q17_dr25_koi", "cumulative"}
def descargar(tabla: str, salida: Path, reintentos: int = 3, espera_s: int = 120) -> pd.DataFrame

# construir_ejemplos.py
def tipo_fp(fila) -> str
def marcar_prerregistrados(dr25: pd.DataFrame, acumulada: pd.DataFrame) -> pd.Series
def asignar_particiones(kois: pd.DataFrame, n: int, semilla: int) -> np.ndarray
def construir(seleccion: pd.DataFrame, dr25: pd.DataFrame, acumulada: pd.DataFrame, protocolo: dict) -> pd.DataFrame

# lineas_base_tabulares.py
def predecir_oof(X: pd.DataFrame, y: np.ndarray, particion: np.ndarray, extra_X=None, extra_y=None, extra_kepid=None, kepid=None) -> np.ndarray
def evaluar(puntaje, y, kepid, protocolo) -> dict     # usa comun.metricas.auc y comun.estadistica

# verificar_contrato.py
def verificar(raiz: Path, protocolo: dict) -> list[str]   # lista de fallos; vacía = pasa
```

## Decisiones técnicas

- TAP síncrono con `urllib.request` (sin dependencias nuevas), `format=csv`, URL codificada con `urllib.parse.urlencode`. El nombre de la tabla sale de una lista cerrada, nunca de texto libre.
- Las líneas base usan como validación de parada temprana la validación interna de `comun.protocolo.separar` (mismo esquema que las redes).
- En la modalidad (ii) se excluyen del entrenamiento todas las filas con `kepid` en la partición de prueba o de validación, y las prerregistradas.
- `verificar_contrato.py` no importa Keras: abre los `.npz` y cuenta archivos. Así corre en segundos.

## Revisión de seguridad

Tabla y columnas de lista cerrada (sin inyección en ADQL). `kepid` validado. Sin escritura fuera de las rutas dadas. Sin `pickle`.

## Orden

Utilidades → descarga → ejemplos → verificación → líneas base.
