# 002 · Dataset temporal — plan técnico

## Archivos

| Archivo | Contenido |
|---|---|
| `src/temporal/descargar_llc.py` | CLI de descarga |
| `src/temporal/curvas.py` | Funciones puras sobre arreglos `numpy` (sin `lightkurve`), para poder probarlas |
| `src/temporal/lectura.py` | Único punto que usa `lightkurve`: lee FITS y devuelve `tiempo, flujo, trimestre` |
| `src/temporal/construir_dataset_temporal.py` | CLI que orquesta |
| `tests/sinteticos.py` | Generador de curvas de prueba |
| `tests/test_extractor.py` | Pruebas |

## Firmas (`curvas.py`)

```python
@dataclass(frozen=True)
class Efemerides:
    periodo: float      # días
    t0: float           # BKJD
    duracion: float     # días

def mascara_transitos(t, efemerides: list[Efemerides], factor: float = 1.5) -> np.ndarray
def aplanar(t, f, enmascarado, ventana: int = 401) -> np.ndarray          # Savitzky–Golay por tramos continuos
def ruido_robusto(f, fuera_de_transito) -> float                           # 1.4826 · MAD
def fase(t, ef: Efemerides) -> np.ndarray                                  # días, en [-P/2, P/2)
def medir_profundidad(t, f, ef, sigma) -> tuple[float, float, int]         # profundidad, sigma_local, n_puntos
def escala(profundidad, sigma_local, min_sigmas: float = 3.0) -> float
def vista_plegada(fase_d, f, centro, semiancho, n_puntos) -> tuple[np.ndarray, float]   # serie y fracción de vacíos
def numero_de_transito(t, ef) -> np.ndarray                                # entero por cadencia
def ventanas_por_transito(t, f, ef, otras: list[Efemerides], cfg) -> list[Ventana]
def seleccionar_transitos(ventanas: list[Ventana], maximo: int = 32) -> list[Ventana]
def estadisticos_consistencia(ventanas: list[Ventana], sigma) -> np.ndarray     # (4,)
def construir_ejemplo(t, f, trimestre, ef, otras, cfg, es_sin_senal: bool) -> dict   # todas las claves de un ejemplo
```

`Ventana`: `indice` (número de tránsito), `centro`, `serie` (61,), `profundidad`, `error`, `valida`, `motivo`.

## Decisiones técnicas

- **Aplanado por tramos:** Savitzky–Golay se aplica por trimestre y, dentro de él, por tramos sin huecos mayores de 0,5 d. Los puntos enmascarados no entran al ajuste; la tendencia sobre ellos se interpola.
- **Plegado por mediana** con `np.digitize` y agrupación con `np.bincount`/ordenación; nada de bucles por punto.
- **Interpolación de la ventana:** `np.interp` sobre una rejilla de 61 puntos en `[−2,5; +2,5] · duración`.
- **Profundidad por tránsito:** −media de los puntos con `|t − centro| < 0,25 · duración`; error = `σ / √n`.
- **Paridad:** la del índice entero del tránsito respecto a `t0`.
- **Ejemplo `sin_senal`:** `Efemerides(P, t0 + 0.25·P, dur)`; la lista `otras` incluye la efeméride real del propio KOI, de modo que RF-12c descarta ventanas que toquen el tránsito real.
- **Semilla:** el extractor es determinista; no usa números aleatorios.
- Los valores de `config/protocolo.json` → `extractor` se pasan como `cfg`; no hay constantes repetidas en el código.

## Pruebas (`tests/test_extractor.py`, datos de `tests/sinteticos.py`)

| Prueba | Entrada sintética | Se exige |
|---|---|---|
| Profundidad | Caja de 1.000 ppm, P = 5 d, ruido 200 ppm, 4 años | `profundidad_medida` dentro del 10 % |
| Escala | La misma | Mínimo de `vista_local[:, 0]` entre −1,2 y −0,8 |
| Par/impar | Impares 1.400 ppm, pares 600 ppm | `consistencia[1]` > 5; canales 1 y 2 de la vista local difieren |
| Planeta | Todos iguales | `|consistencia[1]|` < 3 |
| Huecos | Se borra el 50 % de una ventana | Ese tránsito sale inválido, motivo (a) |
| Otro KOI | Segundo planeta que coincide con un tránsito | Ese tránsito sale inválido, motivo (c) |
| Sin señal | Época desplazada | `snr_medido` < 3; ninguna ventana toca el tránsito real |
| Selección | 100 válidos | 32 elegidos, 16 y 16, orden cronológico, determinista |
| Formas | Cualquiera | Las de `config/protocolo.json`; sin NaN |
| Secundario | Eclipse en fase 0,5 | Canal 3 de la vista local con mínimo < −0,2 |

## Revisión de seguridad

`kepid` validado antes de consultar. Las descargas van a la carpeta dada por argumento (`ruta_dentro_de`). FITS leídos con `lightkurve`/`astropy`, que no ejecutan código. `np.savez_compressed` con arreglos sin objetos.

## Rendimiento esperado

260 estrellas × hasta 17 trimestres de unas 4.400 cadencias. El extractor debe tardar menos de 1 s por ejemplo; si no, revisar bucles.
