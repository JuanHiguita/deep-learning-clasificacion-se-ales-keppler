# 004 · Ramas y comparación de arquitecturas — plan técnico

## Archivos

| Archivo | Contenido |
|---|---|
| `src/temporal/modelos.py` | Constructores de M1, M1r, M2p, M2g, M2a, M4 y capas propias |
| `src/temporal/aumentacion.py` | Funciones de aumentación |
| `src/temporal/linea_base_demo.py` | M0 |
| `src/temporal/exp_ramas.py` | CLI: E0, E1, E2, selección y exportación |
| `src/temporal/exp_sesgos.py` | CLI: E3, E4, E5 |
| `tests/test_modelos.py` | Formas, máscaras, invariancia |

## Firmas

```python
# modelos.py
def codificador_por_transito() -> keras.Model                                  # (61, 1) → (32,)
def m1_astronet(semilla: int, pooling: str = "max") -> keras.Model             # pooling ∈ {"max", "promedio", "paso"}
def m1r_bigru(semilla: int) -> keras.Model
def m2(semilla: int, agregador: str, usar_consistencia: bool = True) -> keras.Model   # agregador ∈ {"promedio", "gru", "atencion"}
def m4_escalares(semilla: int, d: int = 7) -> keras.Model
def submodelo_pesos_atencion(modelo: keras.Model) -> keras.Model               # → (n, 32)

class MediaEnmascarada(keras.layers.Layer)       # call(x, mascara) → media sobre el eje de tránsitos
class AtencionAgregadora(keras.layers.Layer)     # call(x, mascara) → (vector, pesos)

# aumentacion.py
def aumentar_vistas(entradas, rng, cfg) -> Entradas
def aumentar_transitos(entradas, rng, cfg) -> Entradas

# exp_sesgos.py
def permutar_transitos(entradas, rng) -> Entradas                 # permuta ventanas, máscara y paridad juntas; los vacíos quedan al final
def reducir_transitos(entradas, n: int, rng) -> Entradas
def desplazar_vista(entradas, puntos: int) -> Entradas
```

## Decisiones técnicas

- **Capas propias con `keras.ops`** (no `jax.numpy` directo), registradas con `@keras.saving.register_keras_serializable(package="temporal")` para que los `.keras` se puedan cargar con `safe_mode=True`.
- **Máscara:** entra como entrada explícita `(32,)`. Para `MultiHeadAttention` se construye `attention_mask` de forma `(lote, 32, 32)` con el producto externo. Para la GRU se pasa `mask=mascara > 0`. Tras φ, las posiciones vacías se multiplican por 0.
- **Entrada de φ:** `transitos[..., None]` → `(32, 61, 1)`.
- **Pooling "paso":** se quita el `MaxPool` y la segunda convolución del bloque usa `strides=2`. Mismo número de filtros.
- **M0:** se reconstruye la entrada `seq` (61, 2) y `img` con las funciones de `src/demo/demo_dataset.py` desde los TPF de Q5; se guarda en `data/dataset_demo.npz` con `id_ejemplo`. El modelo es `modelo(usar_img, usar_seq)` de la demo, envuelto para tener una capa `embedding`. Si no hay TPF, M0 se omite y se anota: **no se sustituye por las cifras antiguas de la demo**.
- **E3–E5 no entrenan** (salvo las 2 variantes extra de pooling): recargan los modelos de E1/E2 guardados por partición y semilla, y vuelven a predecir. Por eso `exp_ramas.py` guarda los modelos completos en `resultados/dl/<exp>/modelos/` (fuera de git).

## Pruebas (`tests/test_modelos.py`)

- Formas de salida y nombre de la capa `embedding` de cada modelo.
- M2p y M2a: permutar los tránsitos (con su máscara) no cambia la salida (1e-5). M2g: sí la cambia.
- Cambiar el contenido de una ventana con máscara 0 no cambia la salida en ningún M2.
- Guardar y cargar cada modelo con `safe_mode=True` conserva las predicciones.
- Número de parámetros de cada modelo por debajo de 200.000 (guarda contra un error de `Flatten`).

## Presupuesto

Prioridad A: 6 configuraciones × 15 corridas = 90, más 5 de M0 completo. Prioridad B: 5 configuraciones × 15 = 75. Tiempo por corrida: se mide en T3.11.
