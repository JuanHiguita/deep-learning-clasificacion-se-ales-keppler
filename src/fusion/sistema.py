"""El sistema completo: de las entradas de una señal a su respuesta conjunta (spec 011).

    entradas ─► 4 codificadores congelados ─► cabeza de fusión ─► temperatura ─► prior explícito

Dos pasos:

1. `empaquetar(...)` reúne en `modelos/sistema/` todo lo que la inferencia necesita y que no
   estaba guardado como archivo: la estandarización de la entrada de la cabeza, la temperatura y
   la tasa de planetas de entrenamiento de cada partición, y una copia de las cabezas.
2. `Sistema.cargar(...).responder(entradas)` produce la respuesta. No necesita los datos de
   entrenamiento.

Hay un sistema por partición (codificadores y cabeza que nunca vieron las estrellas de esa
partición). Para una estrella del conjunto se usa el de su partición; para una estrella nueva se
promedian los cinco.

Fuera de alcance (es de la tesis, no de este curso): combinar P(planeta) con
P(en zona habitable y rocoso). Aquí todas las señales son de periodo corto.

Uso:
  python -I src/fusion/sistema.py --modelos modelos --dataset data/dataset_temporal.npz \
      --ejemplos compartido/ejemplos.csv --resultados resultados/dl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from comun import entrenamiento as ent  # noqa: E402
from comun import metricas  # noqa: E402
from comun.protocolo import cargar_protocolo, separar  # noqa: E402
from comun.utilidades import sha256_archivo  # noqa: E402
from fusion import fusionar  # noqa: E402
from fusion.calibracion import ajustar_temperatura, aplicar_temperatura  # noqa: E402
from fusion.vectores import RAMAS, armar_entrada, cargar_vectores  # noqa: E402
from temporal import modelos as _capas_propias  # noqa: E402, F401  (registra las capas al importarse)

CARPETA = "sistema"
# Entrada que dice si la rama tiene dato para cada ejemplo (las que faltan valen "siempre")
DISPONIBLE = {
    "vistas": "disponible_vistas",
    "transitos": "disponible_transitos",
    "imagen": "disponible_imagen",
}


# ---------------------------------------------------------------------------------------------
# 1. Empaquetar
# ---------------------------------------------------------------------------------------------


def empaquetar(
    carpeta_modelos: Path, dataset: dict, ejemplos: pd.DataFrame, resultados: Path, protocolo: dict
) -> dict:
    """Escribe `modelos/sistema/`: lo que falta para predecir sin los datos de entrenamiento."""
    destino = Path(carpeta_modelos) / CARPETA
    destino.mkdir(parents=True, exist_ok=True)
    n_particiones = protocolo["particiones"]["n"]
    tabla = ent.alinear(ejemplos, dataset["id_ejemplo"])
    y, particion = tabla["clase"].to_numpy().astype(int), tabla["particion"].to_numpy()
    fusion = Path(resultados) / "fusion" / fusionar.CONFIGURACION_PRINCIPAL
    with np.load(fusion / "pred_semilla0.npz", allow_pickle=False) as archivo:
        pred = {clave: archivo[clave] for clave in archivo.files}

    manifiesto = {"versiones": fusionar.VERSIONES, "ramas": list(RAMAS), "particiones": {}}
    for k in range(n_particiones):
        sep = separar(particion, k, n_particiones)
        # estandarización de la entrada de la cabeza: la misma que usó `fusionar`
        vectores, banderas = cargar_vectores(
            carpeta_modelos, fusionar.VERSIONES, k, dataset["id_ejemplo"], dataset
        )
        x = armar_entrada(vectores, banderas, RAMAS)
        n_vectores = x.shape[1] - len(RAMAS)
        media, desviacion = ent.estadisticas_estandarizacion(x[sep.entrenamiento, :n_vectores])
        # temperatura: ajustada SOLO con la validación interna del modelo de la partición k
        suya = pred["particion_modelo_validacion"] == k
        temperatura = ajustar_temperatura(pred["p_validacion"][suya].astype(float), y[suya])
        tasa_entrenamiento = float((y[sep.entrenamiento & (y < 2)] == 0).mean())

        np.savez(
            destino / f"particion_{k}.npz",
            media=media,
            desviacion=desviacion,
            temperatura=temperatura,
            tasa_entrenamiento=tasa_entrenamiento,
        )
        shutil.copyfile(fusion / "cabezas" / f"particion_{k}.keras", destino / f"cabeza_{k}.keras")
        manifiesto["particiones"][str(k)] = {
            "temperatura": round(temperatura, 6),
            "tasa_entrenamiento": round(tasa_entrenamiento, 6),
            "sha256_cabeza": sha256_archivo(destino / f"cabeza_{k}.keras"),
            "sha256_codificadores": {
                rama: sha256_archivo(
                    Path(carpeta_modelos) / rama / fusionar.VERSIONES[rama] / f"particion_{k}.keras"
                )
                for rama in RAMAS
            },
        }
    (destino / "manifiesto.json").write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return manifiesto


# ---------------------------------------------------------------------------------------------
# 2. Responder
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Respuesta:
    """Respuesta conjunta del sistema para n señales."""

    p: np.ndarray  # (n, 3) calibrada: planeta, falso positivo, sin señal
    p_sin_calibrar: np.ndarray  # (n, 3) salida directa de la cabeza
    q_planeta: np.ndarray  # (n,) P(planeta | hay señal), con la tasa base del entrenamiento
    q_con_tasa_base: np.ndarray | None  # (n,) lo mismo, corregido a la tasa base declarada
    entropia: np.ndarray  # (n,) duda del sistema: 0 = seguro
    ramas_con_dato: np.ndarray  # (n, 4) 1 si la rama tuvo dato, en el orden de `RAMAS`
    vectores: dict[str, np.ndarray]  # rama → vector que produjo su codificador


@dataclass(frozen=True)
class _Codificador:
    modelo: keras.Model
    entradas: tuple[str, ...]
    estandarizacion: dict[str, tuple[np.ndarray, np.ndarray]]


class _SistemaDeParticion:
    """Codificadores, cabeza, temperatura y prior de una partición."""

    def __init__(self, carpeta_modelos: Path, k: int, manifiesto: dict):
        self.k = k
        datos = manifiesto["particiones"][str(k)]
        self.codificadores = {}
        for rama in RAMAS:
            carpeta = carpeta_modelos / rama / manifiesto["versiones"][rama]
            ruta = carpeta / f"particion_{k}.keras"
            if sha256_archivo(ruta) != datos["sha256_codificadores"][rama]:
                raise ValueError(f"{ruta} no coincide con la huella del manifiesto del sistema")
            declarado = json.loads((carpeta / "manifiesto.json").read_text(encoding="utf-8"))
            estandarizacion = {}
            ruta_est = carpeta / f"estandarizacion_particion_{k}.npz"
            if ruta_est.is_file():
                with np.load(ruta_est, allow_pickle=False) as archivo:
                    for clave in declarado["estandarizadas"]:
                        estandarizacion[clave] = (
                            archivo[f"{clave}_media"],
                            archivo[f"{clave}_desviacion"],
                        )
            self.codificadores[rama] = _Codificador(
                keras.saving.load_model(ruta, safe_mode=True),
                tuple(declarado["entradas_modelo"]),
                estandarizacion,
            )
        ruta_cabeza = carpeta_modelos / CARPETA / f"cabeza_{k}.keras"
        if sha256_archivo(ruta_cabeza) != datos["sha256_cabeza"]:
            raise ValueError(f"{ruta_cabeza} no coincide con la huella del manifiesto del sistema")
        self.cabeza = keras.saving.load_model(ruta_cabeza, safe_mode=True)
        with np.load(carpeta_modelos / CARPETA / f"particion_{k}.npz", allow_pickle=False) as a:
            self.media, self.desviacion = a["media"], a["desviacion"]
            self.temperatura = float(a["temperatura"])
            self.tasa_entrenamiento = float(a["tasa_entrenamiento"])

    def _vector(self, rama: str, entradas: dict, con_dato: np.ndarray) -> np.ndarray:
        """Vector de una rama; ceros donde la rama no tiene dato."""
        codificador = self.codificadores[rama]
        dimension = codificador.modelo.output_shape[-1]
        vector = np.zeros((len(con_dato), dimension), np.float32)
        if not con_dato.any():
            return vector
        faltantes = entradas.get("faltantes") if rama == "escalares" else None
        lote = {}
        for clave in codificador.entradas:
            valores = entradas[clave][con_dato]
            if clave in codificador.estandarizacion:
                media, desviacion = codificador.estandarizacion[clave]
                mascara = None if faltantes is None else faltantes[con_dato]
                valores = ent.aplicar_estandarizacion(valores, media, desviacion, mascara)
            lote[clave] = valores
        vector[con_dato] = codificador.modelo.predict(lote, verbose=0, batch_size=256)
        return vector

    def responder(self, entradas: dict, n: int) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
        """(p sin calibrar, p calibrada, vectores, banderas) de n señales."""
        banderas = np.ones((n, len(RAMAS)), np.float32)
        vectores = {}
        for j, rama in enumerate(RAMAS):
            if rama in DISPONIBLE and DISPONIBLE[rama] in entradas:
                banderas[:, j] = entradas[DISPONIBLE[rama]].astype(np.float32)
            vectores[rama] = self._vector(rama, entradas, banderas[:, j] > 0)

        x = armar_entrada(vectores, banderas, RAMAS)
        n_vectores = x.shape[1] - len(RAMAS)
        x[:, :n_vectores] = ent.aplicar_estandarizacion(
            x[:, :n_vectores], self.media, self.desviacion
        )
        inicio = 0
        for j, rama in enumerate(RAMAS):  # un vector ausente sigue en cero tras estandarizar
            ancho = vectores[rama].shape[1]
            x[banderas[:, j] == 0, inicio : inicio + ancho] = 0.0
            inicio += ancho

        p = np.asarray(self.cabeza.predict(x, verbose=0, batch_size=256), dtype=float)
        return p, aplicar_temperatura(p, self.temperatura), vectores, banderas


class Sistema:
    """Los cinco sistemas de partición y la lógica para elegir o promediar."""

    def __init__(self, carpeta_modelos: Path):
        self.carpeta_modelos = Path(carpeta_modelos)
        ruta = self.carpeta_modelos / CARPETA / "manifiesto.json"
        self.manifiesto = json.loads(ruta.read_text(encoding="utf-8"))
        self._cargados: dict[int, _SistemaDeParticion] = {}

    @property
    def particiones(self) -> list[int]:
        return sorted(int(k) for k in self.manifiesto["particiones"])

    def _de_particion(self, k: int) -> _SistemaDeParticion:
        if k not in self._cargados:
            self._cargados[k] = _SistemaDeParticion(self.carpeta_modelos, k, self.manifiesto)
        return self._cargados[k]

    def responder(
        self, entradas: dict, particion: int | None = None, tasa_base: float | None = None
    ) -> Respuesta:
        """Respuesta conjunta para las señales de `entradas`.

        entradas: claves de `dataset_temporal.npz` y, si hay imagen, `img` (n, 11, 11, 4) con
            `disponible_imagen` (n,). Sin esas dos claves la rama 3 se da por ausente.
        particion: partición de prueba de estas señales (0–4). `None` = estrella nueva: se
            promedian los cinco sistemas.
        tasa_base: fracción de planetas que se declara para la población (prior explícito).
        """
        n = len(entradas["escalares"])
        if "img" not in entradas:
            entradas = {**entradas, "disponible_imagen": np.zeros(n, np.float32)}
        usados = self.particiones if particion is None else [particion]

        sin_calibrar, calibradas, q_corregidas = [], [], []
        for k in usados:
            sistema = self._de_particion(k)
            p, p_calibrada, vectores, banderas = sistema.responder(entradas, n)
            sin_calibrar.append(p)
            calibradas.append(p_calibrada)
            if tasa_base is not None:
                q = metricas.q_planeta(p_calibrada)
                q_corregidas.append(
                    metricas.corregir_prior(q, sistema.tasa_entrenamiento, tasa_base)
                )

        p = np.mean(calibradas, axis=0)
        return Respuesta(
            p=p,
            p_sin_calibrar=np.mean(sin_calibrar, axis=0),
            q_planeta=metricas.q_planeta(p),
            q_con_tasa_base=np.mean(q_corregidas, axis=0) if q_corregidas else None,
            entropia=metricas.entropia(p),
            ramas_con_dato=banderas,
            vectores=vectores,  # los de la última partición usada
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Empaqueta el sistema completo en modelos/sistema")
    ap.add_argument("--modelos", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--ejemplos", type=Path, required=True)
    ap.add_argument("--resultados", type=Path, required=True)
    ap.add_argument("--protocolo", type=Path, default=Path("config/protocolo.json"))
    a = ap.parse_args(argv)
    with np.load(a.dataset, allow_pickle=False) as archivo:
        dataset = {clave: archivo[clave] for clave in archivo.files}
    manifiesto = empaquetar(
        a.modelos, dataset, pd.read_csv(a.ejemplos), a.resultados, cargar_protocolo(a.protocolo)
    )
    for k, datos in manifiesto["particiones"].items():
        print(
            f"partición {k}: T = {datos['temperatura']:.3f} · tasa = {datos['tasa_entrenamiento']:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
