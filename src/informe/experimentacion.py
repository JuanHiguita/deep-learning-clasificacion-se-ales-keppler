"""Figuras del notebook `02_experimentacion_hiperparametros.ipynb` (spec 009).

Leen `resultados/dl/busqueda/<arquitectura>/` y `resultados/dl/tablas/busqueda_*.csv`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from informe.datos import Resultados
from informe.graficos import COLOR, bosque, color_si_distingue

NOMBRE = {
    "m1_astronet": "Rama 1 · CNN tipo AstroNet",
    "m2_promedio": "Rama 2 · promedio",
    "m2_gru": "Rama 2 · GRU",
    "m2_atencion": "Rama 2 · atención",
    "m4_escalares": "Rama 4 · escalares",
    "cabeza_fusion": "Cabeza de fusión",
}
NOMBRE_DE_HIPERPARAMETRO = {
    "tasa_aprendizaje": "tasa de aprendizaje",
    "weight_decay": "decaimiento de pesos",
    "lote": "tamaño de lote",
    "dropout": "dropout",
    "ancho": "ancho (multiplica los filtros)",
    "pooling": "pooling",
    "unidades_gru": "unidades de la GRU",
    "cabezas": "cabezas de atención",
    "oculta": "neuronas ocultas",
}
N_PARTICIONES = 5
COLUMNAS_NLL = [f"nll_val_k{k}" for k in range(N_PARTICIONES)]


def arquitecturas_disponibles(res: Resultados) -> list[str]:
    """Arquitecturas cuya búsqueda ya tiene resumen, en el orden de `NOMBRE`."""
    base = res.carpeta / "busqueda"
    return [a for a in NOMBRE if (base / a / "resumen.json").is_file()]


def hiperparametros_de(res: Resultados, arquitectura: str) -> list[str]:
    """Hiperparámetros que se buscaron para una arquitectura."""
    return list(res.json("busqueda", arquitectura, "configuraciones.json")[0])


def _configuraciones(res: Resultados, arquitectura: str) -> pd.DataFrame:
    return pd.read_csv(res.carpeta / "busqueda" / arquitectura / "configuraciones.csv")


def configuraciones(res: Resultados, arquitectura: str) -> pd.DataFrame:
    """NLL de validación interna de cada configuración, partición por partición.

    Cada punto es un modelo (una partición). El rombo es la media de la configuración. La
    configuración 0 (naranja) es la del protocolo. Los puntos con borde negro son los que
    eligió la selección anidada: la mejor de cada partición.
    """
    tabla = _configuraciones(res, arquitectura)
    resumen = res.json("busqueda", arquitectura, "resumen.json")
    orden = tabla.sort_values("nll_val_media").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 3.4))
    for posicion, fila in orden.iterrows():
        color = COLOR["demo"] if fila.configuracion == "cfg00" else COLOR["curso"]
        valores = fila[COLUMNAS_NLL].to_numpy(dtype=float)
        ax.scatter([posicion] * N_PARTICIONES, valores, s=22, color=color, alpha=0.55)
        ax.scatter(posicion, fila.nll_val_media, marker="D", s=46, color=color, zorder=3)
        for k, elegida in enumerate(resumen["elegida_por_particion"]):
            if elegida == fila.configuracion:
                ax.scatter(posicion, valores[k], s=70, facecolor="none", edgecolor="k", zorder=4)
    ax.set_xticks(range(len(orden)), orden.configuracion)
    ax.set_ylabel("NLL en validación interna")
    ax.set_title(
        f"{NOMBRE[arquitectura]}: configuraciones ordenadas de mejor a peor "
        "(naranja = protocolo; borde negro = elegida por su partición)"
    )
    plt.tight_layout()
    plt.show()

    ocultar = [*COLUMNAS_NLL, "auc_prueba", "nll_prueba"]  # la prueba no se mira para elegir
    return orden.drop(columns=ocultar)


def sensibilidad(res: Resultados, arquitectura: str, hiperparametro: str) -> None:
    """¿Importa este hiperparámetro? NLL de validación frente a su valor.

    Con 8 configuraciones y varios hiperparámetros cambiando a la vez, esto solo sugiere
    tendencias: no aísla el efecto de cada uno.
    """
    tabla = _configuraciones(res, arquitectura)
    if hiperparametro not in tabla:
        buscados = ", ".join(hiperparametros_de(res, arquitectura))
        print(f"«{hiperparametro}» no se buscó en {NOMBRE[arquitectura]}. Se buscaron: {buscados}.")
        return
    valores = tabla[hiperparametro]
    es_el_protocolo = tabla.configuracion == "cfg00"
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    if pd.api.types.is_numeric_dtype(valores) and valores.nunique() > 4:
        x = valores.to_numpy(dtype=float)
        ax.set_xscale("log")
    else:  # pocos valores o texto: se tratan como categorías
        categorias = sorted(valores.unique())
        x = np.array([categorias.index(v) for v in valores], dtype=float)
        ax.set_xticks(range(len(categorias)), [str(c) for c in categorias])
    ax.scatter(
        x[~es_el_protocolo], tabla.nll_val_media[~es_el_protocolo], s=45, color=COLOR["curso"]
    )
    ax.scatter(
        x[es_el_protocolo],
        tabla.nll_val_media[es_el_protocolo],
        s=70,
        color=COLOR["demo"],
        label="protocolo",
    )
    for xi, fila in zip(x, tabla.itertuples(), strict=True):
        ax.annotate(
            fila.configuracion[3:],
            (xi, fila.nll_val_media),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
    ax.set_xlabel(NOMBRE_DE_HIPERPARAMETRO.get(hiperparametro, hiperparametro))
    ax.set_ylabel("NLL media en validación interna")
    ax.set_title(f"{NOMBRE[arquitectura]}: sensibilidad (el número es la configuración)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def curvas(res: Resultados, arquitectura: str, configuracion: str) -> None:
    """Curvas de pérdida de una configuración, junto a las del protocolo."""
    carpeta = ("busqueda", arquitectura)
    fig, ejes = plt.subplots(1, 2, figsize=(11, 3.0), sharey=True)
    for ax, nombre in zip(ejes, ("cfg00", configuracion), strict=True):
        ruta = res.carpeta.joinpath(*carpeta, nombre, "historial_semilla0.json")
        if not ruta.is_file():
            ax.text(0.5, 0.5, "sin historial guardado", ha="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        for particion in res.json(*carpeta, nombre, "historial_semilla0.json"):
            ax.plot(particion["perdida_entrenamiento"], color=COLOR["planeta"], alpha=0.5, lw=0.9)
            ax.plot(particion["perdida_validacion"], color=COLOR["fp"], alpha=0.6, lw=0.9)
            ax.axvline(particion["epoca_elegida"] - 1, color="k", alpha=0.15, lw=0.8)
        ax.set_title("protocolo (cfg00)" if nombre == "cfg00" else nombre)
        ax.set_xlabel("época")
    ejes[0].set_ylabel("entropía cruzada")
    plt.suptitle(
        f"{NOMBRE[arquitectura]} · azul: entrenamiento · rojo: validación interna",
        y=1.03,
        fontsize=10,
    )
    plt.tight_layout()
    plt.show()


def fijo_contra_busqueda(res: Resultados) -> pd.DataFrame:
    """Resultado en PRUEBA: protocolo fijo frente a selección anidada, y su diferencia pareada."""
    tabla = res.csv("busqueda_resumen")
    auc = tabla[tabla.metrica == "auc_prueba"]
    arquitecturas = [a for a in NOMBRE if a in set(auc.arquitectura)]

    def de(version: str) -> pd.DataFrame:
        return auc[auc.version == version].set_index("arquitectura").loc[arquitecturas]

    fijo, buscado, diferencia = de("fijo"), de("con_busqueda"), de("diferencia")
    nombres = [NOMBRE[a] for a in arquitecturas]
    fig, (ax_auc, ax_dif) = plt.subplots(1, 2, figsize=(14, 0.55 * len(nombres) + 2.2))

    etiquetas, est, inf, sup, colores = [], [], [], [], []
    for a, nombre in zip(arquitecturas, nombres, strict=True):
        for version, datos, color in (
            ("protocolo", fijo, COLOR["demo"]),
            ("con búsqueda", buscado, COLOR["curso"]),
        ):
            etiquetas.append(f"{nombre} · {version}")
            est.append(datos.loc[a, "estimacion"])
            inf.append(datos.loc[a, "ic_inf"])
            sup.append(datos.loc[a, "ic_sup"])
            colores.append(color)
    bosque(ax_auc, etiquetas, est, inf, sup, colores)
    ax_auc.set_xlabel("AUC planeta vs FP en prueba (semilla 0), IC 95 %")
    ax_auc.set_title("Protocolo fijo (naranja) frente a búsqueda (morado)")

    distinguible = ~((diferencia.ic_inf <= 0) & (0 <= diferencia.ic_sup))
    bosque(
        ax_dif,
        nombres,
        diferencia.estimacion,
        diferencia.ic_inf,
        diferencia.ic_sup,
        color_si_distingue(distinguible),
        referencia=0,
    )
    ax_dif.set_xlabel("AUC con búsqueda − AUC del protocolo (pareada)")
    ax_dif.set_title("¿La búsqueda mejora? (verde = se distingue de cero)")
    plt.tight_layout()
    plt.show()

    salida = pd.DataFrame(
        {
            "modelo": nombres,
            "AUC protocolo": fijo.estimacion.to_numpy(),
            "AUC con búsqueda": buscado.estimacion.to_numpy(),
            "diferencia": diferencia.estimacion.to_numpy(),
            "IC inf": diferencia.ic_inf.to_numpy(),
            "IC sup": diferencia.ic_sup.to_numpy(),
            "se distingue de 0": distinguible.to_numpy(),
        }
    )
    return salida


def pares_con_busqueda(res: Resultados) -> pd.DataFrame:
    """Pregunta 1 otra vez: diferencias entre arquitecturas cuando TODAS tuvieron su búsqueda."""
    tabla = res.csv("busqueda_pares")
    ramas_temporales = ("m1_astronet", "m2_promedio", "m2_gru", "m2_atencion")
    es_temporal = tabla.par.apply(
        lambda par: all(lado in ramas_temporales for lado in par.split(" − "))
    )
    tabla = tabla[es_temporal].reset_index(drop=True)
    etiquetas = [
        " − ".join(
            NOMBRE[lado].replace("Rama 2 · ", "").replace("Rama 1 · ", "")
            for lado in par.split(" − ")
        )
        for par in tabla.par
    ]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(tabla) + 1.6))
    bosque(
        ax,
        etiquetas,
        tabla.estimacion,
        tabla.ic_inf,
        tabla.ic_sup,
        color_si_distingue(tabla.distinguible),
        referencia=0,
    )
    ax.set_xlabel("diferencia de AUC en prueba (pareada, semilla 0), IC 95 %")
    ax.set_title(
        "Arquitecturas temporales, cada una con su búsqueda (verde = se distingue de cero)"
    )
    plt.tight_layout()
    plt.show()
    return tabla.assign(par=etiquetas)


def modelos_guardados(res: Resultados) -> pd.DataFrame:
    """Qué modelo quedó guardado para cada arquitectura y partición (`modelos/mejores/`)."""
    filas = []
    base = res.raiz / "modelos" / "mejores"
    for arquitectura in NOMBRE:
        ruta = base / arquitectura / "manifiesto.json"
        if not ruta.is_file():
            continue
        manifiesto = pd.read_json(ruta, typ="series")
        for k, datos in manifiesto["particiones"].items():
            filas.append(
                {
                    "modelo": NOMBRE[arquitectura],
                    "partición": int(k),
                    "archivo": f"modelos/mejores/{arquitectura}/{datos['archivo']}",
                    "configuración": datos["configuracion"],
                    **datos["hiperparametros"],
                }
            )
    return pd.DataFrame(filas)


def espacio_de_busqueda(arquitectura: str) -> pd.DataFrame:
    """El espacio declarado antes de entrenar y el valor del protocolo para cada hiperparámetro."""
    if arquitectura == "cabeza_fusion":
        from fusion import busqueda_cabeza

        espacio, fija = busqueda_cabeza.ESPACIO, busqueda_cabeza.FIJA
    else:
        from temporal import busqueda_ramas

        espacio, fija = busqueda_ramas.ESPACIOS[arquitectura]
    filas = []
    for nombre, opciones in espacio.items():
        if isinstance(opciones, tuple):
            valores = f"continuo entre {opciones[1]:g} y {opciones[2]:g} (escala logarítmica)"
        else:
            valores = ", ".join(str(o) for o in opciones)
        filas.append(
            {
                "hiperparámetro": NOMBRE_DE_HIPERPARAMETRO.get(nombre, nombre),
                "valores posibles": valores,
                "protocolo": fija[nombre],
            }
        )
    return pd.DataFrame(filas)


def probar_modelo_guardado(res: Resultados, arquitectura: str, particion: int = 0) -> pd.DataFrame:
    """Carga un modelo guardado, predice su partición de prueba y lo compara con lo reportado."""
    from temporal import despliegue

    guardado = despliegue.cargar(res.raiz / "modelos" / "mejores", arquitectura, particion)
    experimento = despliegue.EXPERIMENTOS[arquitectura][0]
    disponible = np.ones(len(res.ids), dtype=bool)
    if experimento.clave_disponible is not None:
        disponible = res.dataset[experimento.clave_disponible].astype(bool)
    filas = np.flatnonzero((res.tabla.particion.to_numpy() == particion) & disponible)
    entradas = {c: v[filas] for c, v in res.dataset.items() if len(v) == len(res.ids)}
    p_guardado = guardado.predecir(entradas)
    reportado = res.npz("busqueda", arquitectura, "pred_con_busqueda.npz")["p_prueba"][filas]

    diferencia = float(np.abs(p_guardado - reportado).max())
    print(
        f"{NOMBRE[arquitectura]}, partición {particion}: {len(filas)} ejemplos que el modelo nunca "
        f"vio. Diferencia máxima con las probabilidades reportadas: {diferencia:.1e}"
    )
    print("Hiperparámetros del modelo guardado:", guardado.hiperparametros)
    return pd.DataFrame(
        {
            "id_ejemplo": res.ids[filas],
            "clase real": [("Planeta", "Falso positivo", "Sin señal")[c] for c in res.y[filas]],
            "P(planeta) del modelo cargado": p_guardado[:, 0].round(3),
            "P(planeta) reportada": reportado[:, 0].round(3),
        }
    ).head(8)
