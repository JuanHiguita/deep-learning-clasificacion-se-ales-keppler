# Clasificación calibrada de señales de tránsito de Kepler

**Red multirrama con un codificador por tránsito con atención** · Proyecto del curso de Deep Learning · Especialización en Inteligencia Artificial, Universidad Nacional de Colombia · Juan Sebastian Higuita Riascos · Entrega: 6 de diciembre de 2026.

> Título provisional. Estado (20-nov-2026): ramas, fusión, calibración, incertidumbre, estudios complementarios, sistema conjunto y auditoría hechos. Falta el informe y la presentación.
>
> **¿Por dónde empezar?** `docs/GUIA_PASO_A_PASO.md`: cómo correrlo, cómo recorrerlo y cómo leer los resultados.

## Preguntas

1. ¿Qué arquitectura lee mejor la señal de tránsito: una CNN 1D sobre la curva plegada, una red recurrente o un codificador por tránsito con atención?
2. ¿Cuánto mejora la decisión al fusionar cuatro ramas (curva plegada, tránsitos uno por uno, imagen de píxeles y escalares)?
3. ¿Las probabilidades salen calibradas, y la incertidumbre avisa de los errores?

## Qué es de este curso y qué no

| Parte | Origen |
|---|---|
| Ramas 1 (vistas), 2 (tránsitos) y 4 (escalares); fusión; calibración e incertidumbre | **Este curso** |
| Rama 3 (imagen de diferencia, CNN 2D) | Código de la demo, anterior a los cursos (codificador v0). Aquí se usa congelada. El curso de Visión por Computador la desarrolla por separado; los proyectos se unen después de las entregas |
| Tabla de ejemplos, particiones y líneas base tabulares | Base común de los dos cursos |

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

## Reproducir

La secuencia completa de comandos está en [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md), sección 5. Los datos se descargan de fuentes públicas de NASA (Exoplanet Archive y MAST); no se versionan.

## Documentos

| Documento | Para qué |
|---|---|
| [`docs/planeacion/04_plan_ejecucion_deep_learning.md`](docs/planeacion/04_plan_ejecucion_deep_learning.md) | Plan, decisiones, protocolo experimental y calendario |
| [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md) | Estructura del repositorio y del modelo |
| [`docs/planeacion/BITACORA_DL.md`](docs/planeacion/BITACORA_DL.md) | Estado y registro de sesiones |
| [`.specify/memory/constitution.md`](.specify/memory/constitution.md) | Reglas que no se negocian |
| [`specs/`](specs/) | Especificación, plan técnico y tareas de cada paquete de trabajo |
| [`CLAUDE.md`](CLAUDE.md) | Guía para agentes de código |

## Reglas del proyecto, en breve

Sin columnas con fuga (`koi_score`, `koi_fpflag_*`, `koi_pdisposition`). Particiones por estrella. La prueba no decide nada. Toda cifra se regenera con un script. Los resultados negativos se reportan igual que los positivos.
