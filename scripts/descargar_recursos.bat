@echo off
rem Prepara el entorno y descarga todos los datos del proyecto (tablas KOI, TPF de Q5 y curvas Q1-Q17).
rem Uso: doble clic, o desde una terminal:  scripts\descargar_recursos.bat
rem Es reanudable: si se corta, se vuelve a ejecutar y continua donde iba.
setlocal
cd /d "%~dp0.."

if not exist .venv (
  py -3 -m venv .venv 2>nul || python -m venv .venv
)
call .venv\Scripts\activate.bat || (echo No se pudo crear el entorno virtual. Instala Python 3.10 o superior. & pause & exit /b 1)

echo === 1/7 Dependencias minimas ===
python -m pip install -q --upgrade pip
pip install -q -r requirements-datos.txt || (echo Fallo la instalacion de dependencias. & pause & exit /b 1)
pip install -q -e . || (echo Fallo pip install -e . & pause & exit /b 1)

echo === 2/7 Pruebas rapidas ===
python -m pytest -q

echo === 3/7 Tablas KOI (NASA Exoplanet Archive) ===
if not exist data\koi_dr25.csv python -I src\comun\descargar_koi.py --tabla q1_q17_dr25_koi --salida data\koi_dr25.csv
if not exist data\koi_cumulative.csv python -I src\comun\descargar_koi.py --tabla cumulative --salida data\koi_cumulative.csv
if not exist data\koi_dr25.csv (echo No se pudo descargar la tabla DR25. Revisa la conexion. & pause & exit /b 1)

echo === 4/7 Tabla maestra de ejemplos (solo la primera vez) ===
if not exist compartido\ejemplos.csv python -I src\comun\construir_ejemplos.py --seleccion data\seleccion_demo.csv --koi data\koi_dr25.csv --acumulada data\koi_cumulative.csv --salida compartido\ejemplos.csv
python -I src\comun\verificar_contrato.py --raiz .

echo === 5/7 Lineas base tabulares ===
if not exist resultados\dl\lineas_base_tabulares\metricas.json python -I src\comun\lineas_base_tabulares.py --ejemplos compartido\ejemplos.csv --koi data\koi_dr25.csv --acumulada data\koi_cumulative.csv --salida resultados\dl\lineas_base_tabulares

echo === 6/7 TPF de Q5 (unos 260 archivos) ===
python -I src\temporal\descargar_mast.py --ejemplos compartido\ejemplos.csv --destino data --que tpf

echo === 7/7 Curvas de luz Q1-Q17 (unos 4400 archivos; puede tardar una hora o mas) ===
python -I src\temporal\descargar_mast.py --ejemplos compartido\ejemplos.csv --destino data --que llc

echo === Dataset temporal (entradas de las ramas 1, 2 y 4) ===
python -I src\temporal\construir_dataset_temporal.py --llc data\llc --ejemplos compartido\ejemplos.csv --koi data\koi_dr25.csv --salida data\dataset_temporal.npz
python -I src\comun\verificar_contrato.py --raiz .

echo === Dependencias completas (Keras + JAX); si fallan, los datos ya quedaron descargados ===
pip install -q -r requirements.txt
set KERAS_BACKEND=jax
python -c "import keras; print('Keras', keras.__version__, 'backend', keras.backend.backend())"

echo.
echo Listo. Si alguna estrella fallo, vuelve a ejecutar este archivo.
pause
