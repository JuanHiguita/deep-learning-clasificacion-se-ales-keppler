@echo off
rem Repite los estudios complementarios (specs 009, 010, 011 y 013). Son reanudables y tardan varias horas en CPU.
rem Lo ya entrenado no se repite: para volver a entrenar algo, borra su carpeta en resultados\dl.
setlocal
cd /d "%~dp0.."
call .venv\Scripts\activate.bat || (echo Falta el entorno .venv. Ejecuta primero scripts\descargar_recursos.bat & pause & exit /b 1)
set KERAS_BACKEND=jax
set D=--dataset data\dataset_temporal.npz --ejemplos compartido\ejemplos.csv
echo === Busqueda de hiperparametros (spec 009) ===
python -I src\temporal\busqueda_ramas.py %D% --salida resultados\dl\busqueda --referencia resultados\dl --arquitecturas m4_escalares m2_promedio m2_gru m2_atencion m1_astronet --guardar-mejores modelos\mejores
python -I src\fusion\busqueda_cabeza.py --modelos modelos %D% --resultados resultados\dl --guardar-mejores modelos\mejores
python -I src\comun\busqueda.py --busqueda resultados\dl\busqueda --ejemplos compartido\ejemplos.csv --tablas resultados\dl\tablas
echo === Estudio de diseno, E2b, E4 y E5 (spec 010) ===
python -I src\temporal\estudio_diseno.py %D% --salida resultados\dl\diseno --referencia resultados\dl --arquitecturas m4_escalares m2_atencion m2_promedio m2_gru m1_astronet
python -I src\temporal\estudio_diseno.py %D% --salida resultados\dl\diseno --referencia resultados\dl --arquitecturas m1_astronet --confirmar act_tanh perdida_suavizado pre_zscore
python -I src\temporal\e4_menos_transitos.py %D% --diseno resultados\dl\diseno --salida resultados\dl\e4
python -I src\comun\estudio.py --diseno resultados\dl\diseno --e4 resultados\dl\e4 --tablas resultados\dl\tablas
echo === Sistema conjunto (spec 011) ===
python -I src\fusion\sistema.py --modelos modelos %D% --resultados resultados\dl
echo === Copia experimental con la rama 1 mejorada (spec 013) ===
python -I src\fusion\experimental_v2.py --modelos modelos %D% --resultados resultados\dl
echo === Notebooks ===
python scripts\construir_notebooks_exploratorios.py
python scripts\construir_notebook_diseno.py
set INFORME_ESTATICO=1
for %%N in (01_exploracion_datos 02_experimentacion_hiperparametros 03_estudio_de_diseno) do jupyter nbconvert --to notebook --execute --inplace notebooks\%%N.ipynb
echo Listo.
pause
