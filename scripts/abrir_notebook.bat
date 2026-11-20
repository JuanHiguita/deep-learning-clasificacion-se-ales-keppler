@echo off
rem Abre el notebook de resultados. Instala lo que falte en el entorno virtual.
setlocal
cd /d "%~dp0.."
call .venv\Scripts\activate.bat || (echo Falta el entorno .venv. Ejecuta primero scripts\descargar_recursos.bat & pause & exit /b 1)
pip install -q jupyterlab ipywidgets matplotlib jinja2
set KERAS_BACKEND=jax
rem Abre la carpeta: 01 exploracion de datos, 02 experimentacion e informe_dl
jupyter lab notebooks
