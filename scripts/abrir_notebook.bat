@echo off
rem Abre el notebook de resultados. Instala lo que falte en el entorno virtual.
setlocal
cd /d "%~dp0.."
call .venv\Scripts\activate.bat || (echo Falta el entorno .venv. Ejecuta primero scripts\descargar_recursos.bat & pause & exit /b 1)
pip install -q jupyterlab ipywidgets matplotlib jinja2
jupyter lab notebooks\informe_dl.ipynb
