@echo off
rem Comprueba que el proyecto esta sano: pruebas, contrato, prueba de extremo a extremo y auditoria.
rem Necesita los datos y los modelos (scripts\descargar_recursos.bat). Tarda unos minutos.
setlocal
cd /d "%~dp0.."
call .venv\Scripts\activate.bat || (echo Falta el entorno .venv. Ejecuta primero scripts\descargar_recursos.bat & pause & exit /b 1)
set KERAS_BACKEND=jax
echo === 1/5 Pruebas rapidas ===
python -m pytest -q || (echo HAY PRUEBAS QUE FALLAN. & pause & exit /b 1)
echo === 2/5 Pruebas que entrenan un modelo pequeno ===
python -m pytest -q -m lenta || (echo HAY PRUEBAS LENTAS QUE FALLAN. & pause & exit /b 1)
echo === 3/5 Contrato de integracion ===
python -I src\comun\verificar_contrato.py --raiz . || (echo EL CONTRATO NO SE CUMPLE. & pause & exit /b 1)
echo === 4/5 Prueba de extremo a extremo: de los FITS a la probabilidad ===
python scripts\prueba_e2e.py --estrellas-por-particion 2 || (echo LA PRUEBA DE EXTREMO A EXTREMO NO COINCIDE. & pause & exit /b 1)
echo === 5/5 Auditoria ===
python -I src\fusion\auditoria.py --raiz . || (echo LA AUDITORIA ENCONTRO FALLOS. & pause & exit /b 1)
echo.
echo TODO CORRECTO.
pause
