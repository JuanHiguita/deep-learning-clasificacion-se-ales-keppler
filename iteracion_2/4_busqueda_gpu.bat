@echo off
rem Iteracion 2, paso 4: busqueda ampliada de hiperparametros en GPU (12 configuraciones por red).
rem Reanudable: si se cierra, volver a abrirlo continua donde quedo. El progreso se ve aqui y
rem tambien queda en iteracion_2\evidencias\registros\4_busqueda.log
setlocal
cd /d "%~dp0.."
call .venv_gpu\Scripts\activate.bat || (echo Falta .venv_gpu & pause & exit /b 1)
set KERAS_BACKEND=torch
set MOSTRAR_PROGRESO=1
set ESTADO=iteracion_2\evidencias\registros\4_estado.txt
set REGISTRO=iteracion_2\evidencias\registros\4_busqueda.log
echo [%date% %time%] INICIO >> %ESTADO%
echo === Prueba de humo (un minuto) ===
powershell -NoProfile -Command "python -u -I iteracion_2\prueba_de_humo.py 2>&1 | Tee-Object -Append -FilePath %REGISTRO%; exit $LASTEXITCODE"
if errorlevel 1 (
  echo [%date% %time%] PRUEBA DE HUMO FALLIDA >> %ESTADO%
  echo La prueba de humo fallo: la busqueda no se lanza. Ver %REGISTRO%
  pause
  exit /b 1
)
echo [%date% %time%] prueba de humo superada >> %ESTADO%
for %%M in (m4_escalares m1_astronet m2_promedio m2_atencion m2_gru) do call :uno %%M
echo [%date% %time%] FIN >> %ESTADO%
pause
exit /b 0
:uno
echo === Buscando %1 ===
powershell -NoProfile -Command "python -u -I src\temporal\busqueda_it2.py --arquitecturas %1 2>&1 | Tee-Object -Append -FilePath %REGISTRO%; exit $LASTEXITCODE"
echo [%date% %time%] %1 terminado, codigo %errorlevel% >> %ESTADO%
exit /b 0
