@echo off
rem Iteracion 2, paso 6: evaluacion UNICA de las 76 senales prerregistradas.
rem Primero un ensayo general con senales normales (incluye la comprobacion). Solo si pasa, la real.
rem La hipotesis esta declarada en iteracion_2\evidencias\prerregistro_declaracion.md
setlocal
cd /d "%~dp0.."
call .venv_gpu\Scripts\activate.bat || (echo Falta .venv_gpu & pause & exit /b 1)
set KERAS_BACKEND=torch
set ESTADO=iteracion_2\evidencias\registros\6_estado.txt
set REGISTRO=iteracion_2\evidencias\registros\6_prerregistrados.log
echo [%date% %time%] INICIO >> %ESTADO%
echo === Ensayo general con senales normales ===
powershell -NoProfile -Command "python -u -I src\fusion\it2_prerregistrados.py --koi data\koi_dr25.csv --simulacro 30 2>&1 | Tee-Object -Append -FilePath %REGISTRO%; exit $LASTEXITCODE"
if errorlevel 1 (
  echo [%date% %time%] ENSAYO FALLIDO: no se evalua >> %ESTADO%
  echo El ensayo fallo: las prerregistradas NO se evaluaron. Ver %REGISTRO%
  pause
  exit /b 1
)
echo [%date% %time%] ensayo superado >> %ESTADO%
echo === Evaluacion unica ===
powershell -NoProfile -Command "python -u -I src\fusion\it2_prerregistrados.py --koi data\koi_dr25.csv 2>&1 | Tee-Object -Append -FilePath %REGISTRO%; exit $LASTEXITCODE"
echo [%date% %time%] FIN, codigo %errorlevel% >> %ESTADO%
pause
