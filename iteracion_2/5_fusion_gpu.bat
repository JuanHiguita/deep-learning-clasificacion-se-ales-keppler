@echo off
rem Iteracion 2, paso 5: elegir los mejores modelos, exportar sus vectores, fusionar (sin imagen)
rem y comparar con los modelos tabulares. Requiere que el paso 4 haya terminado. Reanudable.
setlocal
cd /d "%~dp0.."
call .venv_gpu\Scripts\activate.bat || (echo Falta .venv_gpu & pause & exit /b 1)
set KERAS_BACKEND=torch
set ESTADO=iteracion_2\evidencias\registros\5_estado.txt
set REGISTRO=iteracion_2\evidencias\registros\5_fusion.log
echo [%date% %time%] INICIO >> %ESTADO%
powershell -NoProfile -Command "python -u -I src\comun\sensibilidad_seleccion.py --busqueda iteracion_2\evidencias\resultados\busqueda --ejemplos iteracion_2\datos\ejemplos.csv --protocolo iteracion_2\config\protocolo.json 2>&1 | Tee-Object -Append -FilePath %REGISTRO%; exit $LASTEXITCODE"
echo [%date% %time%] sensibilidad, codigo %errorlevel% >> %ESTADO%
powershell -NoProfile -Command "python -u -I src\fusion\iteracion_2.py --koi data\koi_dr25.csv 2>&1 | Tee-Object -Append -FilePath %REGISTRO%; exit $LASTEXITCODE"
echo [%date% %time%] fusion principal, codigo %errorlevel% >> %ESTADO%
rem Exploratorio: los mismos pasos eligiendo los modelos por AUC de validacion (no es el resultado principal).
powershell -NoProfile -Command "python -u -I src\fusion\iteracion_2.py --koi data\koi_dr25.csv --criterio auc --modelos iteracion_2\modelos_exploratorios_auc --salida iteracion_2\evidencias\resultados\fusion_exploratoria_auc 2>&1 | Tee-Object -Append -FilePath %REGISTRO%; exit $LASTEXITCODE"
echo [%date% %time%] FIN, codigo %errorlevel% >> %ESTADO%
pause
