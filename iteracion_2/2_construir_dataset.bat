@echo off
rem Iteracion 2, paso 2: de las curvas de luz descargadas a las entradas de las ramas (extractor).
rem El progreso se ve en esta ventana y queda en iteracion_2\evidencias\registros\2_dataset.log.
setlocal
cd /d "%~dp0.."
call .venv_gpu\Scripts\activate.bat || (echo Falta .venv_gpu. Ejecuta primero iteracion_2\1_preparar_gpu_y_descargar.bat & pause & exit /b 1)
echo [%date% %time%] INICIO > iteracion_2\evidencias\registros\2_estado.txt
echo === Construyendo el dataset de 9.094 ejemplos con 6 procesos ===
powershell -NoProfile -Command "python -u -I src\temporal\construir_dataset_temporal.py --llc iteracion_2\datos\llc --ejemplos iteracion_2\datos\ejemplos.csv --koi data\koi_dr25.csv --salida iteracion_2\datos\dataset_temporal.npz --procesos 6 2>&1 | Tee-Object -FilePath iteracion_2\evidencias\registros\2_dataset.log; exit $LASTEXITCODE"
echo [%date% %time%] FIN, codigo %errorlevel% >> iteracion_2\evidencias\registros\2_estado.txt
pause
