@echo off
rem Iteracion 2, paso 3: protocolo fijo con los datos ampliados, en GPU (semilla 0). Sirve de referencia
rem para la busqueda y de prueba de que todo el motor entrena con PyTorch. Reanudable.
setlocal
cd /d "%~dp0.."
call .venv_gpu\Scripts\activate.bat || (echo Falta .venv_gpu & pause & exit /b 1)
set KERAS_BACKEND=torch
echo [%date% %time%] INICIO > iteracion_2\evidencias\registros\3_estado.txt
for %%M in (m4_escalares m2_promedio m2_atencion m2_gru m1_astronet) do call :uno %%M
echo [%date% %time%] FIN >> iteracion_2\evidencias\registros\3_estado.txt
pause
exit /b 0
:uno
echo === Entrenando %1 (5 particiones, semilla 0) ===
powershell -NoProfile -Command "python -u -I src\temporal\exp_ramas.py --dataset iteracion_2\datos\dataset_temporal.npz --ejemplos iteracion_2\datos\ejemplos.csv --salida iteracion_2\evidencias\resultados --modelos iteracion_2\modelos_protocolo --protocolo iteracion_2\config\protocolo.json --semillas 0 --experimentos %1 2>&1 | Tee-Object -Append -FilePath iteracion_2\evidencias\registros\3_protocolo.log; exit $LASTEXITCODE"
echo [%date% %time%] %1 terminado, codigo %errorlevel% >> iteracion_2\evidencias\registros\3_estado.txt
exit /b 0
