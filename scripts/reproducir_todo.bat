@echo off
rem Vuelve a producir todos los resultados a partir de los datos ya descargados. Tarda del orden de una hora en CPU.
setlocal
cd /d "%~dp0.."
call .venv\Scripts\activate.bat || (echo Falta el entorno .venv. Ejecuta primero scripts\descargar_recursos.bat & pause & exit /b 1)
set KERAS_BACKEND=jax
pip install -q -r requirements.txt
set D=--dataset data\dataset_temporal.npz --ejemplos compartido\ejemplos.csv
python -m pytest -q
python -I src\temporal\exp_ramas.py %D% --salida resultados\dl --experimentos m1_astronet m2_atencion m4_escalares --cordura || (echo Fallo una prueba de cordura. & pause & exit /b 1)
python -I src\temporal\exp_ramas.py %D% --salida resultados\dl --modelos modelos --experimentos m4_escalares m1_astronet --exportar
python -I src\temporal\exp_ramas.py %D% --salida resultados\dl --experimentos m2_promedio m2_gru m2_atencion
python -I src\temporal\analisis_ramas.py %D% --resultados resultados\dl --modelos modelos --pasos comparar seleccionar permutar exportar pesos
python src\demo\demo_dataset.py data\tpf data\seleccion_demo.csv data\demo_crudo.npz
python -I src\fusion\dataset_demo.py --crudo data\demo_crudo.npz --ejemplos compartido\ejemplos.csv --salida data\dataset_demo.npz
python -I src\fusion\codificador_imagen_v0.py --dataset data\dataset_demo.npz --ejemplos compartido\ejemplos.csv --salida resultados\dl --modelos modelos --experimentos imagen_v0 m0_demo_curva m0_demo_completo
python -I src\fusion\fusionar.py --modelos modelos %D% --resultados resultados\dl
python -I src\fusion\calibracion.py --predicciones resultados\dl\fusion\e6a_cuatro_ramas --ejemplos compartido\ejemplos.csv --koi data\koi_dr25.csv --salida resultados\dl\calibracion
python -I src\fusion\incertidumbre.py --modelos modelos %D% --fusion resultados\dl\fusion\e6a_cuatro_ramas --salida resultados\dl\incertidumbre
python -I src\fusion\consolidar_resultados.py --resultados resultados\dl --ejemplos compartido\ejemplos.csv
python -I src\comun\verificar_contrato.py --raiz .
python scripts\construir_notebook.py
rem Copia que se lee sin ejecutar: figuras fijas. Al abrirla y ejecutarla aparecen los controles.
set INFORME_ESTATICO=1
jupyter nbconvert --to notebook --execute --inplace notebooks\informe_dl.ipynb
echo Listo.
pause
