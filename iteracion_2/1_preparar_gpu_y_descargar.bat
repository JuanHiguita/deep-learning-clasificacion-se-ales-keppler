@echo off
rem Iteracion 2, paso 1: prepara el entorno con GPU (PyTorch + CUDA) y descarga las curvas de luz.
rem Reanudable: si se corta, se vuelve a ejecutar y continua donde iba.
rem El avance queda en iteracion_2\evidencias\registros\ (1_estado.txt, 1_instalacion.log, 1_descarga.log).
setlocal
cd /d "%~dp0.."
if not exist iteracion_2\evidencias\registros mkdir iteracion_2\evidencias\registros
echo [%date% %time%] INICIO > iteracion_2\evidencias\registros\1_estado.txt
if not exist .venv_gpu ( py -3 -m venv .venv_gpu 2>nul || python -m venv .venv_gpu )
call .venv_gpu\Scripts\activate.bat || (echo [%date% %time%] ERROR: no se pudo crear .venv_gpu >> iteracion_2\evidencias\registros\1_estado.txt & pause & exit /b 1)
echo === 1/4 Instalando PyTorch con CUDA y dependencias (unos 2,5 GB la primera vez) ===
python -m pip install --upgrade pip > iteracion_2\evidencias\registros\1_instalacion.log 2>&1
python -c "import torch" 2>nul || pip install torch --index-url https://download.pytorch.org/whl/cu124 >> iteracion_2\evidencias\registros\1_instalacion.log 2>&1
python -c "import torch" 2>nul || pip install torch --index-url https://download.pytorch.org/whl/cu126 >> iteracion_2\evidencias\registros\1_instalacion.log 2>&1
pip install -r iteracion_2\requirements_gpu.txt >> iteracion_2\evidencias\registros\1_instalacion.log 2>&1
pip install -e . >> iteracion_2\evidencias\registros\1_instalacion.log 2>&1
echo [%date% %time%] DEPENDENCIAS INSTALADAS >> iteracion_2\evidencias\registros\1_estado.txt
echo === 2/4 Comprobando la GPU ===
set KERAS_BACKEND=torch
python iteracion_2\comprobar_gpu.py > iteracion_2\evidencias\registros\1_gpu.log 2>&1
echo [%date% %time%] GPU COMPROBADA, codigo %errorlevel% (0 = hay GPU) >> iteracion_2\evidencias\registros\1_estado.txt
type iteracion_2\evidencias\registros\1_gpu.log
echo === 3/4 Descargando curvas de luz de 4.007 estrellas (unos 26 GB; varias horas) ===
for /L %%I in (1,1,3) do (
  python -u -I src\temporal\descargar_mast.py --ejemplos iteracion_2\datos\ejemplos.csv --destino iteracion_2\datos --que llc --hilos 6 --pausa 0.05 --reutilizar data\llc --sin-huellas >> iteracion_2\evidencias\registros\1_descarga.log 2>&1
  echo [%date% %time%] DESCARGA, PASADA %%I TERMINADA >> iteracion_2\evidencias\registros\1_estado.txt
)
echo === 4/4 Listo ===
echo [%date% %time%] FIN >> iteracion_2\evidencias\registros\1_estado.txt
pause
