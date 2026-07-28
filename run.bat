@echo off
REM Lance FaceGuard. Voir le README pour l'installation.

cd /d "%~dp0"
set "TF_CPP_MIN_LOG_LEVEL=3"
set "PYTHONUNBUFFERED=1"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app\server.py
) else (
  python app\server.py
)

if errorlevel 1 (
  echo.
  echo Le lancement a echoue. Verifie que les dependances sont installees :
  echo    pip install -r requirements.txt
  pause
)
