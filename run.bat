@echo off
echo.
echo  Kaspi.kz (KSPI) Stock Dashboard
echo  ================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo  [FEHLER] Python nicht gefunden. Bitte Python 3.10+ installieren.
    pause
    exit /b 1
)

echo  Installiere Abhaengigkeiten...
pip install -q -r requirements.txt

echo  Generiere Icons (einmalig)...
if not exist "static\icon-192.png" python generate_icons.py

echo.
echo  Starte Server...
start "" http://localhost:5000
python app.py

pause
