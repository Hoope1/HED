@echo off

REM === Automatisch VENV einrichten & starten ===
IF NOT EXIST .venv (
    echo [*] Erstelle virtuelle Umgebung...
    python -m venv .venv
)

echo [*] Aktiviere virtuelle Umgebung
call .venv\Scripts\activate.bat

REM === Pakete installieren ===
echo [*] Installiere Anforderungen...
pip install --upgrade pip
pip install -r requirements.txt

REM === Starte Haupt-Pipeline ===
python main_pipeline.py
