# HED
Edge Detection 

Projektübersicht

Dieses Projekt stellt eine End-to-End-Pipeline zur automatisierten Kantenerkennung und Vektorisierung von Bilddateien bereit. Mittels einer grafischen Oberfläche (Streamlit) können Anwender auf einfache Weise einen Ordner mit JPG- und PNG-Bildern auswählen. Die Pipeline führt für jedes Bild folgende Schritte aus:

1. Vorverarbeitung (Graustufen, Kontrastanpassung, Rauschreduktion)


2. Kantendetektion mit zwei Methoden:

HED (Holistically-Nested Edge Detection) via OpenCV DNN

PiDiNet (Pytorch-Implementation)



3. Nachbearbeitung der Kantenergebnisse (Glättung, Histogramm-Equalisierung)


4. Kombination beider Ergebnisse zu einem finalen Kantenset


5. Binarisierung für Vektorisierung


6. Vektorisierung der Kanten-Bitmap zu SVG (mit Potrace)


7. PDF-Export der SVG-Grafiken (mit ReportLab)


8. Vorschau-Generierung und lokaler Webserver für schnelles Durchblättern der Ergebnisse



Zusätzlich enthält das Projekt ein Windows-Batch-Skript, das eine virtuelle Umgebung aufsetzt, sämtliche Abhängigkeiten installiert und die Pipeline startet.


---

Features

Automatischer Modell-Download
Lädt benötigte Dateien (HED-Prototxt, HED-Gewichte, PiDiNet-Python-Modul und -Gewichte) beim ersten Lauf automatisch herunter.

Dual-Edge-Detection
Kombiniert die Vorteile von HED (präzise, global betrachtet) und PiDiNet (feine, lokale Strukturen).

Flexible Vorverarbeitung
CLAHE-Kontrastoptimierung und Bilateralfilter reduzieren Rauschen bei gleichzeitigem Kontrasterhalt.

Adaptive Binarisierung
Gauss’sche adaptive Thresholding-Methode für robuste Schwellenwertsetzung auf heterogenen Bildern.

Vollständige Vektordaten
Ausgabe sowohl als SVG (Skalierbarkeit) als auch als PDF (Druck- und Layout-freundlich).

Streamlit-Web-App
Einfache Bedienoberfläche mit Fortschrittsanzeige, Statusmeldungen und Button-Steuerung.

Integrierter Vorschau-Server
Zeigt alle verarbeiteten Vorschaubilder im Browser unter http://localhost:8000 an.

Automatisierung via Batch-Skript
Vollständige Installation und Start der Pipeline per Doppelklick unter Windows.



---

Verzeichnisstruktur

├── main_pipeline.py          # Hauptskript mit Streamlit-App
├── requirements.txt          # Alle Python-Abhängigkeiten
├── run_pipeline.bat          # Windows-Batch-Skript zur Einrichtung & Ausführung
├── models/
│   ├── hed/
│   │   ├── deploy.prototxt   # HED-Modellstruktur
│   │   └── hed_pretrained_bsds.caffemodel  # HED-Gewichte
│   └── sae/
│       ├── pidinet.py        # PiDiNet-Implementierung
│       └── pidinet_v2_converted.pth  # PiDiNet-Gewichte
└── [Arbeitsverzeichnisse bei Ausführung]
    ├── 01_preprocessed/
    ├── 02_HED_raw/
    ├── 03_SAE_raw/
    ├── 04_HED_processed/
    ├── 05_SAE_processed/
    ├── 06_combined/
    ├── 07_binarized/
    ├── 08_vectorized/
    │   ├── *.svg
    │   └── *.pdf
    └── 09_previews/


---

Installation

1. Repository klonen

git clone <repository-url>
cd <repository-ordner>


2. Virtuelle Umgebung einrichten (Windows)
Doppelklick auf run_pipeline.bat oder manuell:

python -m venv .venv
.venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt


3. Abhängigkeiten
Enthalten in requirements.txt:

streamlit
opencv-python
numpy
torch
tk
potrace
svglib
reportlab
Pillow
requests
protobuf




---

Nutzung

cd zum/projekt/verzeichnis
streamlit run main_pipeline.py

1. Im erscheinenden Browser-Fenster den Pfad zu einem Ordner mit JPG/PNG-Bildern eingeben.


2. Auf „🚀 Starte Verarbeitung“ klicken.


3. Fortschrittsbalken, Statusanzeige und Daueranzeige beobachten.


4. Nach Abschluss: Vorschau im Browser unter http://localhost:8000 öffnen.




---

Detaillierter Ablauf der Pipeline

1. Modell-Download & Setup

Beim ersten Start werden die Modelle nur heruntergeladen, falls sie lokal nicht existieren.

Nutzt Python requests.get zum Herunterladen an definierte Pfade unter models/.


2. Vorverarbeitung

Graustufen: cv2.cvtColor

CLAHE: Adaptive Kontrastverstärkung (cv2.createCLAHE(clipLimit=2.0))

Bilateralfilter: Rauschreduzierung bei Kantenerhalt (cv2.bilateralFilter(…, 9, 75, 75))


3. HED-Kantendetektion

Erzeugung eines Blobs: cv2.dnn.blobFromImage

Laden des Caffe-Modells: cv2.dnn.readNetFromCaffe

Ausgabe-Map via net.forward(), Skalierung und Umwandlung in 8-Bit-Image.


4. PiDiNet-Kantendetektion

Dynamischer Import des pidinet.py-Moduls per importlib.

Inferenz auf CPU/GPU: Tensor-Erstellung (T.ToTensor()), Modell im No-Grad-Modus.

Extraktion der letzten Feature-Stufe, Skalierung auf 0–255.


5. Nachbearbeitung

HED: Gaussian Blur (cv2.GaussianBlur(…, (5,5), 0))

PiDiNet: Histogramm-Equalisierung (cv2.equalizeHist)


6. Kombination & Binarisierung

Pixelweises Maximum der beiden bearbeiteten Kantenergebnisse.

Adaptive Thresholding zu Bitmap (cv2.adaptiveThreshold(..., ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,11,2)).


7. Vektorisierung

Aufruf von Potrace via Kommandozeile:

potrace -s -o 08_vectorized/<filename>.svg 07_binarized/<filename>.bmp


8. PDF-Export

Einlesen des SVG mit svg2rlg (svglib) und Einbetten in PDF via ReportLab:

drawing = svg2rlg(svg_path)
c = canvas.Canvas(pdf_path)
renderPDF.draw(drawing, c, 0, 0)
c.save()


9. Vorschau & Webserver

Vorschau-Bild: nebeneinander-gerenderte Original- und Kantenergebnis-Snippets (je 300×300 px).

Automatisch gestarteter HTTP-Server in 09_previews auf Port 8000 mittels http.server und socketserver.



---

Batch-Skript (run_pipeline.bat)

@echo off
REM Automatisch virtuelle Umgebung erstellen
IF NOT EXIST .venv (
    python -m venv .venv
)
call .venv\Scripts\activate.bat

REM Abhängigkeiten installieren
pip install --upgrade pip
pip install -r requirements.txt

REM Pipeline starten
python main_pipeline.py

Zweck: Ein-Klick-Installation und -Ausführung unter Windows.

Funktion: .venv anlegen, aktivieren, Pakete installieren, Hauptskript starten.



---

Hinweise & Erweiterungsmöglichkeiten

GPU-Nutzung: Standardmäßig wird GPU automatisch verwendet, wenn CUDA verfügbar.

Parameter-Tuning: CLAHE-Cliplimit, Bilateralfilter-Stärke, Thresholding-Fenstergröße können direkt im Code angepasst werden.

Erweiterung:

Weitere Edge-Detection-Modelle integrieren (z. B. Canny, Sobel).

Streamlit-Controls (Slider) für dynamisches Parameter-Tuning.

Docker-Containerisierung für plattformunabhängiges Deployment.




---

Lizenz & Haftung

Dieses Projekt steht unter der MIT-Lizenz. Die verwendeten Modelle (HED, PiDiNet) unterliegen jeweils ihren eigenen Lizenzbedingungen (siehe Original-Repositories). Jegliche Haftung für fehlerhafte Kanten- oder Vektorergebnisse wird ausgeschlossen.


