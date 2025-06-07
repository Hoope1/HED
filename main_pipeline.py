import os
import cv2
import time
import torch
import requests
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
from reportlab.pdfgen import canvas
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
import http.server
import socketserver
import threading
from torchvision import transforms as T
import importlib.util
import sys

# === MODEL-QUELLEN ===
HED_PROTO_URL = "https://raw.githubusercontent.com/s9xie/hed/master/examples/hed/deploy.prototxt"
HED_MODEL_URL = "https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel"
PIDINET_MODEL_URL = "https://raw.githubusercontent.com/hellozhuo/pidinet/master/trained_models/table5_pidinet.pth"
PIDINET_FILES = {
    "pidinet.py": "https://raw.githubusercontent.com/hellozhuo/pidinet/master/models/pidinet.py",
    "ops.py": "https://raw.githubusercontent.com/hellozhuo/pidinet/master/models/ops.py",
    "config.py": "https://raw.githubusercontent.com/hellozhuo/pidinet/master/models/config.py",
    "ops_theta.py": "https://raw.githubusercontent.com/hellozhuo/pidinet/master/models/ops_theta.py",
}

# === LOKALE DATEIPFADE ===
Path("models/hed").mkdir(parents=True, exist_ok=True)
Path("models/sae").mkdir(parents=True, exist_ok=True)

HED_PROTO = "models/hed/deploy.prototxt"
HED_MODEL = "models/hed/hed_pretrained_bsds.caffemodel"
PIDINET_MODEL = "models/sae/pidinet_v2_converted.pth"
PIDINET_PY = "models/sae/pidinet.py"

# === Dateien herunterladen wenn nicht vorhanden ===
def download(url, path):
    if not Path(path).exists():
        print(f"⬇️ Lade {path}...")
        r = requests.get(url)
        with open(path, 'wb') as f:
            f.write(r.content)

download(HED_PROTO_URL, HED_PROTO)
download(HED_MODEL_URL, HED_MODEL)
download(PIDINET_MODEL_URL, PIDINET_MODEL)
for fname, url in PIDINET_FILES.items():
    download(url, f"models/sae/{fname}")

# === PiDiNet laden ===
sys.modules.setdefault("sae", importlib.util.module_from_spec(importlib.util.spec_from_loader("sae", loader=None)))
for file in PIDINET_FILES:
    if file != "pidinet.py":
        mod_name = f"sae.{file[:-3]}"
        spec = importlib.util.spec_from_file_location(mod_name, f"models/sae/{file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
spec = importlib.util.spec_from_file_location("sae.pidinet", PIDINET_PY)
pidinet = importlib.util.module_from_spec(spec)
sys.modules["sae.pidinet"] = pidinet
sys.modules["pidinet"] = pidinet
spec.loader.exec_module(pidinet)

class _Args:
    def __init__(self):
        self.config = "carv4"
        self.dil = True
        self.sa = True

pidinet_cfg = pidinet.pidinet_converted(_Args())
sae_model = pidinet.PiDiNet(pidinet_cfg)
sae_model.load_state_dict(torch.load(PIDINET_MODEL, map_location="cpu"))
sae_model.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# === Streamlit-GUI ===
st.title("✏️ Kantenerkennung & Vektorisierung")
input_dir = st.text_input("📁 Eingabeordner", "")

if input_dir and Path(input_dir).exists():
    if st.button("🚀 Starte Verarbeitung"):
        start_time = time.time()
        progress = st.progress(0)
        status = st.empty()

        DIRS = [
            "01_preprocessed", "02_HED_raw", "03_SAE_raw", "04_HED_processed",
            "05_SAE_processed", "06_combined", "07_binarized", "08_vectorized",
            "09_previews"
        ]
        for d in DIRS:
            Path(d).mkdir(parents=True, exist_ok=True)

        files = list(Path(input_dir).glob("*.jpg")) + list(Path(input_dir).glob("*.png"))
        total = len(files)

        for i, file in enumerate(files):
            status.text(f"Verarbeite: {file.name} ({i+1}/{total})")
            img = cv2.imread(str(file))

            # === Vorverarbeitung ===
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0)
            gray = clahe.apply(gray)
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            cv2.imwrite(f"01_preprocessed/{file.name}", gray)

            # === HED ===
            blob = cv2.dnn.blobFromImage(img, 1.0, (256, 256),
                                         (104.00698793, 116.66876762, 122.67891434),
                                         swapRB=False, crop=False)
            net = cv2.dnn.readNetFromCaffe(HED_PROTO, HED_MODEL)
            net.setInput(blob)
            hed = net.forward()
            hed = cv2.resize(hed[0, 0], (img.shape[1], img.shape[0]))
            hed = (255 * hed).astype(np.uint8)
            cv2.imwrite(f"02_HED_raw/{file.name}", hed)

            # === PiDiNet Inferenz ===
            transform = T.Compose([T.ToTensor()])
            tensor = transform(gray).unsqueeze(0).to(next(sae_model.parameters()).device)
            with torch.no_grad():
                out = sae_model(tensor)[-1]  # letzte Stufe
            sae = out.squeeze().cpu().numpy()
            sae = (255 * sae).astype(np.uint8)
            cv2.imwrite(f"03_SAE_raw/{file.name}", sae)

            # === Nachbearbeitung ===
            hed_proc = cv2.GaussianBlur(hed, (5, 5), 0)
            sae_proc = cv2.equalizeHist(sae)
            cv2.imwrite(f"04_HED_processed/{file.name}", hed_proc)
            cv2.imwrite(f"05_SAE_processed/{file.name}", sae_proc)

            # === Kombination ===
            combined = np.maximum(hed_proc, sae_proc)
            cv2.imwrite(f"06_combined/{file.name}", combined)

            # === Binarisierung ===
            bin_img = cv2.adaptiveThreshold(combined, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
            bin_path = f"07_binarized/{file.stem}.bmp"
            cv2.imwrite(bin_path, bin_img)

            # === Vektorisierung ===
            os.system(f"potrace -s -o 08_vectorized/{file.stem}.svg {bin_path}")

            # === PDF-Export ===
            drawing = svg2rlg(f"08_vectorized/{file.stem}.svg")
            pdf_file = f"08_vectorized/{file.stem}.pdf"
            c = canvas.Canvas(pdf_file)
            renderPDF.draw(drawing, c, 0, 0)
            c.save()

            # === Vorschau ===
            combined_color = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
            preview = np.hstack((cv2.resize(img, (300, 300)),
                                 cv2.resize(combined_color, (300, 300))))
            cv2.imwrite(f"09_previews/{file.name}", preview)

            progress.progress(int((i+1) / total * 100))

        st.success("✅ Alle Bilder verarbeitet")
        st.text(f"⏱️ Dauer: {time.time() - start_time:.2f} Sekunden")

        # === Webserver ===
        def run_server():
            os.chdir("09_previews")
            handler = http.server.SimpleHTTPRequestHandler
            with socketserver.TCPServer(("", 8000), handler) as httpd:
                httpd.serve_forever()

        threading.Thread(target=run_server, daemon=True).start()
        st.markdown("🌐 Vorschau: [localhost:8000](http://localhost:8000)")
else:
    st.warning("Bitte gültigen Ordnerpfad angeben")
