import os
import sys
from pathlib import Path
import modal

# Define local paths
BACKEND_DIR = Path(__file__).resolve().parent

# Define a function to download easyocr models during the image build stage
def download_easyocr_models():
    import easyocr
    print("Downloading EasyOCR model files...")
    # This downloads detection and recognition models to /root/.EasyOCR/model/
    easyocr.Reader(['en'], gpu=False)

# Define the container image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "tesseract-ocr",
        "tesseract-ocr-eng",
    )
    .pip_install(
        "fastapi==0.115.0",
        "uvicorn[standard]==0.30.6",
        "python-multipart==0.0.9",
        "torch==2.2.2",
        "torchvision==0.17.2",
        "ultralytics==8.2.0",
        "opencv-python-headless==4.9.0.80",
        "numpy==1.26.4",
        "easyocr==1.7.1",
        "pytesseract==0.3.13",
        "Pillow==10.3.0",
        "scikit-image==0.23.2",
        "pandas==2.2.2",
        "fpdf2==2.8.1",
        "psycopg2-binary==2.9.9",
    )
    .run_function(download_easyocr_models)
    .add_local_dir(BACKEND_DIR, remote_path="/root/backend")
)

# Define the Modal App
app = modal.App("trafficviolens")

@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("trafficviolens-secrets")],  # Secret containing DATABASE_URL
    timeout=600,
)
@modal.asgi_app()
def fastapi_app():
    # Insert backend root to sys.path so backend imports work correctly
    sys.path.insert(0, "/root")
    
    # Pre-configure YOLO weight cache directory
    os.environ["YOLO_CONFIG_DIR"] = "/root/backend/data"
    
    # Import the FastAPI instance
    from backend.main import app as fastapi_instance
    return fastapi_instance
