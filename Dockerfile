# ═══════════════════════════════════════════════════════════════
# Stage 1 — Build the React frontend
# ═══════════════════════════════════════════════════════════════
FROM node:20-slim AS frontend
ARG CACHEBUST=7
WORKDIR /fe
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build          # outputs /fe/dist

# ═══════════════════════════════════════════════════════════════
# Stage 2 — Python backend + bundled frontend
# ═══════════════════════════════════════════════════════════════
FROM python:3.11-slim

# System libs: OpenCV (headless) runtime + Tesseract OCR engine
# Debian Trixie renamed libgl1-mesa-glx -> libgl1; headless OpenCV needs no X11 libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Backend code + pipeline + model assets
COPY backend/ .

# Built React app → served as static files by FastAPI
COPY --from=frontend /fe/dist ./static

# Hugging Face Spaces / most PaaS expose port 7860; honour $PORT if set
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
