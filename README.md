# TrafficVioLens

**Automated Photo Identification & Classification for Traffic Violations Using Computer Vision**

🔗 **Live Demo:** [https://trafficviolens-frontend.vercel.app/](https://trafficviolens-frontend.vercel.app/)

---

## Overview

TrafficVioLens is an end-to-end AI system that automatically processes traffic images, detects vehicles and road users, identifies traffic violations, classifies them, performs license plate recognition, and generates court-ready annotated evidence — all from a single photograph.

Built as a scalable, deployable web application (React + FastAPI), it covers all 8 task areas of the challenge with real, measurable performance.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **7 Violation Types** | Helmet, triple riding, seatbelt, wrong-side, stop-line, red-light, illegal parking |
| **Self-Calibrating Enhancer** | Feedback-loop preprocessing handles low-light, blur, rain — no per-camera tuning |
| **Evidence Defensibility Score** | Every violation scored 0-100, auto-routed to issue/review/discard |
| **Tamper-Evident Hash Chain** | SHA-256 append-only chain — court-admissible integrity guarantee |
| **Explainable AI** | Plain-English justification for every flagged violation |
| **License Plate OCR** | Dual-engine (EasyOCR + Tesseract) with registration metadata parsing |
| **Court-Ready PDF Export** | Annotated images + metadata + hash seal in downloadable PDF |
| **Batch Triage** | Process multiple images, risk-sorted, human-effort minimized |
| **Real Evaluation** | Confusion matrix, robustness benchmark, per-class metrics — not fabricated |

---

## Performance Metrics

| Metric | Score |
|--------|-------|
| Detection F1 | 93.6% |
| mAP@0.5 | 81.8% |
| Violation F1 | 89.0% |
| Plate Exact Match | 56.5% |
| False-Positive Rate | 0% (on clean images) |
| Latency (CPU) | 266 ms/image |
| Latency (GPU projected) | ~33 ms/image |
| Scalability | 432,000 images/hour (4-GPU cluster) |

---

## Architecture

```
Image → Enhance → Detect → Violations → Plate OCR → Evidence → EDS Triage → Action
         │           │           │            │            │           │
    Feedback     YOLOv8n    7 Geometric   Haar Cascade   SHA-256    Auto-Issue
    Loop         @1280px    Rules Engine  + EasyOCR +    Hash       / Review
    Enhancer                              Tesseract      Chain      / Discard
```

**Tech Stack:**
- Backend: Python 3.11, FastAPI, OpenCV, Ultralytics YOLOv8, EasyOCR, Tesseract
- Frontend: React 18, Vite
- Deployment: Docker, Modal (Backend), Vercel (Frontend)
- Database: SQLite with append-only hash chain

---

## Project Structure

```
trafficviolens-web/
├── Dockerfile              # Single-image Docker build (React + Python)
├── backend/
│   ├── main.py             # FastAPI server (all endpoints)
│   ├── evaluate.py         # Evaluation benchmark harness
│   ├── requirements.txt
│   ├── src/                # ML pipeline modules
│   │   ├── preprocessing/  # Feedback-loop enhancer
│   │   ├── detection/      # YOLOv8 vehicle/person detector
│   │   ├── violations/     # 7-rule violation engine
│   │   ├── plate_recognition/  # OCR pipeline
│   │   ├── evidence/       # Hash chain, PDF, explainer, defensibility
│   │   ├── analytics/      # Reporting + database
│   │   └── evaluation/     # Metrics computation
│   ├── eval/               # Evaluation harness + tools
│   │   ├── prelabel.py     # Pre-annotation (model proposes, human verifies)
│   │   ├── validate_labels.py  # Label schema validator
│   │   ├── robustness_benchmark.py
│   │   ├── results.json    # Computed benchmark results
│   │   └── README.md       # Evaluation workflow guide
│   ├── demo/               # 5 curated demo images
│   └── data/               # Model weights (yolov8n.pt, helmet_v2/)
└── frontend/
    ├── src/
    │   ├── pages/          # Analyze, Analytics, Evaluation, About
    │   ├── components/     # UI components
    │   └── api.js          # API client
    ├── package.json
    └── vite.config.js
```

---

## How It Works (8 Stages)

### 1. Image Preprocessing
Feedback-loop enhancer: measures brightness, contrast, blur, exposure → applies corrections (gamma, CLAHE, unsharp mask) → re-measures → iterates until convergence. Self-calibrating, no per-camera config needed. **+19.4% detection improvement on low-light images.**

### 2. Vehicle & Road User Detection
YOLOv8n at 1280px input. Detects cars, motorcycles, buses, trucks, bicycles, auto-rickshaws, persons, and traffic lights. Dual-pass helmet detection for small/distant riders. **F1: 93.6%, mAP: 81.8%.**

### 3. Traffic Violation Detection (All 7 Types)
Spatial-geometric rules engine — no task-specific training data required:
- **Helmet**: YOLO helmet model + head-shape geometry fallback
- **Triple riding**: Person-vehicle spatial association (>2 tight riders)
- **Seatbelt**: Torso crop → Hough line diagonal strap detection
- **Wrong-side**: Pack-isolation lateral outlier analysis
- **Stop-line**: Zebra-crossing auto-detection + line-crossing geometry
- **Red-light**: HSV signal color analysis + stop-line crossing
- **Illegal parking**: Zone containment + crossing obstruction

### 4. Violation Classification
Each violation assigned: type, confidence score, severity (high/medium/low), bounding box, description. Evidence Defensibility Score (0-100) routes to auto-issue / human-review / discard.

### 5. License Plate Recognition
Haar Cascade plate localizer → crop enhancement → dual OCR (EasyOCR + Tesseract, best-of-2) → registration parsing (state/RTO/number).

### 6. Evidence Generation
Annotated images, context crops, plate crops, metadata packages. Tamper-evident SHA-256 hash chain (content_hash + append-only chain). Court-ready PDF export with embedded images and integrity seal.

### 7. Analytics & Reporting
Violation statistics, searchable records by plate/type/date, batch triage with risk ordering, exportable reports. Live operational metrics + evaluation dashboard.

### 8. Performance Evaluation
Real metrics from labelled validation set: Precision, Recall, F1, mAP per class. Per-violation-type confusion matrix. Robustness benchmark (degraded conditions). False-positive stress test (0% FP). Computational efficiency and scalability measurements.

---

## Run Locally

### Option A: Docker (recommended)
```bash
docker build -t trafficviolens .
docker run -p 7860:7860 trafficviolens
# Open http://localhost:7860
```

### Option B: Dev mode
```bash
# Terminal 1 - Backend
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --port 8000

# Terminal 2 - Frontend
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/analyze` | Analyze single image (full pipeline) |
| POST | `/api/analyze_batch` | Bulk image triage |
| GET | `/api/demo` | List curated demo images |
| POST | `/api/demo/analyze/{file}` | Analyze a demo image |
| GET | `/api/analytics` | Summary stats + records |
| POST | `/api/search` | Search by plate number |
| GET | `/api/evaluation` | Benchmark + operational metrics |
| GET | `/api/verify` | Hash-chain integrity verification |
| GET | `/api/evidence/pdf/{id}` | Download court-ready PDF |
| GET | `/api/health` | Health check |

---

## Evaluation

Run the benchmark harness:
```bash
cd backend
# Pre-annotate images (model proposes, you verify)
python eval/prelabel.py
# Validate labels
python eval/validate_labels.py eval/ground_truth.json
# Compute metrics
python evaluate.py
```

See `backend/eval/README.md` for the full workflow.

---

## Scalability

| Metric | CPU | GPU (projected) |
|--------|-----|-----------------|
| Latency/image | 266 ms | ~33 ms |
| Throughput | 3.76 FPS | ~30 FPS |
| 4-GPU cluster | — | ~120 FPS = 432K img/hr |

A city with 2,000 cameras at 1 frame/min generates 120,000 images/hour — handled with 3.6× headroom.

---

## License

AGPL-3.0 (inherited from Ultralytics YOLOv8)
