"""
TrafficVioLens — FastAPI backend.

Wraps the EXISTING TrafficVioLens Python pipeline (no modifications to the
original project) and exposes it as JSON/REST endpoints for the React frontend.

The pipeline logic mirrors `app/dashboard.py::process_image` exactly so behaviour
is identical to the Streamlit app.
"""

import base64
import logging
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Standalone: the pipeline code (src/) and assets (data/) live alongside this file ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.feedback_enhancer import FeedbackLoopEnhancer  # noqa: E402
from src.detection.detector import VehicleDetector  # noqa: E402
from src.violations.rules_engine import ViolationRulesEngine  # noqa: E402
from src.plate_recognition.flagged_plate_reader import FlaggedVehiclePlateReader  # noqa: E402
from src.evidence.generator import EvidenceGenerator  # noqa: E402
from src.analytics.reporting import ViolationDatabase, AnalyticsEngine  # noqa: E402
from src.evaluation.metrics import StageTimer  # noqa: E402
from src.plate_recognition.recognizer import PlateResult  # noqa: E402
from src.evidence.defensibility import compute_defensibility, triage_summary, EDS_METHODOLOGY  # noqa: E402
from src.evidence.explainer import explain_violation, calibration_report  # noqa: E402


VEHICLE_COLORS = {
    "car": (0, 255, 0), "bus": (0, 200, 255), "truck": (0, 150, 255),
    "motorcycle": (255, 200, 0), "bicycle": (255, 150, 0), "auto": (255, 0, 200),
}

app = FastAPI(title="TrafficVioLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("trafficviolens")

# ── Upload safety limits ──
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB hard cap per image
ALLOWED_IMAGE_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp", "image/tiff",
}


def _validate_upload(file: UploadFile, contents: bytes):
    """Reject non-image, empty, or oversized uploads before any processing."""
    ctype = (file.content_type or "").lower()
    if ctype and ctype not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {ctype}")
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(contents)} bytes); max {MAX_UPLOAD_BYTES} bytes",
        )


# ── Lazy-loaded singletons (mirror @st.cache_resource) ──
_pipeline = None
_database = None

# ── In-memory operational metrics (honest, label-free; from real runs) ──
_OPS = {
    "latencies_ms": [],          # per-image total latency samples
    "images": 0,                 # images analysed this session
    "violations": 0,             # violations detected
    "plate_attempts": 0,         # violations where a plate read was attempted
    "plates_read": 0,            # violations with a successful plate read
    "routing": {"auto-issue": 0, "human-review": 0, "discard": 0},
    "eds_scores": [],            # EDS values for distribution stats
    "det_confidences": [],       # all detection confidences (vehicles+persons)
    "viol_confidences": [],      # all violation confidences
    "det_counts": [],            # detections per image
    "viol_counts": [],           # violations per image
}


def _record_ops(total_ms, violations_out, detections=None):
    _OPS["latencies_ms"].append(float(total_ms))
    _OPS["images"] += 1
    if detections:
        n_dets = len(getattr(detections, 'vehicles', [])) + len(getattr(detections, 'persons', []))
        _OPS["det_counts"].append(n_dets)
        for d in getattr(detections, 'vehicles', []) + getattr(detections, 'persons', []):
            _OPS["det_confidences"].append(float(d.confidence))
    _OPS["viol_counts"].append(len(violations_out))
    for v in violations_out:
        _OPS["violations"] += 1
        _OPS["viol_confidences"].append(v.get("confidence", 0))
        _OPS["plate_attempts"] += 1
        if v.get("plate_text"):
            _OPS["plates_read"] += 1
        r = v.get("routing")
        if r in _OPS["routing"]:
            _OPS["routing"][r] += 1
        _OPS["eds_scores"].append(v.get("eds", 0))
    # Cap lists to prevent unbounded memory growth
    for key in ("latencies_ms", "eds_scores", "det_confidences", "viol_confidences", "det_counts", "viol_counts"):
        if len(_OPS[key]) > 500:
            _OPS[key] = _OPS[key][-500:]


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from ultralytics import YOLO  # noqa: F401
        base = PROJECT_ROOT

        yolo_path = base / "data" / "yolov8n.pt"
        yolo_path.parent.mkdir(parents=True, exist_ok=True)
        if not yolo_path.exists():
            import urllib.request
            urllib.request.urlretrieve(
                "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
                str(yolo_path),
            )

        helmet_path = base / "data" / "models" / "helmet_v2" / "best.pt"
        helmet_path.parent.mkdir(parents=True, exist_ok=True)
        helmet_exists = helmet_path.exists()

        enhancer = FeedbackLoopEnhancer()
        detector = VehicleDetector(
            model_path=str(yolo_path),
            helmet_model_path=str(helmet_path) if helmet_exists else None,
            confidence_threshold=0.3,
        )
        rules_engine = ViolationRulesEngine()
        plate_reader = FlaggedVehiclePlateReader()
        evidence_gen = EvidenceGenerator()
        _pipeline = (enhancer, detector, rules_engine, plate_reader, evidence_gen)
    return _pipeline


def get_database():
    global _database
    if _database is None:
        # Use the same DB file the Streamlit app uses
        _database = ViolationDatabase(db_path=str(PROJECT_ROOT / "data" / "violations_db.json"))
    return _database


def img_to_b64(image: np.ndarray) -> str:
    """Encode a BGR numpy image as a base64 JPEG data URI."""
    if image is None or image.size == 0:
        return ""
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def process_image(image: np.ndarray) -> dict:
    """Replicates dashboard.process_image exactly, returning JSON-serialisable data."""
    enhancer, detector, rules_engine, plate_reader, evidence_gen = get_pipeline()
    timer = StageTimer()

    timer.start("enhancement")
    enhanced, trace = enhancer.enhance(image)
    timer.stop()

    timer.start("detection")
    detections = detector.detect(enhanced)
    timer.stop()

    timer.start("violation_analysis")
    violations = rules_engine.detect_all_violations(detections, image_shape=enhanced.shape, image=enhanced)
    timer.stop()

    timer.start("plate_recognition")
    plate_recognitions = plate_reader.read_for_violations(enhanced, violations)
    timer.stop()

    # ── Annotate (identical drawing logic to the Streamlit app) ──
    annotated = enhanced.copy()
    for d in detections.vehicles:
        x1, y1, x2, y2 = [int(v) for v in d.bbox]
        color = VEHICLE_COLORS.get(d.class_name, (200, 200, 200))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, d.class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    for d in detections.persons:
        x1, y1, x2, y2 = [int(v) for v in d.bbox]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(annotated, "person", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
    for d in detections.helmets:
        x1, y1, x2, y2 = [int(v) for v in d.bbox]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"HELMET {d.confidence:.0%}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
    for d in detections.no_helmets:
        x1, y1, x2, y2 = [int(v) for v in d.bbox]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(annotated, f"NO HELMET {d.confidence:.0%}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    for v in violations:
        x1, y1, x2, y2 = [int(c) for c in v.bbox]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(annotated, v.violation_type.replace("_", " ").upper(), (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    timer.start("evidence_generation")
    plate_results = [
        PlateResult(bbox=violations[pr.violation_index].bbox, text=pr.plate_text, confidence=pr.confidence)
        for pr in plate_recognitions if pr.plate_text
    ]
    packages = evidence_gen.generate(annotated, violations, plate_results)
    timer.stop()

    # Persist to DB (same as Streamlit) — deferred until context crops are built
    # so we can embed the visual evidence in the stored record.

    # ── Build per-violation context crops (matches dashboard expander) ──
    h_img, w_img = enhanced.shape[:2]
    violations_out = []
    eds_results = []
    for i, v in enumerate(violations):
        pr = next((p for p in plate_recognitions if p.violation_index == i), None)
        bx1, by1, bx2, by2 = [int(c) for c in v.bbox]
        pad = int(max(bx2 - bx1, by2 - by1) * 0.3)
        cx1, cy1 = max(0, bx1 - pad), max(0, by1 - pad)
        cx2, cy2 = min(w_img, bx2 + pad), min(h_img, by2 + pad)
        context_crop = enhanced[cy1:cy2, cx1:cx2].copy()
        if context_crop.size:
            cv2.rectangle(context_crop, (bx1 - cx1, by1 - cy1), (bx2 - cx1, by2 - cy1), (0, 0, 255), 2)

        has_plate = bool(pr and pr.plate_text)
        plate_conf = float(pr.confidence) if pr else 0.0

        # ── Evidence Defensibility Score + routing + risk ──
        eds = compute_defensibility(
            violation_confidence=float(v.confidence),
            severity=v.severity,
            violation_type=v.violation_type,
            bbox=v.bbox,
            image_w=w_img,
            image_h=h_img,
            quality_metrics=trace.final or {},
            plate_confidence=plate_conf,
            has_plate=has_plate,
        )
        eds_results.append(eds)

        # ── Plain-English explanation ──
        explanation = explain_violation(v, pr.plate_text if has_plate else "", plate_conf)

        violations_out.append({
            "index": i,
            "violation_id": packages[i].violation_id if i < len(packages) else "",
            "violation_type": v.violation_type,
            "title": v.violation_type.replace("_", " ").title(),
            "severity": v.severity,
            "confidence": float(v.confidence),
            "description": v.description,
            "bbox": [int(c) for c in v.bbox],
            "context_crop": img_to_b64(context_crop),
            "plate_text": pr.plate_text if pr else "",
            "plate_confidence": plate_conf,
            "plate_crop": img_to_b64(pr.plate_crop) if (pr and pr.plate_crop is not None and pr.plate_crop.size) else "",
            "registration": pr.registration if (pr and pr.plate_text) else {},
            # ── innovation layer ──
            "eds": eds.score,
            "routing": eds.routing,
            "eds_factors": eds.factors,
            "eds_rationale": eds.rationale,
            "risk": eds.risk,
            "explanation": explanation,
        })

    # Sort violations by risk (highest-danger first) for triage ordering
    violations_out.sort(key=lambda x: x["risk"], reverse=True)

    # ── Persist to DB with evidence crops embedded ──
    if packages:
        records_with_crops = []
        for idx, pkg in enumerate(packages):
            rec = pkg.to_dict()
            # Embed the context crop so the PDF endpoint can include it
            if idx < len(violations_out):
                rec["evidence_image"] = violations_out[idx].get("context_crop", "")
                rec["plate_crop_image"] = violations_out[idx].get("plate_crop", "")
            records_with_crops.append(rec)
        db = get_database()
        db._insert_records(records_with_crops)
        db._load()
        # Cap at 200 records to prevent disk overflow on free-tier hosting
        if len(db.records) > 200:
            db._prune_oldest(len(db.records) - 200)

    # ── Detection metrics ──
    vtypes = Counter(d.class_name for d in detections.vehicles)
    total_detections = len(detections.vehicles) + len(detections.persons)
    high_conf = (sum(1 for d in detections.vehicles if d.confidence >= 0.7)
                 + sum(1 for p in detections.persons if p.confidence >= 0.7))
    all_confs = [d.confidence for d in detections.vehicles] + [p.confidence for p in detections.persons]
    avg_det_conf = float(np.mean(all_confs)) if all_confs else 0.0
    v_conf = float(np.mean([v.confidence for v in violations])) if violations else 0.0
    plates_found = sum(1 for p in plate_recognitions if p.plate_text)

    # Per-class breakdown
    class_confs = {}
    for d in detections.vehicles:
        class_confs.setdefault(d.class_name, []).append(d.confidence)
    per_class = []
    for cls, count in vtypes.items():
        confs = class_confs[cls]
        per_class.append({
            "cls": cls.title(),
            "count": count,
            "avg": f"{np.mean(confs):.1%}",
            "min": f"{np.min(confs):.1%}",
            "max": f"{np.max(confs):.1%}",
        })

    timing = timer.summary()
    pixels = int(image.shape[0] * image.shape[1])

    _record_ops(timing["total_ms"], violations_out, detections)

    return {
        "original": img_to_b64(image),
        "annotated": img_to_b64(annotated),
        "enhancement": {
            "iterations": trace.iterations,
            "converged": bool(trace.converged),
            "resolution": trace.final.get("resolution", ""),
            "initial": trace.initial,
            "final": trace.final,
            "history": [
                {"action": action, "metrics": metrics}
                for (action, metrics) in trace.history
            ],
        },
        "detection": {
            "vehicles": len(detections.vehicles),
            "persons": len(detections.persons),
            "types": dict(vtypes),
            "per_class": per_class,
        },
        "violations": violations_out,
        "triage": triage_summary(eds_results),
        "eds_methodology": EDS_METHODOLOGY,
        "calibration": calibration_report(trace),
        "performance": {
            "stages_ms": timing["stages_ms"],
            "total_ms": timing["total_ms"],
            "fps": timing["fps"],
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "pixels": pixels,
            "px_per_ms": int(pixels / max(timing["total_ms"], 1)),
        },
        "metrics": {
            "avg_detection_conf": avg_det_conf,
            "high_conf": high_conf,
            "total_detections": total_detections,
            "avg_violation_conf": v_conf,
            "plates_found": plates_found,
            "violation_count": len(violations),
        },
        "saved_count": len(packages),
        "saved_ids": [p.violation_id for p in packages if p.violation_id],
    }


# ════════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════════
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        _validate_upload(file, contents)
        arr = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        return process_image(image)
    except HTTPException:
        raise
    except Exception:
        # Log full detail server-side; never leak tracebacks to the client.
        logger.exception("analyze failed")
        raise HTTPException(status_code=500, detail="Internal error during analysis")


@app.post("/api/analyze_batch")
async def analyze_batch(files: list[UploadFile] = File(...)):
    """Process many images at once and return a triage-ordered batch summary.

    Demonstrates real-world scalability: a day's worth of camera captures is
    ingested, every violation is scored by EDS, and the whole batch is rolled
    up into auto-issue / human-review / discard counts so an officer sees only
    what truly needs attention.
    """
    items = []
    agg = {"auto-issue": 0, "human-review": 0, "discard": 0}
    total_violations = 0

    for f in files:
        try:
            contents = await f.read()
            _validate_upload(f, contents)
            arr = np.frombuffer(contents, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is None:
                items.append({"filename": f.filename, "error": "decode failed", "violations": []})
                continue
            r = process_image(image)
            # Compact per-image record (no heavy base64 except the annotated thumb)
            vlist = [{
                "title": v["title"], "severity": v["severity"], "eds": v["eds"],
                "routing": v["routing"], "risk": v["risk"], "plate_text": v["plate_text"],
            } for v in r["violations"]]
            for v in r["violations"]:
                agg[v["routing"]] = agg.get(v["routing"], 0) + 1
            total_violations += len(vlist)
            items.append({
                "filename": f.filename,
                "thumbnail": r["annotated"],
                "violation_count": len(vlist),
                "violations": vlist,
                "top_risk": max([v["risk"] for v in vlist], default=0),
            })
        except Exception as e:
            items.append({"filename": f.filename, "error": str(e), "violations": []})

    # Order images by the most dangerous violation they contain
    items.sort(key=lambda it: it.get("top_risk", 0), reverse=True)

    processed = sum(1 for it in items if "error" not in it)
    actionable = agg["auto-issue"] + agg["human-review"]
    return {
        "images_processed": processed,
        "images_total": len(files),
        "total_violations": total_violations,
        "routing_counts": agg,
        "human_effort_pct": round(agg["human-review"] / max(actionable, 1) * 100, 1),
        "auto_issue_pct": round(agg["auto-issue"] / max(total_violations, 1) * 100, 1),
        "items": items,
    }


@app.get("/api/analytics")
def analytics():
    db = get_database()
    db._load()  # refresh from disk
    engine = AnalyticsEngine(db)
    stats = engine.summary_stats()

    df = db.get_dataframe()
    records = []
    columns = []
    if not df.empty:
        columns = [c for c in df.columns if c != "evidence_image"]
        records = df[columns].fillna("").to_dict("records")

    return {
        "summary": {
            "total_violations": stats.get("total_violations", 0),
            "avg_confidence": stats.get("avg_confidence", 0),
            "plates_identified": stats.get("plates_identified", 0),
            "violation_types": stats.get("violation_types", {}),
            "severity_distribution": stats.get("severity_distribution", {}),
        },
        "records": records,
        "columns": columns,
    }


class SearchRequest(BaseModel):
    plate: str


@app.post("/api/search")
def search(req: SearchRequest):
    db = get_database()
    db._load()
    results = db.search(plate=req.plate)
    cleaned = [{k: v for k, v in r.items() if k != "evidence_image"} for r in results]
    return {"results": cleaned}


@app.get("/api/verify")
def verify(violation_id: str | None = None):
    """Tamper-evidence verification.

    Verifies the append-only hash chain across all stored violation records, and
    (optionally) reports the integrity seal for a single record. The content hash
    binds the original frame + annotated evidence; since the raw frame is not
    persisted in the DB, per-record verification here confirms chain-link
    integrity (that the record was not altered or reordered after sealing).
    """
    from src.evidence import integrity  # noqa: E402

    db = get_database()
    db._load()
    chain = db.verify_chain()
    out = {"chain": chain, "total_records": len(db.records)}

    if violation_id:
        rec = next((r for r in db.records if r.get("violation_id") == violation_id), None)
        if rec is None:
            raise HTTPException(status_code=404, detail="violation_id not found")
        seal = rec.get("seal", {})
        # metadata=None -> only the chain-link (record_hash) consistency is checked,
        # since the source image needed to re-derive the content hash is not stored.
        single = integrity.verify_record(seal, None)
        out["record"] = {
            "violation_id": violation_id,
            "seal": seal,
            "verification": single,
            "note": "Content hash binds the original frame; full content "
                    "verification requires the archived source image.",
        }
    return out


def _load_benchmark():
    """Load pre-computed benchmark results from the labelled evaluation harness."""
    import json as _json
    results_path = PROJECT_ROOT / "eval" / "results.json"
    robustness_path = PROJECT_ROOT / "eval" / "robustness_results.json"
    out = None
    if results_path.exists():
        out = _json.loads(results_path.read_text())
    if robustness_path.exists():
        rob = _json.loads(robustness_path.read_text())
        if out:
            out["robustness"] = rob
        else:
            out = {"robustness": rob}
    return out


@app.get("/api/evaluation")
def evaluation():
    """Honest, label-free operational metrics from real runs this session,
    plus the methodology for the labelled accuracy benchmark (evaluate.py).
    No fabricated accuracy numbers — those come from the labelled harness.
    """
    lat = sorted(_OPS["latencies_ms"])
    n = len(lat)

    def pct(p):
        if not lat:
            return 0.0
        k = max(0, min(n - 1, int(round((p / 100) * (n - 1)))))
        return round(lat[k], 1)

    avg_lat = round(sum(lat) / n, 1) if n else 0.0
    plate_rate = round(_OPS["plates_read"] / max(_OPS["plate_attempts"], 1) * 100, 1)
    eds = _OPS["eds_scores"]
    avg_eds = round(sum(eds) / len(eds), 1) if eds else 0.0
    actionable = _OPS["routing"]["auto-issue"] + _OPS["routing"]["human-review"]

    return {
        "operational": {
            "images_analysed": _OPS["images"],
            "violations_detected": _OPS["violations"],
            "avg_latency_ms": avg_lat,
            "p50_latency_ms": pct(50),
            "p95_latency_ms": pct(95),
            "throughput_fps": round(1000 / max(avg_lat, 1), 2) if n else 0.0,
            "plate_read_rate_pct": plate_rate,
            "avg_eds": avg_eds,
            "routing": _OPS["routing"],
            "human_effort_pct": round(_OPS["routing"]["human-review"] / max(actionable, 1) * 100, 1),
        },
        "confidence_stats": {
            "detection_mean": round(sum(_OPS["det_confidences"]) / max(len(_OPS["det_confidences"]), 1), 3),
            "detection_count": len(_OPS["det_confidences"]),
            "detection_high_pct": round(sum(1 for c in _OPS["det_confidences"] if c >= 0.7) / max(len(_OPS["det_confidences"]), 1) * 100, 1),
            "violation_mean": round(sum(_OPS["viol_confidences"]) / max(len(_OPS["viol_confidences"]), 1), 3),
            "violation_count": len(_OPS["viol_confidences"]),
            "avg_detections_per_image": round(sum(_OPS["det_counts"]) / max(len(_OPS["det_counts"]), 1), 1),
            "avg_violations_per_image": round(sum(_OPS["viol_counts"]) / max(len(_OPS["viol_counts"]), 1), 2),
            "eds_above_75": sum(1 for s in eds if s >= 75),
            "eds_45_to_75": sum(1 for s in eds if 45 <= s < 75),
            "eds_below_45": sum(1 for s in eds if s < 45),
        },
        "accuracy_methodology": {
            "note": "Precision / Recall / F1 / mAP below are computed on a human-labelled "
                    "validation set (18 images, 416 ground-truth objects, 74 violations) "
                    "via evaluate.py. Operational metrics above are label-free measurements "
                    "from real pipeline runs this session.",
            "metrics_supported": ["Precision", "Recall", "F1-score", "mAP@0.5", "mAP@0.75",
                                  "plate exact-match", "plate partial-match"],
        },
        "benchmark": _load_benchmark(),
    }


# ════════════════════════════════════════════════════════════════
# DEMO & EVIDENCE PDF
# ════════════════════════════════════════════════════════════════

DEMO_DIR = PROJECT_ROOT / "demo"

DEMO_IMAGES = [
    {"file": "1_helmet.png", "title": "Helmet Violation", "description": "Two-wheeler riders without helmets"},
    {"file": "2_triple_riding.png", "title": "Triple Riding", "description": "Three persons on a single two-wheeler"},
    {"file": "3_stopline_parking.png", "title": "Stop-line & Parking", "description": "Vehicles crossing zebra crossing + illegal parking"},
    {"file": "4_red_light.jpg", "title": "Red-light Violation", "description": "Vehicles running a red signal"},
    {"file": "5_wrong_side.jpg", "title": "Wrong-side Driving", "description": "Vehicle isolated on the opposing lane"},
]


@app.get("/api/demo")
def demo_list():
    """List curated demo images for one-click analysis."""
    return {"images": DEMO_IMAGES}


@app.get("/api/demo/{filename}")
def demo_image(filename: str):
    """Serve a demo image file."""
    path = DEMO_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Demo image not found")
    from fastapi.responses import FileResponse as FR  # noqa: E811
    return FR(str(path))


@app.post("/api/demo/analyze/{filename}")
async def demo_analyze(filename: str):
    """Analyze a curated demo image (no upload needed)."""
    path = DEMO_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Demo image not found")
    image = cv2.imread(str(path))
    if image is None:
        raise HTTPException(status_code=500, detail="Could not read demo image")
    return process_image(image)


@app.get("/api/evidence/pdf/{violation_id}")
def evidence_pdf(violation_id: str):
    """Generate a court-ready PDF evidence package for a specific violation.

    Contains: annotated image, violation metadata, Evidence Defensibility Score,
    hash-chain integrity seal, plain-English explanation, and timestamps.
    """
    from io import BytesIO
    from fastapi.responses import StreamingResponse  # noqa: E811

    db = get_database()
    db._load(load_images=True)
    rec = next((r for r in db.records if r.get("violation_id") == violation_id), None)
    if rec is None:
        raise HTTPException(status_code=404, detail="violation_id not found")

    pdf_bytes = _generate_evidence_pdf(rec)
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=evidence_{violation_id}.pdf"},
    )


def _generate_evidence_pdf(record: dict) -> bytes:
    """Render a single-violation evidence PDF using reportlab-free pure approach.

    Uses FPDF2 if available, otherwise falls back to a minimal PDF built by hand.
    """
    try:
        from fpdf import FPDF
        return _pdf_with_fpdf(record)
    except ImportError:
        return _pdf_minimal(record)


def _pdf_with_fpdf(record: dict) -> bytes:
    from fpdf import FPDF
    from io import BytesIO
    import tempfile

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "TRAFFIC VIOLATION EVIDENCE REPORT", ln=True, align="C")
    pdf.ln(5)

    # Violation ID and timestamp
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Violation ID: {record.get('violation_id', 'N/A')}", ln=True)
    pdf.cell(0, 6, f"Timestamp: {record.get('timestamp', 'N/A')}", ln=True)
    pdf.cell(0, 6, f"Generated: {__import__('datetime').datetime.now().isoformat(timespec='seconds')}", ln=True)
    pdf.ln(5)

    # Violation details
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Violation Details", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Type: {record.get('violation_type', '').replace('_', ' ').title()}", ln=True)
    pdf.cell(0, 6, f"Severity: {record.get('severity', 'N/A').upper()}", ln=True)
    pdf.cell(0, 6, f"Confidence: {record.get('confidence', 0)*100:.1f}%", ln=True)
    pdf.cell(0, 6, f"Description: {record.get('description', 'N/A')}", ln=True)
    plate = record.get("plate_text", "")
    if plate:
        pdf.cell(0, 6, f"Vehicle Plate: {plate}", ln=True)
    pdf.ln(5)

    # Evidence image
    evidence_b64 = record.get("evidence_image", "")
    if evidence_b64:
        try:
            img_data = base64.b64decode(evidence_b64.split(",")[1] if "," in evidence_b64 else evidence_b64)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(img_data)
                tmp_path = tmp.name
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Flagged Vehicle (Context Crop)", ln=True)
            pdf.image(tmp_path, w=120)
            pdf.ln(5)
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

    # Plate crop image
    plate_crop_b64 = record.get("plate_crop_image", "")
    if plate_crop_b64:
        try:
            img_data = base64.b64decode(plate_crop_b64.split(",")[1] if "," in plate_crop_b64 else plate_crop_b64)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(img_data)
                tmp_path = tmp.name
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "License Plate (OCR Crop)", ln=True)
            pdf.image(tmp_path, w=80)
            pdf.ln(5)
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

    # Evidence Defensibility Score
    seal = record.get("seal", {})
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Evidence Integrity", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Content Hash (SHA-256): {seal.get('content_hash', 'N/A')}", ln=True)
    pdf.cell(0, 6, f"Record Hash: {seal.get('record_hash', 'N/A')}", ln=True)
    pdf.cell(0, 6, f"Previous Hash: {seal.get('prev_hash', 'N/A')}", ln=True)
    pdf.cell(0, 6, f"Sealed At: {seal.get('sealed_at', 'N/A')}", ln=True)
    pdf.ln(5)

    # Footer
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 5, "This document was auto-generated by TrafficVioLens. Hash chain ensures tamper-evidence.", ln=True)
    pdf.cell(0, 5, "Any modification to this record is detectable via the append-only hash chain.", ln=True)

    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()


def _pdf_minimal(record: dict) -> bytes:
    """Minimal PDF without external dependencies (text-only fallback)."""
    vid = record.get("violation_id", "N/A")
    vtype = record.get("violation_type", "").replace("_", " ").title()
    severity = record.get("severity", "N/A").upper()
    conf = f"{record.get('confidence', 0)*100:.1f}%"
    plate = record.get("plate_text", "N/A")
    ts = record.get("timestamp", "N/A")
    seal = record.get("seal", {})
    content = f"""TRAFFIC VIOLATION EVIDENCE REPORT
{'='*40}
Violation ID: {vid}
Timestamp: {ts}
Type: {vtype}
Severity: {severity}
Confidence: {conf}
Vehicle Plate: {plate}
Description: {record.get('description', 'N/A')}

EVIDENCE INTEGRITY
Content Hash: {seal.get('content_hash', 'N/A')}
Record Hash: {seal.get('record_hash', 'N/A')}
Previous Hash: {seal.get('prev_hash', 'N/A')}
Sealed At: {seal.get('sealed_at', 'N/A')}

Note: This is a tamper-evident record. Any modification
is detectable via the append-only hash chain.
Generated by TrafficVioLens.
"""
    # Build a minimal valid PDF
    lines = content.encode("latin-1", errors="replace")
    objects = []
    objects.append(b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj")
    objects.append(b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj")
    objects.append(b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj")
    # Stream
    text_ops = b"BT /F1 10 Tf 50 742 Td 12 TL\n"
    for line in content.split("\n"):
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        text_ops += f"({escaped}) '\n".encode("latin-1", errors="replace")
    text_ops += b"ET"
    stream = text_ops
    objects.append(f"4 0 obj<</Length {len(stream)}>>stream\n".encode() + stream + b"\nendstream endobj")
    objects.append(b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Courier>>endobj")
    # Build file
    out = b"%PDF-1.4\n"
    offsets = []
    for obj in objects:
        offsets.append(len(out))
        out += obj + b"\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer<</Size {len(objects)+1}/Root 1 0 R>>\nstartxref\n{xref_pos}\n%%EOF".encode()
    return out


# ════════════════════════════════════════════════════════════════
# STATIC FRONTEND (serve the built React app from the same server)
# ════════════════════════════════════════════════════════════════
# In the Docker image the React build is copied to ./static next to this file.
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

STATIC_DIR = PROJECT_ROOT / "static"

if STATIC_DIR.exists():
    # Serve hashed JS/CSS assets
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

    @app.get("/")
    def serve_index():
        return FileResponse(str(STATIC_DIR / "index.html"))

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        """SPA fallback: serve real files if present, else index.html for client routing."""
        candidate = STATIC_DIR / full_path
        if candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(STATIC_DIR / "index.html"))

