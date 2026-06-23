"""
TrafficVioLens — Evaluation Benchmark Harness.

Computes REAL performance metrics (Precision, Recall, F1, mAP, plate-match
rate) by comparing the pipeline's predictions against human-labelled ground
truth. This is the honest, defensible way to report accuracy — no fabricated
numbers; every figure is derived from labelled data you provide.

USAGE
-----
1. Put your evaluation images in   eval/images/
2. Label them in                   eval/ground_truth.json   (see template below)
3. Run                             python evaluate.py
4. Read the printed report + the written eval/results.json

GROUND-TRUTH FORMAT (eval/ground_truth.json)
--------------------------------------------
{
  "image_name.jpg": {
    "detections": [
      {"bbox": [x1, y1, x2, y2], "class": "motorcycle"},
      {"bbox": [x1, y1, x2, y2], "class": "person"}
    ],
    "violations": [
      {"type": "helmet_violation", "bbox": [x1, y1, x2, y2]}
    ],
    "plates": [
      {"text": "KA18EJ8800"}
    ]
  },
  ...
}

Only label what you can see — boxes are in pixel coordinates of the ORIGINAL
image. The harness handles everything else.
"""

import json
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.preprocessing.feedback_enhancer import FeedbackLoopEnhancer
from src.detection.detector import VehicleDetector
from src.violations.rules_engine import ViolationRulesEngine
from src.plate_recognition.flagged_plate_reader import FlaggedVehiclePlateReader
from src.evaluation.metrics import (
    evaluate_detections, compute_map, evaluate_violations,
    evaluate_plate_recognition, StageTimer,
)

EVAL_DIR = HERE / "eval"
IMAGES_DIR = EVAL_DIR / "images"
GT_PATH = EVAL_DIR / "ground_truth.json"
RESULTS_PATH = EVAL_DIR / "results.json"


def _load_pipeline():
    base = HERE
    yolo = base / "data" / "yolov8n.pt"
    helmet = base / "data" / "models" / "helmet_v2" / "best.pt"
    enhancer = FeedbackLoopEnhancer()
    detector = VehicleDetector(
        model_path=str(yolo),
        helmet_model_path=str(helmet) if helmet.exists() else None,
        confidence_threshold=0.3,
    )
    return enhancer, detector, ViolationRulesEngine(), FlaggedVehiclePlateReader()


VIOLATION_TYPES = [
    "helmet_violation", "triple_riding", "seatbelt_violation",
    "wrong_side_driving", "stop_line_violation", "red_light_violation",
    "illegal_parking",
]


def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = (a[2]-a[0]) * (a[3]-a[1])
    ab = (b[2]-b[0]) * (b[3]-b[1])
    return inter / max(aa + ab - inter, 1)


def _build_confusion_matrix(pred_viols, gt_viols):
    """Build a confusion matrix: for each GT violation, find its best-matching
    prediction (IoU ≥ 0.3) and record predicted type vs actual type.
    Unmatched GTs go to 'missed'; unmatched preds go to 'false_alarm'."""
    labels = VIOLATION_TYPES
    n = len(labels)
    idx = {t: i for i, t in enumerate(labels)}
    # matrix[actual][predicted] — rows=actual, cols=predicted
    matrix = [[0] * (n + 1) for _ in range(n + 1)]  # +1 col = missed, +1 row = false alarm

    matched_preds = set()
    for gt in gt_viols:
        gt_type = gt["type"]
        gt_bbox = tuple(gt["bbox"])
        best_iou, best_idx = 0, -1
        for pi, p in enumerate(pred_viols):
            if pi in matched_preds:
                continue
            iou = _iou(gt_bbox, tuple(p["bbox"]))
            if iou > best_iou:
                best_iou = iou
                best_idx = pi
        row = idx.get(gt_type, n)
        if best_iou >= 0.3 and best_idx >= 0:
            matched_preds.add(best_idx)
            pred_type = pred_viols[best_idx]["type"]
            col = idx.get(pred_type, n)
            matrix[row][col] += 1
        else:
            matrix[row][n] += 1  # missed

    # Unmatched predictions = false alarms
    for pi, p in enumerate(pred_viols):
        if pi not in matched_preds:
            col = idx.get(p["type"], n)
            matrix[n][col] += 1

    return {
        "labels": labels + ["_unmatched"],
        "matrix": matrix,
    }


def run(allow_unverified: bool = False):
    if not GT_PATH.exists():
        print(f"[!] No ground truth found at {GT_PATH}")
        print("    Create it using the format documented at the top of this file,")
        print("    or copy eval/ground_truth_template.json and fill it in.")
        return

    ground_truth = json.loads(GT_PATH.read_text())

    # ── Verification gate (defensibility) ──────────────────────────────────
    # Pre-annotated drafts are the model grading ITSELF — scoring them would
    # produce meaningless ~100% numbers. Refuse unless a human has verified the
    # labels (_meta.verified = true) or the operator explicitly overrides.
    meta = ground_truth.get("_meta", {})
    image_items = {k: v for k, v in ground_truth.items() if not k.startswith("_")}

    if not meta.get("verified", False) and not allow_unverified:
        print("[!] Ground truth is NOT marked verified (_meta.verified != true).")
        print("    This looks like an unreviewed draft from prelabel.py. Scoring it")
        print("    would grade the model against its own predictions — meaningless.")
        print("    Verify the labels in eval/preview/, set _meta.verified = true,")
        print("    then re-run. To override anyway (NOT defensible), pass")
        print("    --allow-unverified.")
        return
    if not meta.get("verified", False) and allow_unverified:
        print("[warn] Running on UNVERIFIED labels (--allow-unverified). These numbers")
        print("       are NOT defensible — for debugging the harness only.\n")

    enhancer, detector, rules, plate_reader = _load_pipeline()

    # Accumulators across the whole eval set
    all_pred_dets, all_gt_dets = [], []
    all_pred_viols, all_gt_viols = [], []
    all_pred_plates, all_gt_plates = [], []
    latencies = []

    print(f"\nEvaluating {len(image_items)} labelled image(s)...\n")

    for name, gt in image_items.items():
        img_path = IMAGES_DIR / name
        if not img_path.exists():
            print(f"  [skip] {name} — image not found")
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [skip] {name} — could not read")
            continue

        timer = StageTimer()
        timer.start("total")
        enhanced, _ = enhancer.enhance(image)
        detections = detector.detect(enhanced)
        violations = rules.detect_all_violations(detections, image_shape=enhanced.shape, image=enhanced)
        plate_recs = plate_reader.read_for_violations(enhanced, violations)
        timer.stop()
        latencies.append(timer.stages["total"])

        # The enhancer may upscale the frame; predictions are in ENHANCED coords
        # while ground truth is labelled in ORIGINAL image pixels. Map every
        # predicted box back to original scale so IoU vs ground truth is valid.
        oh, ow = image.shape[:2]
        eh, ew = enhanced.shape[:2]
        sx, sy = (ow / ew if ew else 1.0), (oh / eh if eh else 1.0)

        def _to_orig(bbox):
            x1, y1, x2, y2 = bbox
            return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)

        # Collect predictions in the metric format (original coordinates)
        for d in detections.vehicles + detections.persons:
            all_pred_dets.append({"bbox": _to_orig(d.bbox), "class": d.class_name, "confidence": d.confidence})
        for v in violations:
            all_pred_viols.append({"type": v.violation_type, "bbox": _to_orig(v.bbox), "confidence": v.confidence})
        for pr in plate_recs:
            if pr.plate_text:
                all_pred_plates.append({"text": pr.plate_text})

        # Collect ground truth
        for d in gt.get("detections", []):
            all_gt_dets.append({"bbox": tuple(d["bbox"]), "class": d["class"]})
        for v in gt.get("violations", []):
            all_gt_viols.append({"type": v["type"], "bbox": tuple(v["bbox"])})
        for p in gt.get("plates", []):
            all_gt_plates.append({"text": p["text"]})

        print(f"  [ok] {name}: {len(detections.vehicles)} vehicles, "
              f"{len(violations)} violations, "
              f"{sum(1 for p in plate_recs if p.plate_text)} plates read")

    # ── Compute real metrics ──
    det_metrics = evaluate_detections(all_pred_dets, all_gt_dets, iou_threshold=0.5)
    map_metrics = compute_map(all_pred_dets, all_gt_dets, iou_thresholds=[0.5, 0.75])
    viol_metrics = evaluate_violations(all_pred_viols, all_gt_viols)
    plate_metrics = evaluate_plate_recognition(all_pred_plates, all_gt_plates)

    # ── Confusion matrix for violation classification ──
    confusion = _build_confusion_matrix(all_pred_viols, all_gt_viols)

    avg_latency = round(sum(latencies) / max(len(latencies), 1), 1)
    fps = round(1000 / max(avg_latency, 1), 2)

    results = {
        "images_evaluated": len(latencies),
        "detection": det_metrics["overall"],
        "detection_per_class": det_metrics["per_class"],
        "mAP": map_metrics,
        "violations": viol_metrics["overall"],
        "violations_per_type": viol_metrics.get("per_class", {}),
        "confusion_matrix": confusion,
        "plate_recognition": plate_metrics,
        "performance": {"avg_latency_ms": avg_latency, "throughput_fps": fps},
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    # ── Pretty report ──
    print("\n" + "=" * 60)
    print("  TRAFFICVIOLENS — EVALUATION RESULTS")
    print("=" * 60)
    o = det_metrics["overall"]
    print(f"\n  DETECTION (IoU=0.5)")
    print(f"    Precision : {o['precision']:.3f}")
    print(f"    Recall    : {o['recall']:.3f}")
    print(f"    F1-score  : {o['f1']:.3f}")
    print(f"    Accuracy  : {o['accuracy']:.3f}")
    print(f"    mAP@0.5   : {map_metrics['mAP']:.3f}")
    v = viol_metrics["overall"]
    print(f"\n  VIOLATION CLASSIFICATION (IoU=0.3)")
    print(f"    Precision : {v['precision']:.3f}   Recall: {v['recall']:.3f}   F1: {v['f1']:.3f}")
    print(f"\n  PLATE RECOGNITION")
    print(f"    Exact-match rate   : {plate_metrics.get('exact_match_rate', 0):.3f}")
    print(f"    Partial-match rate : {plate_metrics.get('partial_match_rate', 0):.3f}")
    print(f"\n  PERFORMANCE")
    print(f"    Avg latency : {avg_latency} ms   Throughput: {fps} FPS")
    print("\n  Full results written to eval/results.json")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compute real accuracy metrics against labelled ground truth.")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="Score even if labels are not human-verified (NOT defensible; debug only).")
    run(allow_unverified=ap.parse_args().allow_unverified)
