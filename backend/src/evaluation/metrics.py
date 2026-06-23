"""
Performance Evaluation Module.

Provides:
  1. Detection metrics: Precision, Recall, F1-score, mAP (IoU-based)
  2. Pipeline timing: per-stage latency measurements
  3. Scalability assessment: throughput vs image resolution
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# TIMING
# ============================================================

@dataclass
class StageTimer:
    """Records execution time for each pipeline stage."""
    stages: dict = field(default_factory=dict)
    _start: float = 0.0
    _current_stage: str = ""

    def start(self, stage_name: str):
        self._current_stage = stage_name
        self._start = time.perf_counter()

    def stop(self):
        if self._current_stage:
            elapsed = (time.perf_counter() - self._start) * 1000  # ms
            self.stages[self._current_stage] = round(elapsed, 1)
            self._current_stage = ""

    def total_ms(self) -> float:
        return round(sum(self.stages.values()), 1)

    def summary(self) -> dict:
        return {
            "stages_ms": dict(self.stages),
            "total_ms": self.total_ms(),
            "fps": round(1000 / max(self.total_ms(), 1), 2),
        }


# ============================================================
# DETECTION METRICS
# ============================================================

def compute_iou(box_a, box_b) -> float:
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    """Compute precision, recall, F1 from counts."""
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


def evaluate_detections(predictions: list, ground_truths: list, iou_threshold: float = 0.5) -> dict:
    """
    Evaluate detection predictions against ground truth.

    Args:
        predictions: list of {"bbox": (x1,y1,x2,y2), "class": str, "confidence": float}
        ground_truths: list of {"bbox": (x1,y1,x2,y2), "class": str}
        iou_threshold: minimum IoU to count as a match

    Returns:
        dict with per-class and overall metrics
    """
    # Sort predictions by confidence (descending)
    preds = sorted(predictions, key=lambda p: p["confidence"], reverse=True)
    gt_matched = [False] * len(ground_truths)

    tp, fp = 0, 0
    per_class = {}

    for pred in preds:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truths):
            if gt_matched[gt_idx]:
                continue
            if gt["class"] != pred["class"]:
                continue
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        cls = pred["class"]
        if cls not in per_class:
            per_class[cls] = {"tp": 0, "fp": 0, "fn": 0}

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            per_class[cls]["tp"] += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1
            per_class[cls]["fp"] += 1

    # Count false negatives (unmatched ground truths)
    fn = sum(1 for m in gt_matched if not m)
    for gt_idx, gt in enumerate(ground_truths):
        if not gt_matched[gt_idx]:
            cls = gt["class"]
            if cls not in per_class:
                per_class[cls] = {"tp": 0, "fp": 0, "fn": 0}
            per_class[cls]["fn"] += 1

    # Compute metrics
    overall = precision_recall_f1(tp, fp, fn)
    overall["accuracy"] = round(tp / max(tp + fp + fn, 1), 4)

    class_metrics = {}
    for cls, counts in per_class.items():
        class_metrics[cls] = precision_recall_f1(counts["tp"], counts["fp"], counts["fn"])

    return {"overall": overall, "per_class": class_metrics}


def compute_ap(precisions: list, recalls: list) -> float:
    """Compute Average Precision using 11-point interpolation."""
    if not precisions or not recalls:
        return 0.0

    # Sort by recall
    pairs = sorted(zip(recalls, precisions), key=lambda x: x[0])
    recalls_sorted = [p[0] for p in pairs]
    precisions_sorted = [p[1] for p in pairs]

    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p_interp = 0.0
        for r, p in zip(recalls_sorted, precisions_sorted):
            if r >= t:
                p_interp = max(p_interp, p)
        ap += p_interp
    return round(ap / 11, 4)


def compute_map(predictions: list, ground_truths: list, iou_thresholds: list = None) -> dict:
    """
    Compute mean Average Precision across IoU thresholds.

    Args:
        predictions: list of {"bbox", "class", "confidence"}
        ground_truths: list of {"bbox", "class"}
        iou_thresholds: list of thresholds (default: [0.5])
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    aps = []
    for iou_thresh in iou_thresholds:
        result = evaluate_detections(predictions, ground_truths, iou_thresh)
        # Use overall precision/recall as single point for AP
        p = result["overall"]["precision"]
        r = result["overall"]["recall"]
        aps.append(compute_ap([p], [r]))

    return {
        "mAP": round(float(np.mean(aps)), 4),
        "mAP_thresholds": iou_thresholds,
        "per_threshold_AP": dict(zip([str(t) for t in iou_thresholds], aps)),
    }


# ============================================================
# VIOLATION EVALUATION
# ============================================================

def evaluate_violations(predicted_violations: list, ground_truth_violations: list) -> dict:
    """
    Evaluate violation detection accuracy.

    Args:
        predicted_violations: list of {"type": str, "bbox": tuple}
        ground_truth_violations: list of {"type": str, "bbox": tuple}
    """
    # Treat as detection problem with violation types as classes
    preds = [{"bbox": v["bbox"], "class": v["type"], "confidence": v.get("confidence", 1.0)}
             for v in predicted_violations]
    gts = [{"bbox": v["bbox"], "class": v["type"]} for v in ground_truth_violations]

    return evaluate_detections(preds, gts, iou_threshold=0.3)


def evaluate_plate_recognition(predicted_plates: list, ground_truth_plates: list) -> dict:
    """
    Evaluate plate OCR accuracy.

    Args:
        predicted_plates: list of {"text": str}
        ground_truth_plates: list of {"text": str}
    """
    if not ground_truth_plates:
        return {"accuracy": 0.0, "exact_match": 0, "partial_match": 0, "total": 0}

    exact = 0
    partial = 0
    total = len(ground_truth_plates)

    for gt in ground_truth_plates:
        gt_text = gt["text"].replace(" ", "").upper()
        best_match = 0.0
        for pred in predicted_plates:
            pred_text = pred["text"].replace(" ", "").upper()
            if pred_text == gt_text:
                exact += 1
                best_match = 1.0
                break
            # Partial match: character-level accuracy
            if gt_text and pred_text:
                common = sum(1 for a, b in zip(pred_text, gt_text) if a == b)
                ratio = common / max(len(gt_text), len(pred_text))
                best_match = max(best_match, ratio)
        if best_match >= 0.7 and best_match < 1.0:
            partial += 1

    return {
        "exact_match_rate": round(exact / total, 4),
        "partial_match_rate": round((exact + partial) / total, 4),
        "exact_matches": exact,
        "partial_matches": partial,
        "total_ground_truth": total,
    }
