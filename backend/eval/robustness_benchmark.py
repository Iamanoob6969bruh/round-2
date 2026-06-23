"""
TrafficVioLens — Degraded-Image Robustness Benchmark.

Quantifies the value of the feedback-loop enhancer by comparing detection
performance on synthetically degraded images WITH vs WITHOUT enhancement.

Degradations applied:
  - Low-light (gamma darkening)
  - Motion blur (directional kernel)
  - Rain overlay (diagonal streak noise)

Writes eval/robustness_results.json — consumed by the /api/evaluation endpoint.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))

from src.preprocessing.feedback_enhancer import FeedbackLoopEnhancer  # noqa: E402
from src.detection.detector import VehicleDetector  # noqa: E402

IMAGES_DIR = HERE / "images"
RESULTS_PATH = HERE / "robustness_results.json"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _apply_low_light(img):
    table = np.array([((i / 255.0) ** 2.5) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def _apply_motion_blur(img):
    k = 15
    kernel = np.zeros((k, k))
    kernel[k // 2, :] = np.ones(k) / k
    return cv2.filter2D(img, -1, kernel)


def _apply_rain(img):
    rain = np.zeros_like(img)
    h, w = img.shape[:2]
    for _ in range(int(h * w * 0.001)):
        x = np.random.randint(0, w - 3)
        y = np.random.randint(0, h - 20)
        length = np.random.randint(10, 25)
        cv2.line(rain, (x, y), (x + 2, y + length), (180, 180, 200), 1)
    return cv2.addWeighted(img, 0.85, rain, 0.4, 0)


DEGRADATIONS = {
    "low_light": _apply_low_light,
    "motion_blur": _apply_motion_blur,
    "rain": _apply_rain,
}


def run():
    images = sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        print("[!] No images in eval/images/"); return

    # Use a subset (up to 8) to keep benchmark fast on CPU
    images = images[:8]

    yolo = BACKEND / "data" / "yolov8n.pt"
    helmet = BACKEND / "data" / "models" / "helmet_v2" / "best.pt"
    enhancer = FeedbackLoopEnhancer()
    detector = VehicleDetector(
        model_path=str(yolo),
        helmet_model_path=str(helmet) if helmet.exists() else None,
        confidence_threshold=0.3,
    )

    results = {"images_used": len(images), "conditions": {}}

    print(f"\nRobustness benchmark on {len(images)} images...\n")

    for deg_name, deg_fn in DEGRADATIONS.items():
        raw_counts, raw_confs = [], []
        enh_counts, enh_confs = [], []

        for path in images:
            img = cv2.imread(str(path))
            if img is None:
                continue
            degraded = deg_fn(img.copy())

            # WITHOUT enhancer
            dets_raw = detector.detect(degraded)
            n_raw = len(dets_raw.vehicles) + len(dets_raw.persons)
            confs_raw = [d.confidence for d in dets_raw.vehicles + dets_raw.persons]
            raw_counts.append(n_raw)
            raw_confs.extend(confs_raw)

            # WITH enhancer
            enhanced, _ = enhancer.enhance(degraded)
            dets_enh = detector.detect(enhanced)
            n_enh = len(dets_enh.vehicles) + len(dets_enh.persons)
            confs_enh = [d.confidence for d in dets_enh.vehicles + dets_enh.persons]
            enh_counts.append(n_enh)
            enh_confs.extend(confs_enh)

        avg_raw = round(sum(raw_counts) / max(len(raw_counts), 1), 1)
        avg_enh = round(sum(enh_counts) / max(len(enh_counts), 1), 1)
        conf_raw = round(np.mean(raw_confs), 3) if raw_confs else 0
        conf_enh = round(np.mean(enh_confs), 3) if enh_confs else 0
        improvement = round((avg_enh - avg_raw) / max(avg_raw, 1) * 100, 1)

        results["conditions"][deg_name] = {
            "without_enhancer": {"avg_detections": avg_raw, "avg_confidence": conf_raw},
            "with_enhancer": {"avg_detections": avg_enh, "avg_confidence": conf_enh},
            "detection_improvement_pct": improvement,
        }
        print(f"  {deg_name:12s}: raw={avg_raw:.1f} det (conf {conf_raw:.3f}) → enhanced={avg_enh:.1f} det (conf {conf_enh:.3f}) | +{improvement}%")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {RESULTS_PATH}\n")


if __name__ == "__main__":
    run()
