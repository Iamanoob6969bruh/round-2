"""
TrafficVioLens — Pre-annotation helper (turns labelling into *verification*).

Hand-labelling every box from scratch is slow. This tool runs the SAME pipeline
used at inference over every image in ``eval/images/`` and writes a *draft*
ground-truth file pre-filled with the model's predictions, plus a visual preview
of every image with the predicted boxes drawn on it.

Your job then shrinks to: open each preview, and in the draft JSON
  • delete boxes the model hallucinated,
  • add boxes/violations the model missed,
  • fix wrong classes and plate text.

⚠️  CRITICAL HONESTY NOTE
The draft is the model grading itself — if you evaluate against it unchanged you
will get meaningless ~100% scores. The draft is therefore marked
``_meta.verified = false`` and ``evaluate.py`` REFUSES to score it until you have
reviewed it and set ``verified`` to true (done for you by ``--accept`` once you
are satisfied, or manually). Only human-verified labels yield defensible numbers.

USAGE
-----
    python eval/prelabel.py                 # draft + previews for eval/images/*
    python eval/prelabel.py --out eval/ground_truth.draft.json

Then review eval/preview/*.jpg, correct eval/ground_truth.draft.json, rename it
to eval/ground_truth.json, set _meta.verified=true, and run evaluate.py.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))

from src.preprocessing.feedback_enhancer import FeedbackLoopEnhancer  # noqa: E402
from src.detection.detector import VehicleDetector  # noqa: E402
from src.violations.rules_engine import ViolationRulesEngine  # noqa: E402
from src.plate_recognition.flagged_plate_reader import FlaggedVehiclePlateReader  # noqa: E402

IMAGES_DIR = HERE / "images"
PREVIEW_DIR = HERE / "preview"
DEFAULT_OUT = HERE / "ground_truth.draft.json"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Colours (BGR) for the preview overlay
C_VEHICLE = (0, 200, 0)
C_PERSON = (255, 160, 0)
C_VIOLATION = (0, 0, 255)
C_PLATE = (0, 215, 255)


def _load_pipeline():
    yolo = BACKEND / "data" / "yolov8n.pt"
    helmet = BACKEND / "data" / "models" / "helmet_v2" / "best.pt"
    enhancer = FeedbackLoopEnhancer()
    detector = VehicleDetector(
        model_path=str(yolo),
        helmet_model_path=str(helmet) if helmet.exists() else None,
        confidence_threshold=0.3,
    )
    return enhancer, detector, ViolationRulesEngine(), FlaggedVehiclePlateReader()


def _draw(img, bbox, color, label, thick=2):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    if label:
        cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def prelabel(out_path: Path):
    if not IMAGES_DIR.exists():
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f"[!] No images found in {IMAGES_DIR}")
        print("    Drop traffic images there first, then re-run.")
        return 1

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    enhancer, detector, rules, plate_reader = _load_pipeline()

    draft = {
        "_meta": {
            "verified": False,
            "generated_by": "prelabel.py",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": (
                "DRAFT — these are MODEL PREDICTIONS, not ground truth. Review each "
                "preview in eval/preview/, correct the boxes/classes/plates below, "
                "then set verified=true. evaluate.py refuses to score until then."
            ),
            "review_checklist": [
                "Delete boxes the model hallucinated.",
                "Add objects/violations the model missed.",
                "Fix any wrong vehicle class.",
                "Fix plate text to exactly match what is legible (uppercase, no spaces).",
                "Remove a violation entry if it is actually NOT a violation.",
            ],
        }
    }

    print(f"\nPre-annotating {len(images)} image(s) from {IMAGES_DIR} ...\n")
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"  [skip] {path.name} — could not read")
            continue

        enhanced, _ = enhancer.enhance(image)
        detections = detector.detect(enhanced)
        violations = rules.detect_all_violations(detections, image_shape=enhanced.shape, image=enhanced)
        plate_recs = plate_reader.read_for_violations(enhanced, violations)

        # The enhancer may upscale the frame, so detections come back in
        # ENHANCED coordinates. Ground truth is labelled in ORIGINAL image
        # pixels, so map every predicted box back to the original scale before
        # drawing/recording — otherwise boxes land off-frame and IoU is wrong.
        oh, ow = image.shape[:2]
        eh, ew = enhanced.shape[:2]
        sx, sy = (ow / ew if ew else 1.0), (oh / eh if eh else 1.0)

        def to_orig(bbox):
            x1, y1, x2, y2 = bbox
            return [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)]

        preview = image.copy()
        det_records = []
        for d in detections.vehicles + detections.persons:
            color = C_PERSON if d.class_name == "person" else C_VEHICLE
            obox = to_orig(d.bbox)
            _draw(preview, obox, color, f"{d.class_name} {d.confidence:.0%}")
            det_records.append({
                "bbox": obox,
                "class": d.class_name,
                "_pred_confidence": round(float(d.confidence), 3),
            })

        viol_records = []
        for v in violations:
            obox = to_orig(v.bbox)
            _draw(preview, obox, C_VIOLATION,
                  v.violation_type.replace("_", " ").upper(), thick=3)
            viol_records.append({
                "type": v.violation_type,
                "bbox": obox,
                "_pred_confidence": round(float(v.confidence), 3),
            })

        plate_records = []
        for pr in plate_recs:
            if pr.plate_text:
                plate_records.append({
                    "text": pr.plate_text,
                    "_pred_confidence": round(float(pr.confidence), 3),
                })

        draft[path.name] = {
            "detections": det_records,
            "violations": viol_records,
            "plates": plate_records,
        }

        preview_path = PREVIEW_DIR / path.name
        cv2.imwrite(str(preview_path), preview)
        print(f"  [ok] {path.name}: {len(det_records)} detections, "
              f"{len(viol_records)} violations, {len(plate_records)} plates "
              f"→ preview/{path.name}")

    out_path.write_text(json.dumps(draft, indent=2))
    print(f"\nDraft written to {out_path}")
    print(f"Previews written to {PREVIEW_DIR}/")
    print("\nNEXT STEPS")
    print("  1. Open each image in eval/preview/ and compare to the draft labels.")
    print("  2. Correct eval/ground_truth.draft.json (delete/add/fix; drop the _pred_confidence hints).")
    print("  3. Rename it to eval/ground_truth.json and set _meta.verified = true.")
    print("  4. Run:  python evaluate.py\n")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pre-annotate eval images into a draft ground-truth file.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output draft JSON path.")
    raise SystemExit(prelabel(ap.parse_args().out))
