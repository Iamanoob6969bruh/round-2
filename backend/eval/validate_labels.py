"""
TrafficVioLens — Ground-truth validator.

Catches labelling mistakes BEFORE you run the benchmark, so your metrics are not
silently corrupted by typos, out-of-bounds boxes, or invalid class/violation
names. Run it on either the draft or the final ground-truth file.

USAGE
-----
    python eval/validate_labels.py                       # checks eval/ground_truth.json
    python eval/validate_labels.py eval/ground_truth.draft.json

Exit code 0 = clean, 1 = errors found.
"""

import json
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
IMAGES_DIR = HERE / "images"
DEFAULT_GT = HERE / "ground_truth.json"

# Allowed label vocabulary — keep in lock-step with the detector / rules engine.
ALLOWED_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle", "auto", "person"}
ALLOWED_VIOLATIONS = {
    "helmet_violation", "triple_riding", "seatbelt_violation",
    "wrong_side_driving", "stop_line_violation", "red_light_violation",
    "illegal_parking",
}


def _check_bbox(bbox, w, h, where, errors):
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        errors.append(f"{where}: bbox must be [x1,y1,x2,y2], got {bbox!r}")
        return
    x1, y1, x2, y2 = bbox
    if not all(isinstance(v, (int, float)) for v in bbox):
        errors.append(f"{where}: bbox values must be numbers, got {bbox!r}")
        return
    if x2 <= x1 or y2 <= y1:
        errors.append(f"{where}: bbox not well-formed (x2>x1 and y2>y1 required): {bbox}")
    if w and h and (x1 < 0 or y1 < 0 or x2 > w or y2 > h):
        errors.append(f"{where}: bbox {bbox} out of image bounds ({w}x{h})")


def validate(gt_path: Path) -> int:
    if not gt_path.exists():
        print(f"[!] No ground-truth file at {gt_path}")
        return 1

    try:
        data = json.loads(gt_path.read_text())
    except json.JSONDecodeError as e:
        print(f"[!] {gt_path} is not valid JSON: {e}")
        return 1

    errors, warnings = [], []
    meta = data.get("_meta", {})
    image_keys = [k for k in data if not k.startswith("_")]

    if not image_keys:
        errors.append("No image entries found (only metadata).")

    if not meta.get("verified"):
        warnings.append(
            "_meta.verified is not true — this looks like an unreviewed draft. "
            "evaluate.py will refuse to score it until you verify the labels."
        )

    for name in image_keys:
        entry = data[name]
        img_path = IMAGES_DIR / name
        w = h = None
        if not img_path.exists():
            errors.append(f"{name}: image not found in {IMAGES_DIR}")
        else:
            img = cv2.imread(str(img_path))
            if img is None:
                errors.append(f"{name}: image could not be read")
            else:
                h, w = img.shape[:2]

        if not isinstance(entry, dict):
            errors.append(f"{name}: entry must be an object")
            continue

        for i, d in enumerate(entry.get("detections", [])):
            where = f"{name} detections[{i}]"
            if d.get("class") not in ALLOWED_CLASSES:
                errors.append(f"{where}: class {d.get('class')!r} not in {sorted(ALLOWED_CLASSES)}")
            _check_bbox(d.get("bbox"), w, h, where, errors)

        for i, v in enumerate(entry.get("violations", [])):
            where = f"{name} violations[{i}]"
            if v.get("type") not in ALLOWED_VIOLATIONS:
                errors.append(f"{where}: type {v.get('type')!r} not in {sorted(ALLOWED_VIOLATIONS)}")
            _check_bbox(v.get("bbox"), w, h, where, errors)

        for i, p in enumerate(entry.get("plates", [])):
            where = f"{name} plates[{i}]"
            text = p.get("text", "")
            if not isinstance(text, str) or not text.strip():
                errors.append(f"{where}: plate text missing/empty")
            elif text != text.upper().replace(" ", ""):
                warnings.append(f"{where}: plate {text!r} should be uppercase with no spaces")

    # ── Report ──
    print(f"\nValidating {gt_path.name} — {len(image_keys)} image(s)\n")
    for w_ in warnings:
        print(f"  [warn] {w_}")
    for e in errors:
        print(f"  [ERROR] {e}")

    if errors:
        print(f"\n✗ {len(errors)} error(s), {len(warnings)} warning(s). Fix errors before evaluating.\n")
        return 1
    print(f"\n✓ Schema valid. {len(warnings)} warning(s).")
    if not meta.get("verified"):
        print("  (Still a draft — verify the labels and set _meta.verified=true to score it.)")
    print()
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GT
    raise SystemExit(validate(target))
