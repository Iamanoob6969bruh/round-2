# Evaluation — getting REAL accuracy numbers

This produces genuine Precision / Recall / F1 / mAP and plate-match rates by
comparing the pipeline against ground truth **you verify**. Nothing is
fabricated, and the harness actively refuses to report numbers from unverified
labels (see "Verification gate" below).

## The fast workflow: verify, don't draw

Hand-drawing every box is slow. Instead, let the model propose boxes and just
**correct** them.

```bash
# from backend/
# 1. Drop traffic images into eval/images/
cp /path/to/your/*.jpg eval/images/

# 2. Pre-annotate: runs the pipeline, writes a DRAFT + visual previews
python eval/prelabel.py
#   → eval/ground_truth.draft.json   (model predictions, marked unverified)
#   → eval/preview/<image>.jpg        (boxes drawn so you can eyeball them)

# 3. VERIFY: open each eval/preview/*.jpg next to the draft JSON and:
#      • delete boxes the model hallucinated
#      • add objects / violations it missed
#      • fix wrong classes and plate text
#      • drop the "_pred_confidence" hint fields (optional)
#    Then set  "_meta": { "verified": true }  in the draft.

# 4. Sanity-check the labels (bounds, classes, violation types, image existence)
python eval/validate_labels.py eval/ground_truth.draft.json

# 5. Promote to the real file and score
mv eval/ground_truth.draft.json eval/ground_truth.json
python evaluate.py
```

Read the terminal report and the machine-readable `eval/results.json`.

## Verification gate (why your numbers are defensible)

The draft is the model grading **itself** — scoring it unchanged gives a
meaningless ~100%. So:

- `prelabel.py` marks the draft `_meta.verified = false`.
- `evaluate.py` **refuses to run** until `_meta.verified = true`.
- `--allow-unverified` overrides this **for debugging only** and prints a loud
  "NOT defensible" warning.

This means any metric you publish came from human-verified ground truth.

## Label vocabulary

Keep labels within these sets (the validator enforces it):

| Field | Allowed values |
|-------|----------------|
| detection `class` | `car`, `motorcycle`, `bus`, `truck`, `bicycle`, `auto`, `person` |
| violation `type` | `helmet_violation`, `triple_riding`, `seatbelt_violation`, `wrong_side_driving`, `stop_line_violation`, `red_light_violation`, `illegal_parking` |
| plate `text` | uppercase, no spaces, e.g. `GJ08CA5023` |

Bounding boxes are `[x1, y1, x2, y2]` in **original-image pixel** coordinates.
The previews and the harness both rescale model output back to original pixels
for you (the enhancer may upscale internally), so just match what you see in the
preview.

## Ground-truth format

```json
{
  "_meta": { "verified": true, "note": "human-reviewed" },
  "image_name.jpg": {
    "detections": [
      {"bbox": [120, 200, 360, 520], "class": "motorcycle"},
      {"bbox": [150, 80, 320, 480], "class": "person"}
    ],
    "violations": [
      {"type": "helmet_violation", "bbox": [150, 80, 360, 520]}
    ],
    "plates": [
      {"text": "GJ08CA5023"}
    ]
  }
}
```

## What you get

| Metric | Meaning |
|--------|---------|
| Detection P / R / F1 | How well vehicles/persons are found (IoU ≥ 0.5) |
| mAP@0.5, mAP@0.75 | Mean Average Precision across IoU thresholds |
| Violation P / R / F1 | How well violations are classified (IoU ≥ 0.3) |
| Plate exact / partial match | OCR accuracy vs ground-truth plate strings |
| Avg latency / FPS | Real measured throughput |

These are the exact metrics the problem statement asks for, computed honestly
on labelled data — defensible if a judge asks "how do you know?"

## How many images?

Aim for **50–100+** covering a spread of conditions: day/night, sparse/dense
traffic, clear/blurred, and at least a handful of true examples of each
violation type you claim to detect. More diverse images → more trustworthy
numbers.
