"""
Evidence Defensibility Score (EDS).

The core innovation: every violation is scored 0-100 on how *defensible* its
evidence is — i.e. how likely it would hold up under human / legal review.
This converts the system from a blunt "accuser" into a calibrated
decision-support tool that knows when NOT to trust itself.

EDS fuses four independent signals (all already produced by the pipeline):
  1. Violation confidence  — how sure the rule engine is of the violation
  2. Image quality         — photometric quality of the source frame
  3. Object clarity        — how large / unoccluded the violating object is
  4. Identity confidence   — OCR confidence on the number plate

Each case is then ROUTED:
  EDS >= 75            -> AUTO-ISSUE      (strong, defensible evidence)
  45 <= EDS < 75       -> HUMAN REVIEW    (ambiguous, needs an officer)
  EDS < 45             -> DISCARD         (insufficient evidence)

A separate RISK score (severity x violation-type danger) drives triage
ordering so officer attention goes to the most dangerous cases first.
"""

from dataclasses import dataclass, field


# Relative weights of the four EDS factors (sum = 1.0)
WEIGHTS = {
    "violation_confidence": 0.40,
    "image_quality": 0.25,
    "object_clarity": 0.15,
    "identity_confidence": 0.20,
}

# Routing thresholds on the 0-100 EDS scale
AUTO_ISSUE_THRESHOLD = 75.0
REVIEW_THRESHOLD = 45.0

# Inherent danger of each violation type (0-1) — drives the risk score
TYPE_RISK = {
    "helmet_violation": 0.90,
    "triple_riding": 0.85,
    "red_light_violation": 1.00,
    "wrong_side_driving": 0.95,
    "stop_line_violation": 0.55,
    "seatbelt_violation": 0.70,
    "illegal_parking": 0.30,
}

SEVERITY_WEIGHT = {"high": 1.0, "medium": 0.6, "low": 0.3}


# Transparent, auditable description of the EDS model itself. Surfaced in the UI
# so the score is never a "black box" — and so we are honest that the underlying
# confidences are raw model outputs used conservatively (EDS is advisory, with a
# human-review band precisely for the cases calibration is least certain about).
EDS_METHODOLOGY = {
    "model": "Transparent weighted linear model (glass-box, not a black box).",
    "weights": {
        "Violation confidence": "40%",
        "Image quality": "25%",
        "Object clarity": "15%",
        "Identity (plate) confidence": "20%",
    },
    "calibration_note": (
        "Inputs are raw detector/OCR confidences, used conservatively. The "
        "45–75 'human-review' band deliberately captures the cases where model "
        "confidence is least reliable, so uncertainty defers to a human rather "
        "than to an automated citation. Every factor is inspectable — no opaque scoring."
    ),
}


@dataclass
class DefensibilityResult:
    score: float                       # 0-100 EDS
    routing: str                       # "auto-issue" | "human-review" | "discard"
    factors: dict = field(default_factory=dict)   # per-factor 0-100 contributions
    risk: float = 0.0                  # 0-100 risk score for triage
    rationale: str = ""                # one-line summary of the routing decision


def _image_quality_score(quality_metrics: dict) -> float:
    """Derive a 0-1 image-quality score from the enhancer's final metrics.

    quality_metrics is the trace.final dict:
      {brightness, contrast, sharpness, overexposure, underexposure, ...}
    Closer to the ideal photometric profile => higher score.
    """
    if not quality_metrics:
        return 0.6  # neutral prior when unknown

    b = quality_metrics.get("brightness", 120)
    c = quality_metrics.get("contrast", 60)
    s = quality_metrics.get("sharpness", 300)
    over = quality_metrics.get("overexposure", 0.0)
    under = quality_metrics.get("underexposure", 0.0)

    # Brightness: ideal 120, full credit within +/-40
    q_b = max(0.0, 1.0 - abs(b - 120) / 90.0)
    # Contrast: ideal >= 50
    q_c = min(c / 55.0, 1.0)
    # Sharpness: ideal >= 250 (saturating)
    q_s = min(s / 250.0, 1.0)
    # Exposure penalties: blown/crushed pixels reduce defensibility
    q_e = max(0.0, 1.0 - (over * 4.0) - (under * 3.0))

    return max(0.0, min(0.30 * q_b + 0.25 * q_c + 0.30 * q_s + 0.15 * q_e, 1.0))


def _object_clarity_score(bbox, image_w, image_h) -> float:
    """How large/clear is the violating object relative to the frame.

    Tiny, distant objects produce weak evidence; large, near objects are
    defensible. Uses sqrt of area fraction so it saturates gracefully.
    """
    if not bbox or image_w <= 0 or image_h <= 0:
        return 0.5
    x1, y1, x2, y2 = bbox
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area_frac = (w * h) / float(image_w * image_h)
    # sqrt scaling: 1% of frame -> 0.32, 4% -> 0.63, 10% -> 1.0 (capped)
    clarity = min((area_frac / 0.10) ** 0.5, 1.0)
    return max(0.15, clarity)  # floor so a small-but-valid object isn't zeroed


def compute_defensibility(
    violation_confidence: float,
    severity: str,
    violation_type: str,
    bbox,
    image_w: int,
    image_h: int,
    quality_metrics: dict,
    plate_confidence: float = 0.0,
    has_plate: bool = False,
) -> DefensibilityResult:
    """Compute the Evidence Defensibility Score and routing for one violation."""

    # ── Factor 1: violation confidence (0-1) ──
    f_conf = max(0.0, min(violation_confidence, 1.0))

    # ── Factor 2: image quality (0-1) ──
    f_img = _image_quality_score(quality_metrics)

    # ── Factor 3: object clarity (0-1) ──
    f_clarity = _object_clarity_score(bbox, image_w, image_h)

    # ── Factor 4: identity confidence (0-1) ──
    # If a plate was read, use its OCR confidence; if not, identity is weak but
    # the violation itself can still be valid — use a reduced neutral prior so
    # missing identity lowers (but does not destroy) defensibility.
    f_identity = max(0.0, min(plate_confidence, 1.0)) if has_plate else 0.35

    # ── Weighted fusion -> 0-100 ──
    score01 = (
        WEIGHTS["violation_confidence"] * f_conf
        + WEIGHTS["image_quality"] * f_img
        + WEIGHTS["object_clarity"] * f_clarity
        + WEIGHTS["identity_confidence"] * f_identity
    )
    score = round(score01 * 100, 1)

    # ── Routing decision ──
    if score >= AUTO_ISSUE_THRESHOLD:
        routing = "auto-issue"
        rationale = "Strong, defensible evidence — eligible for automated citation."
    elif score >= REVIEW_THRESHOLD:
        routing = "human-review"
        rationale = "Ambiguous evidence — flagged for officer verification before action."
    else:
        routing = "discard"
        rationale = "Insufficient evidence quality — not actionable without better capture."

    # ── Risk score (independent of EDS): danger of the act, for triage order ──
    type_risk = TYPE_RISK.get(violation_type, 0.5)
    sev = SEVERITY_WEIGHT.get(severity, 0.6)
    risk = round((0.6 * type_risk + 0.4 * sev) * 100, 1)

    return DefensibilityResult(
        score=score,
        routing=routing,
        factors={
            "Violation confidence": round(f_conf * 100, 1),
            "Image quality": round(f_img * 100, 1),
            "Object clarity": round(f_clarity * 100, 1),
            "Identity (plate) confidence": round(f_identity * 100, 1),
        },
        risk=risk,
        rationale=rationale,
    )


def triage_summary(results: list) -> dict:
    """Aggregate routing decisions across all violations for the triage panel."""
    counts = {"auto-issue": 0, "human-review": 0, "discard": 0}
    for r in results:
        counts[r.routing] = counts.get(r.routing, 0) + 1
    total = max(len(results), 1)
    review_or_issue = counts["auto-issue"] + counts["human-review"]
    return {
        "counts": counts,
        "total": len(results),
        "auto_issue_pct": round(counts["auto-issue"] / total * 100, 1),
        "human_review_pct": round(counts["human-review"] / total * 100, 1),
        "discard_pct": round(counts["discard"] / total * 100, 1),
        # Headline scalability stat: fraction needing a human
        "human_effort_pct": round(counts["human-review"] / max(review_or_issue, 1) * 100, 1),
    }
