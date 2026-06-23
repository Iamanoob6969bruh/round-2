"""
Evidence Generation Module.

Produces annotated images + metadata for each detected violation.
Designed to create court-admissible evidence packages.
"""

import cv2
import numpy as np
import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from ..violations.rules_engine import Violation
from ..plate_recognition.recognizer import PlateResult
from . import integrity


@dataclass
class EvidencePackage:
    violation_id: str
    timestamp: str
    violation_type: str
    violation_description: str
    confidence: float
    severity: str
    vehicle_plate: str
    plate_confidence: float
    bbox: tuple
    image_hash: str  # full SHA-256 of the original frame (tamper-evident)
    annotated_image: Optional[np.ndarray] = None
    # Full SHA-256 binding the original frame + annotated evidence + metadata.
    # Populated by EvidenceGenerator.generate(); the DB folds it into a chain.
    content_hash: str = ""

    def core_metadata(self) -> dict:
        """The substantive, hashed fields (excludes integrity/chain fields)."""
        return {
            "violation_id": self.violation_id,
            "timestamp": self.timestamp,
            "violation_type": self.violation_type,
            "description": self.violation_description,
            "confidence": float(self.confidence),
            "severity": self.severity,
            "vehicle_plate": self.vehicle_plate,
            "plate_confidence": float(self.plate_confidence),
            "bbox": [int(x) for x in self.bbox],
            "image_hash": self.image_hash,
        }

    def to_dict(self) -> dict:
        d = self.core_metadata()
        d["content_hash"] = self.content_hash
        return d


# Color scheme for violations
VIOLATION_COLORS = {
    "helmet_violation": (0, 0, 255),       # Red
    "triple_riding": (0, 128, 255),        # Orange
    "red_light_violation": (0, 0, 200),    # Dark Red
    "stop_line_violation": (0, 165, 255),  # Orange
    "wrong_side_driving": (255, 0, 255),   # Magenta
    "illegal_parking": (255, 255, 0),      # Cyan
    "seatbelt_violation": (0, 200, 200),   # Yellow
}

SEVERITY_LABELS = {"low": "⚠️", "medium": "🔶", "high": "🔴"}


class EvidenceGenerator:
    """Generate annotated evidence for violations."""

    def __init__(self, output_dir: str = "data/evidence"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, image: np.ndarray, violations: list, plate_results: list = None) -> list:
        """
        Generate evidence packages for all violations in an image.
        Returns list of EvidencePackage.
        """
        if image is None or image.size == 0:
            return []

        timestamp = datetime.now().isoformat()
        # Full SHA-256 of the original frame (was previously truncated to 16 hex).
        image_hash = integrity.hash_bytes(image)
        annotated = image.copy()
        packages = []

        plate_map = self._map_plates_to_violations(violations, plate_results or [])

        for i, violation in enumerate(violations):
            vid = f"VIO-{image_hash[:8]}-{i:03d}"
            plate_text = plate_map.get(i, ("", 0.0))

            package = EvidencePackage(
                violation_id=vid,
                timestamp=timestamp,
                violation_type=violation.violation_type,
                violation_description=violation.description,
                confidence=round(violation.confidence, 3),
                severity=violation.severity,
                vehicle_plate=plate_text[0],
                plate_confidence=round(plate_text[1], 3),
                bbox=violation.bbox,
                image_hash=image_hash,
            )

            # Draw on annotated image
            annotated = self._draw_violation(annotated, violation, vid, plate_text[0])
            package.annotated_image = annotated.copy()

            # Tamper-evident content hash: binds the original frame, the annotated
            # evidence, and the substantive metadata into one SHA-256.
            package.content_hash = integrity.compute_content_hash(
                metadata=package.core_metadata(),
                original=image,
                annotated=package.annotated_image,
            )
            packages.append(package)

        return packages

    def generate_annotated_image(self, image: np.ndarray, violations: list, plate_results: list = None) -> np.ndarray:
        """Just produce the annotated image without full packages."""
        annotated = image.copy()
        plate_map = self._map_plates_to_violations(violations, plate_results or [])

        for i, violation in enumerate(violations):
            vid = f"V{i+1}"
            plate_text = plate_map.get(i, ("", 0.0))[0]
            annotated = self._draw_violation(annotated, violation, vid, plate_text)

        return annotated

    def save_evidence(self, packages: list, annotated_image: np.ndarray) -> str:
        """Save evidence to disk. Returns path to evidence directory."""
        if not packages:
            return ""

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        case_dir = self.output_dir / ts
        case_dir.mkdir(parents=True, exist_ok=True)

        # Save annotated image
        cv2.imwrite(str(case_dir / "annotated.jpg"), annotated_image)

        # Save metadata
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "total_violations": len(packages),
            "violations": [p.to_dict() for p in packages],
        }
        with open(case_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return str(case_dir)

    def _draw_violation(self, image: np.ndarray, violation: Violation, label: str, plate: str) -> np.ndarray:
        """Draw violation bounding box and label on image."""
        x1, y1, x2, y2 = [int(v) for v in violation.bbox]
        color = VIOLATION_COLORS.get(violation.violation_type, (0, 255, 0))

        # Draw bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Label background
        text = f"{label}: {violation.violation_type.replace('_', ' ').title()} ({violation.confidence:.0%})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw + 5, y1), color, -1)
        cv2.putText(image, text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Plate text if available
        if plate:
            plate_text = f"Plate: {plate}"
            cv2.putText(image, plate_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return image

    def _map_plates_to_violations(self, violations: list, plate_results: list) -> dict:
        """Map plate results to violations by proximity."""
        mapping = {}
        for i, violation in enumerate(violations):
            best_plate = ""
            best_conf = 0.0
            vx, vy = (violation.bbox[0] + violation.bbox[2]) / 2, (violation.bbox[1] + violation.bbox[3]) / 2

            for plate in plate_results:
                px, py = (plate.bbox[0] + plate.bbox[2]) / 2, (plate.bbox[1] + plate.bbox[3]) / 2
                dist = np.sqrt((vx - px)**2 + (vy - py)**2)
                # Associate plate with closest violation within reasonable distance
                if dist < 300 and plate.confidence > best_conf:
                    best_plate = plate.text
                    best_conf = plate.confidence

            mapping[i] = (best_plate, best_conf)
        return mapping
