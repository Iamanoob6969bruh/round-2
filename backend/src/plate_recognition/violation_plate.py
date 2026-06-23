"""
Violation → Plate Zoom Pipeline.

For every detected violation, crop the violating vehicle region,
apply aggressive enhancement, and extract the license plate.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..preprocessing.enhancer import AdaptivePreprocessor
from ..plate_recognition.recognizer import PlateDetector, PlateOCR


@dataclass
class ViolationPlate:
    violation_index: int
    vehicle_crop: np.ndarray
    enhanced_crop: np.ndarray
    plate_crop: Optional[np.ndarray]
    plate_text: str
    plate_confidence: float


class ViolationPlateExtractor:
    """Crop violating vehicle → super-enhance → extract plate."""

    def __init__(self):
        self.preprocessor = AdaptivePreprocessor()
        self.plate_detector = PlateDetector()
        self.ocr = PlateOCR()

    def extract_plates_for_violations(self, image: np.ndarray, violations: list) -> list:
        """For each violation, crop the vehicle and try to read its plate."""
        results = []
        h, w = image.shape[:2]

        for i, violation in enumerate(violations):
            # Get violation bbox — expand it to capture full vehicle + plate area
            vbbox = violation.bbox
            x1, y1, x2, y2 = [int(v) for v in vbbox]

            # Expand bbox downward (plates are usually at bottom of vehicle)
            pad_x = int((x2 - x1) * 0.1)
            pad_y_top = int((y2 - y1) * 0.05)
            pad_y_bottom = int((y2 - y1) * 0.3)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y_top)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y_bottom)

            vehicle_crop = image[y1:y2, x1:x2]
            if vehicle_crop.size == 0:
                continue

            # Aggressively enhance this crop
            enhanced_crop = self.preprocessor.enhance_plate_crop(vehicle_crop)

            # Detect plate in enhanced crop
            plate_bboxes = self.plate_detector.detect_plates(enhanced_crop)

            plate_text = ""
            plate_conf = 0.0
            plate_crop = None

            if plate_bboxes:
                # Try each candidate, keep best OCR result
                for pbbox in plate_bboxes:
                    px1, py1, px2, py2 = [int(v) for v in pbbox]
                    pcrop = enhanced_crop[py1:py2, px1:px2]
                    if pcrop.size == 0:
                        continue

                    # Further enhance plate region
                    pcrop_enhanced = self.preprocessor.enhance_plate_crop(pcrop)
                    text, conf = self.ocr.read_plate(pcrop_enhanced)

                    if conf > plate_conf:
                        plate_text = text
                        plate_conf = conf
                        plate_crop = pcrop_enhanced
            else:
                # No plate region found — try OCR on full enhanced vehicle crop
                text, conf = self.ocr.read_plate(enhanced_crop)
                if text:
                    plate_text = text
                    plate_conf = conf

            results.append(ViolationPlate(
                violation_index=i,
                vehicle_crop=vehicle_crop,
                enhanced_crop=enhanced_crop,
                plate_crop=plate_crop,
                plate_text=plate_text,
                plate_confidence=plate_conf,
            ))

        return results
