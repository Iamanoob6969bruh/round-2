"""
Number Plate Recognition for FLAGGED vehicles only.

Approach:
  1. Crop the flagged vehicle bbox from image
  2. If too small (<150px) → skip
  3. Upscale to 500px wide → run EasyOCR directly (it detects text regions internally)
  4. Filter OCR results for Indian plate pattern
  5. If no result → retry on lower 50% of crop
  6. Mock registration lookup
"""

import cv2
import numpy as np
import re
from dataclasses import dataclass, field
from typing import Optional

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import pytesseract
except ImportError:
    pytesseract = None


_MOCK_STATES = {
    "MH": "Maharashtra", "DL": "Delhi", "KA": "Karnataka", "TN": "Tamil Nadu",
    "UP": "Uttar Pradesh", "GJ": "Gujarat", "RJ": "Rajasthan", "WB": "West Bengal",
    "AP": "Andhra Pradesh", "TS": "Telangana", "KL": "Kerala", "HR": "Haryana",
    "PB": "Punjab", "MP": "Madhya Pradesh", "BR": "Bihar",
}
_MOCK_OWNERS = ["Rajesh Kumar", "Priya Sharma", "Amit Patel", "Sneha Reddy",
                "Vikram Singh", "Anjali Gupta", "Suresh Nair", "Deepa Iyer"]


@dataclass
class PlateRecognition:
    violation_index: int
    plate_text: str
    confidence: float
    attempts: int
    plate_crop: Optional[np.ndarray] = None
    vehicle_crop: Optional[np.ndarray] = None
    registration: dict = field(default_factory=dict)


class FlaggedVehiclePlateReader:
    """Reads plates using EasyOCR's built-in text detection + plate pattern filtering."""

    def __init__(self):
        self.reader = None
        # Standard state-series plate: LL DD L(LL) DDDD  (e.g. MH12AB1234)
        self.plate_pattern = re.compile(r'[A-Z]{2}[\s-]?\d{1,2}[\s-]?[A-Z]{1,3}[\s-]?\d{1,4}')
        # Bharat (BH) series: DD BH DDDD L(L)  (e.g. 22BH1234AA)
        self.bh_pattern = re.compile(r'\d{2}[\s-]?BH[\s-]?\d{4}[\s-]?[A-Z]{1,2}')
        self.min_vehicle_width = 150  # px — below this, plate is unreadable
        # Load Haar cascade plate detector (for locating plates in larger crops)
        self.plate_cascade = None
        try:
            import os
            cascade_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "models", "indian_license_plate.xml"
            )
            cascade_path = os.path.abspath(cascade_path)
            if os.path.exists(cascade_path):
                cascade = cv2.CascadeClassifier(cascade_path)
                if not cascade.empty():
                    self.plate_cascade = cascade
        except Exception:
            self.plate_cascade = None

    def _init_reader(self):
        if self.reader is None:
            if easyocr is None:
                raise ImportError("easyocr required")
            try:
                import torch
                use_gpu = torch.cuda.is_available()
            except ImportError:
                use_gpu = False
            self.reader = easyocr.Reader(['en'], gpu=use_gpu)

    def read_for_violations(self, image: np.ndarray, violations: list) -> list:
        results = []
        seen_bboxes = set()

        for i, violation in enumerate(violations):
            vehicle = self._get_vehicle(violation)
            if vehicle is None:
                results.append(PlateRecognition(i, "", 0.0, 0, None, None, {}))
                continue

            key = tuple(int(v) for v in vehicle.bbox)
            if key in seen_bboxes:
                continue
            seen_bboxes.add(key)

            rec = self._read_plate(image, vehicle, i)
            results.append(rec)

        return results

    def _get_vehicle(self, violation):
        for obj in violation.involved_objects:
            if hasattr(obj, "class_name") and obj.class_name in (
                "car", "motorcycle", "bus", "truck", "auto", "bicycle"):
                return obj
        if violation.bbox and violation.bbox != (0, 0, 0, 0):
            from ..detection.detector import Detection
            return Detection(bbox=violation.bbox, class_name="unknown", confidence=violation.confidence)
        return None

    def _read_plate(self, image: np.ndarray, vehicle, violation_index: int) -> PlateRecognition:
        x1, y1, x2, y2 = [int(v) for v in vehicle.bbox]
        h_img, w_img = image.shape[:2]
        vw = x2 - x1

        # Too small — can't read plate
        if vw < self.min_vehicle_width:
            # Try with a slightly larger context crop
            pad = int(max(x2 - x1, y2 - y1) * 0.3)
            cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
            cx2, cy2 = min(w_img, x2 + pad), min(h_img, y2 + pad)
            vehicle_crop = image[cy1:cy2, cx1:cx2]
        else:
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(w_img, x2), min(h_img, y2)
            vehicle_crop = image[cy1:cy2, cx1:cx2]

        if vehicle_crop.size == 0:
            return PlateRecognition(violation_index, "", 0.0, 0, None, None, {})

        # Upscale to 500px wide for OCR
        crop_h, crop_w = vehicle_crop.shape[:2]
        if crop_w < 500:
            scale = 500 / crop_w
            ocr_img = cv2.resize(vehicle_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            ocr_img = vehicle_crop

        self._init_reader()

        # Sequential OCR strategies, tried until one yields text. We record the
        # REAL number of strategies attempted (honest telemetry — previously this
        # was hardcoded to 3 regardless of what actually ran).
        attempts_made = 0

        # Attempt 0: Haar-cascade plate detection + tesseract LSTM (best overall)
        attempts_made += 1
        text, conf, plate_crop = self._try_tesseract_ocr(vehicle_crop)

        # Attempt 1: Detect white plate region via HSV, then inverted binary OCR
        if not text:
            attempts_made += 1
            text, conf, plate_crop = self._try_plate_region_ocr(vehicle_crop)

        # Attempt 2: OCR on full vehicle crop
        if not text:
            attempts_made += 1
            text, conf, plate_crop = self._try_ocr(ocr_img)

        # Attempt 3: OCR on lower 50% only (plates are at bottom)
        if not text:
            attempts_made += 1
            lower = ocr_img[ocr_img.shape[0] // 2:, :]
            text, conf, plate_crop = self._try_ocr(lower)

        # Attempt 4: Bottom 30-40% focused crop (motorcycle plate region)
        if not text:
            attempts_made += 1
            h = ocr_img.shape[0]
            plate_region = ocr_img[int(h * 0.55):int(h * 0.85), :]
            if plate_region.size > 0:
                text, conf, plate_crop = self._try_ocr(plate_region)

        # Attempt 5: Adaptive threshold on plate region for low-contrast plates
        if not text:
            attempts_made += 1
            h = ocr_img.shape[0]
            region = ocr_img[int(h * 0.45):int(h * 0.9), :]
            if region.size > 0:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
                gray = cv2.bilateralFilter(gray, 9, 75, 75)
                adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
                text, conf, plate_crop = self._try_ocr(adapt)

        # Attempt 6: CLAHE enhanced full image
        if not text:
            attempts_made += 1
            gray = cv2.cvtColor(ocr_img, cv2.COLOR_BGR2GRAY) if len(ocr_img.shape) == 3 else ocr_img
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            text, conf, plate_crop = self._try_ocr(enhanced)

        registration = self._mock_registration(text) if text else {}

        return PlateRecognition(
            violation_index=violation_index,
            plate_text=text,
            confidence=round(float(conf), 3),
            attempts=attempts_made,
            plate_crop=plate_crop if plate_crop is not None else vehicle_crop,
            vehicle_crop=vehicle_crop,
            registration=registration,
        )

    def _try_tesseract_ocr(self, img: np.ndarray) -> tuple:
        """Detect plate regions (Haar cascade) then OCR each with tesseract LSTM.

        Handles both full scenes (locates plate within) and pre-cropped plates.
        Returns (plate_text, confidence, plate_crop).
        """
        if pytesseract is None:
            return ("", 0.0, None)

        h, w = img.shape[:2]
        best = ("", 0.0, None)
        matches = []  # (plate_text, conf, crop) for full-pattern hits

        # 1. Use Haar cascade to find candidate plate regions in the image
        candidates = self._detect_plate_boxes(img)

        # 2. OCR each candidate with several expansion factors (haar boxes crop tight)
        for (x, y, pw, ph) in candidates:
            for exf in (0.10, 0.18, 0.28, 0.0):
                ex, ey = int(pw * exf), int(ph * exf * 0.6)
                x1, y1 = max(0, x - ex), max(0, y - ey)
                x2, y2 = min(w, x + pw + ex), min(h, y + ph + ey)
                crop = img[y1:y2, x1:x2]
                text, conf = self._ocr_plate_crop(crop)
                if not text:
                    continue
                if self._is_valid_plate(text.replace(" ", "")):
                    matches.append((text, conf, crop))
                elif conf > best[1]:
                    best = (text, conf, crop)
            # If we already have a full match for this box, prefer the most complete one
            if matches:
                # pick the longest plate string (captures full 4-digit tail)
                matches.sort(key=lambda m: len(m[0].replace(" ", "")), reverse=True)
                return matches[0]

        # 3. Fallback: treat the whole image as a plate crop (pre-cropped input)
        text, conf = self._ocr_plate_crop(img)
        if text and self._is_valid_plate(text.replace(" ", "")):
            return (text, conf, img)
        if conf > best[1]:
            best = (text, conf, img)

        return best

    def _detect_plate_boxes(self, img: np.ndarray) -> list:
        """Detect candidate license-plate boxes. Returns list of (x, y, w, h)."""
        if self.plate_cascade is None:
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        try:
            boxes = self.plate_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 15)
            )
        except Exception:
            return []
        boxes = [tuple(int(v) for v in b) for b in boxes]
        # Sort by area (largest first) and de-duplicate overlapping boxes
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        kept = []
        for (x, y, bw, bh) in boxes:
            overlap = False
            for (kx, ky, kw, kh) in kept:
                if not (x + bw < kx or x > kx + kw or y + bh < ky or y > ky + kh):
                    overlap = True
                    break
            if not overlap:
                kept.append((x, y, bw, bh))
        return kept[:6]

    def _ocr_plate_crop(self, crop: np.ndarray) -> tuple:
        """Denoise → upscale → CLAHE → tesseract (multi-PSM). Returns (text, conf)."""
        if crop is None or crop.size == 0 or pytesseract is None:
            return ("", 0.0)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        cw = gray.shape[1]
        denoised = cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7, searchWindowSize=21)
        if cw < 400:
            scale = 400 / cw
            denoised = cv2.resize(denoised, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(denoised)

        best_text, best_conf = "", 0.0
        for psm in (7, 8, 11, 6):
            cfg = f'--oem 1 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            try:
                raw = pytesseract.image_to_string(enhanced, config=cfg).strip()
            except Exception:
                continue
            cleaned = self._clean(raw)
            if not cleaned or len(cleaned) < 5:
                continue
            corrected = self._correct_plate_ocr(cleaned.replace(" ", ""))
            extracted = self._extract_plate(corrected)
            if extracted:
                return (extracted, 0.85)  # strong match
            if len(corrected) > len(best_text):
                best_text, best_conf = corrected, 0.55
        return (best_text, best_conf)

    def _try_plate_region_ocr(self, vehicle_crop: np.ndarray) -> tuple:
        """Detect white plate region via HSV, upscale heavily, use inverted binary for OCR."""
        h, w = vehicle_crop.shape[:2]
        hsv = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2HSV)
        # White/light plate: low saturation, high value
        mask = cv2.inRange(hsv, (0, 0, 140), (180, 70, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find plate-shaped white region in lower half
        candidates = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if ch < 8 or cw < 15:
                continue
            ar = cw / max(ch, 1)
            area = cw * ch
            if 0.8 <= ar <= 5.0 and y > h * 0.3 and area > 300:
                candidates.append((x, y, cw, ch, area))

        if not candidates:
            return ("", 0.0, None)

        # Try top candidates by area
        candidates.sort(key=lambda c: c[4], reverse=True)

        for x, y, cw, ch, _ in candidates[:3]:
            pad = 5
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w, x + cw + pad), min(h, y + ch + pad)
            crop = vehicle_crop[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Upscale to at least 400px wide
            crop_w = crop.shape[1]
            scale = max(400 / crop_w, 4.0)
            big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
            denoised = cv2.bilateralFilter(gray, 11, 75, 75)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            cl = clahe.apply(denoised)
            _, binary = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            inverted = cv2.bitwise_not(binary)

            text, conf, plate_crop = self._try_ocr(inverted)
            if text:
                return (text, conf, plate_crop if plate_crop is not None else crop)

            # Also try non-inverted
            text, conf, plate_crop = self._try_ocr(binary)
            if text:
                return (text, conf, plate_crop if plate_crop is not None else crop)

        return ("", 0.0, None)

    def _try_ocr(self, img: np.ndarray) -> tuple:
        """Run EasyOCR, combine nearby text boxes for multi-line plates. Returns (text, conf, crop)."""
        try:
            results = self.reader.readtext(
                img,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                paragraph=False,
                min_size=10,
            )
        except Exception:
            return ("", 0.0, None)

        if not results:
            return ("", 0.0, None)

        # Strategy 1: Combine all nearby text boxes that could form a plate
        combined_text, combined_conf, combined_crop = self._combine_plate_lines(results, img)
        if combined_text:
            return (combined_text, combined_conf, combined_crop)

        # Strategy 2: Single best result (fallback)
        best_text = ""
        best_conf = 0.0
        best_score = 0.0
        best_crop = None

        for (bbox_pts, text, conf) in results:
            cleaned = self._clean(text)
            if len(cleaned) < 4:
                continue

            is_plate = self._is_valid_plate(self._correct_plate_ocr(cleaned.replace(" ", "")))
            score = conf + (0.5 if is_plate else 0) + min(len(cleaned) / 12, 0.2)

            if score > best_score:
                best_score = score
                best_conf = conf
                best_text = cleaned
                pts = np.array(bbox_pts).astype(int)
                px1, py1 = pts.min(axis=0)
                px2, py2 = pts.max(axis=0)
                h, w = img.shape[:2] if len(img.shape) == 3 else (img.shape[0], img.shape[1])
                px1, py1 = max(0, px1 - 5), max(0, py1 - 5)
                px2, py2 = min(w, px2 + 5), min(h, py2 + 5)
                best_crop = img[py1:py2, px1:px2] if len(img.shape) == 3 else cv2.cvtColor(img[py1:py2, px1:px2], cv2.COLOR_GRAY2BGR)

        return (best_text, best_conf, best_crop)

    def _combine_plate_lines(self, results, img) -> tuple:
        """Combine vertically adjacent text boxes into one plate string (for two-line plates)."""
        if len(results) < 2:
            return ("", 0.0, None)

        # Get bounding boxes with their text
        entries = []
        for (bbox_pts, text, conf) in results:
            cleaned = self._clean(text)
            if len(cleaned) < 2:
                continue
            pts = np.array(bbox_pts).astype(int)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            entries.append({"text": cleaned, "conf": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "cx": cx, "cy": cy, "h": y2 - y1})

        if len(entries) < 2:
            return ("", 0.0, None)

        # Try all pairs of text boxes — combine if vertically stacked and horizontally aligned
        best_text = ""
        best_conf = 0.0
        best_crop = None

        for i in range(len(entries)):
            for j in range(len(entries)):
                if i == j:
                    continue
                a, b = entries[i], entries[j]
                # a should be above b
                if a["cy"] >= b["cy"]:
                    continue

                # Check vertical proximity (gap < 1.5x line height)
                gap = b["y1"] - a["y2"]
                line_h = max(a["h"], b["h"])
                if gap > line_h * 1.5 or gap < -line_h * 0.5:
                    continue

                # Check horizontal overlap (centers within reasonable range)
                overlap_x = min(a["x2"], b["x2"]) - max(a["x1"], b["x1"])
                min_w = min(a["x2"] - a["x1"], b["x2"] - b["x1"])
                if overlap_x < min_w * 0.3:
                    continue

                # Combine
                merged = a["text"] + b["text"]
                merged_nospace = merged.replace(" ", "")
                avg_conf = (a["conf"] + b["conf"]) / 2

                # Apply Indian plate OCR corrections on merged text
                corrected = self._correct_plate_ocr(merged_nospace)
                is_plate = self._is_valid_plate(corrected)
                if is_plate or len(merged_nospace) >= 8:
                    score = avg_conf + (0.5 if is_plate else 0)
                    if score > best_conf:
                        best_conf = avg_conf
                        best_text = corrected
                        extracted = self._extract_plate(corrected)
                        if extracted:
                            best_text = extracted
                        # Crop encompassing both boxes
                        px1 = max(0, min(a["x1"], b["x1"]) - 5)
                        py1 = max(0, min(a["y1"], b["y1"]) - 5)
                        h, w = img.shape[:2]
                        px2 = min(w, max(a["x2"], b["x2"]) + 5)
                        py2 = min(h, max(a["y2"], b["y2"]) + 5)
                        best_crop = img[py1:py2, px1:px2] if len(img.shape) == 3 else cv2.cvtColor(img[py1:py2, px1:px2], cv2.COLOR_GRAY2BGR)

        return (best_text, best_conf, best_crop)

    @staticmethod
    def _digits(s: str) -> str:
        """Coerce common OCR letter→digit confusions (for numeric fields)."""
        return (s.replace('O', '0').replace('Q', '0').replace('D', '0')
                 .replace('I', '1').replace('L', '1').replace('S', '5')
                 .replace('B', '8').replace('Z', '2').replace('G', '6'))

    @staticmethod
    def _letters(s: str) -> str:
        """Coerce common OCR digit→letter confusions (for alphabetic fields)."""
        return (s.replace('0', 'O').replace('1', 'I').replace('5', 'S')
                 .replace('8', 'B').replace('2', 'Z').replace('6', 'G'))

    def _is_valid_plate(self, text: str) -> bool:
        """True if text matches a recognised format (standard state-series or BH)."""
        t = text.upper()
        return bool(self.plate_pattern.search(t) or self.bh_pattern.search(t))

    def _extract_plate(self, text: str) -> str:
        """Return the matched plate substring (BH preferred, then standard); else ''."""
        t = text.upper()
        m = self.bh_pattern.search(t)
        if m:
            return m.group().replace(' ', '').replace('-', '')
        m = self.plate_pattern.search(t)
        if m:
            return m.group().replace(' ', '').replace('-', '')
        return ""

    def _correct_bh_series(self, t: str) -> str:
        """Correct toward Bharat-series DD BH DDDD L(L); '' if it doesn't fit."""
        if len(t) not in (9, 10):
            return ""
        mid = t[2:4].replace('8', 'B').replace('0', 'O')
        if not (mid[0] == 'B' and mid[1] == 'H'):
            return ""
        year = self._digits(t[:2])
        num = self._digits(t[4:8])
        series = self._letters(t[8:])
        if not (year.isdigit() and len(year) == 2 and num.isdigit() and len(num) == 4):
            return ""
        candidate = f"{year}BH{num}{series}"
        return candidate if self.bh_pattern.fullmatch(candidate) else ""

    def _correct_standard(self, t: str) -> str:
        """Correct toward standard state-series LL DD L(LL) DDDD; '' if it doesn't fit."""
        if len(t) < 9:
            return ""
        # Drop a spurious digit misread between state code and district code
        if len(t) > 2 and t[1].isdigit() and t[2].isalpha():
            t = t[0] + t[2:]
        head = self._letters(t[:2])              # state code → letters
        dist = self._digits(t[2:4])              # district code → digits
        tail = t[4:]
        letter_part, digit_part = "", ""
        for i, ch in enumerate(tail):
            if ch.isdigit() and i > 0:
                digit_part = tail[i:]
                break
            letter_part += ch
        letter_part = self._letters(letter_part)  # series letters
        digit_part = self._digits(digit_part)      # 1-4 digit number
        candidate = head + dist + letter_part + digit_part
        return candidate if self.plate_pattern.fullmatch(candidate) else ""

    def _correct_plate_ocr(self, text: str) -> str:
        """Positional OCR digit↔letter correction.

        Tries the Bharat (BH) series first, then the standard state-series. If
        the text matches NEITHER recognised structure it is returned UNCHANGED —
        we never force the standard ``LL DD L DDDD`` template onto temporary,
        VIP, diplomatic or otherwise non-standard plates (doing so previously
        corrupted them silently).
        """
        t = text.upper().replace(' ', '').replace('-', '')
        if len(t) < 6:
            return t
        bh = self._correct_bh_series(t)
        if bh:
            return bh
        std = self._correct_standard(t)
        if std:
            return std
        return t  # unknown / non-standard format — do not mangle

    def _clean(self, text: str) -> str:
        """Normalise OCR output (uppercase, strip punctuation/whitespace).

        Deliberately NON-destructive: it does not apply format-specific digit↔
        letter swaps. Structural correction is the job of _correct_plate_ocr so
        that non-standard plates survive cleaning intact.
        """
        cleaned = re.sub(r'[^A-Za-z0-9\s]', '', text.upper())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def _mock_registration(self, plate: str) -> dict:
        plate_compact = plate.replace(" ", "")
        state_code = plate_compact[:2] if len(plate_compact) >= 2 else "XX"
        state = _MOCK_STATES.get(state_code, "Unknown State")
        owner = _MOCK_OWNERS[hash(plate_compact) % len(_MOCK_OWNERS)]
        return {
            "plate": plate, "state": state, "registered_owner": owner,
            "vehicle_class": "Private", "status": "Active",
            "note": "MOCK DATA — no real registry connected",
        }
