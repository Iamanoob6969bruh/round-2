"""
License Plate Recognition: Detection + Enhancement + OCR.

Pipeline: Detect plate region → crop → enhance for OCR → extract text.
"""

import cv2
import numpy as np
import re
from dataclasses import dataclass
from typing import Optional

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import pytesseract
except ImportError:
    pytesseract = None


@dataclass
class PlateResult:
    bbox: tuple  # plate bounding box in original image
    text: str
    confidence: float
    enhanced_crop: Optional[np.ndarray] = None


class PlateDetector:
    """Detect license plate regions using contour analysis + morphology."""

    def detect_plates(self, image: np.ndarray) -> list:
        """Find candidate plate regions. Returns list of (x1, y1, x2, y2) bboxes."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h_img, w_img = image.shape[:2]

        # Method 1: Edge + contour based
        plates = self._detect_by_contour(gray, h_img, w_img)

        # Method 2: Morphological approach (better for low quality)
        plates.extend(self._detect_by_morphology(gray, h_img, w_img))

        # Remove duplicates
        plates = self._nms(plates)
        return plates[:5]

    def _detect_by_contour(self, gray, h_img, w_img):
        plates = []
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(blurred, 30, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:50]:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if 4 <= len(approx) <= 6:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / max(h, 1)
                if 1.5 <= aspect_ratio <= 6.0 and w > 40 and h > 10:
                    area_ratio = (w * h) / (w_img * h_img)
                    if 0.001 < area_ratio < 0.15:
                        plates.append((x, y, x + w, y + h))
        return plates

    def _detect_by_morphology(self, gray, h_img, w_img):
        """Use morphology to find bright rectangular plate-like regions."""
        plates = []

        # Plates are usually brighter than surroundings
        # Apply blackhat to find bright regions on dark background
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        # Threshold
        _, thresh = cv2.threshold(tophat, 80, 255, cv2.THRESH_BINARY)

        # Dilate to connect characters
        kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        dilated = cv2.dilate(thresh, kernel_d, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / max(h, 1)
            if 1.5 <= aspect_ratio <= 6.0 and w > 40 and h > 10:
                area_ratio = (w * h) / (w_img * h_img)
                if 0.001 < area_ratio < 0.1:
                    plates.append((x, y, x + w, y + h))
        return plates

    def _nms(self, boxes: list, overlap_thresh: float = 0.5) -> list:
        if not boxes:
            return []
        boxes = np.array(boxes)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = areas.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= overlap_thresh)[0]
            order = order[inds + 1]

        return [tuple(boxes[i]) for i in keep]


class PlateOCR:
    """OCR for license plate text extraction."""

    def __init__(self):
        self.reader = None
        # Indian plate pattern: XX-00-XX-0000 or variations
        self.plate_pattern = re.compile(r'[A-Z]{2}\s*\d{1,2}\s*[A-Z]{1,3}\s*\d{1,4}')

    def _init_reader(self):
        if self.reader is None:
            if easyocr is None:
                raise ImportError("easyocr package required. Install via: pip install easyocr")
            try:
                import torch
                use_gpu = torch.cuda.is_available()
            except ImportError:
                use_gpu = False
            self.reader = easyocr.Reader(['en'], gpu=use_gpu)

    def read_plate(self, plate_crop: np.ndarray) -> tuple:
        """Extract text from plate crop. Returns (text, confidence)."""
        if plate_crop is None or plate_crop.size == 0:
            return ("", 0.0)

        # Preprocess: denoise at original res, upscale, CLAHE
        processed = self._preprocess_for_ocr(plate_crop)

        # Try pytesseract first (much better for noisy plates)
        if pytesseract is not None:
            text, conf = self._try_tesseract(processed)
            if text:
                return (text, conf)

        # Fallback to easyocr with multiple variants
        self._init_reader()
        attempts = self._get_ocr_variants(plate_crop)

        best_text = ""
        best_conf = 0.0

        for img in attempts:
            try:
                results = self.reader.readtext(img, paragraph=False)
            except Exception:
                continue
            if not results:
                continue
            full_text = " ".join([r[1] for r in results])
            avg_conf = float(np.mean([r[2] for r in results]))
            cleaned = self._clean_plate_text(full_text)
            if cleaned and avg_conf > best_conf:
                best_text = cleaned
                best_conf = avg_conf

        return (best_text, best_conf)

    def _try_tesseract(self, img: np.ndarray) -> tuple:
        """Run tesseract LSTM on preprocessed plate image."""
        cfg = '--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        try:
            text = pytesseract.image_to_string(img, config=cfg).strip()
        except Exception:
            return ("", 0.0)
        cleaned = self._clean_plate_text(text)
        if cleaned:
            return (cleaned, 0.85)
        # Try PSM 8 (single word) as fallback
        try:
            text = pytesseract.image_to_string(img, config=cfg.replace('--psm 7', '--psm 8')).strip()
        except Exception:
            return ("", 0.0)
        cleaned = self._clean_plate_text(text)
        return (cleaned, 0.80) if cleaned else ("", 0.0)

    def _get_ocr_variants(self, crop: np.ndarray) -> list:
        """Generate multiple preprocessed versions for OCR — more chances to read."""
        variants = []
        h, w = crop.shape[:2]

        # Upscale small plates
        if w < 200:
            scale = 200 / w
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Variant 1: Original (often good enough for clear plates)
        variants.append(crop)

        # Variant 2: Grayscale + CLAHE (handles uneven lighting)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        variants.append(enhanced_gray)

        # Variant 3: Bilateral filter + Otsu (clean binarization)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)

        # Variant 4: Inverted (white text on dark plate)
        variants.append(cv2.bitwise_not(otsu))

        # Variant 5: Sharpened color
        gaussian = cv2.GaussianBlur(crop, (0, 0), 2.0)
        sharpened = cv2.addWeighted(crop, 1.8, gaussian, -0.8, 0)
        variants.append(sharpened)

        return variants

    def _preprocess_for_ocr(self, crop: np.ndarray) -> np.ndarray:
        """Denoise → trim borders → upscale → CLAHE. Optimized for tesseract."""
        h, w = crop.shape[:2] if len(crop.shape) == 2 else crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

        # Only trim borders on raw crops (not already enhanced images)
        cur_h, cur_w = gray.shape[:2]
        if cur_w < 350:
            border_y, border_x = max(2, cur_h // 8), max(2, cur_w // 20)
            gray = gray[border_y:cur_h - border_y, border_x:cur_w - border_x]

        # Denoise at original resolution (noise is small-scale, easier to remove here)
        gray = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)

        # Upscale for OCR
        cur_h, cur_w = gray.shape[:2]
        if cur_w < 400:
            scale = 400 / cur_w
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Gentle CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        return gray

    def _clean_plate_text(self, text: str) -> str:
        """Clean and format plate text. Keep result even if not perfect Indian format."""
        # Remove special chars, keep alphanumeric and spaces
        cleaned = re.sub(r'[^A-Za-z0-9\s]', '', text.upper())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Try to match Indian plate format first (without corrections)
        match = self.plate_pattern.search(cleaned)
        if match:
            return match.group()

        # Apply OCR corrections only in expected positions for Indian plates:
        # Format: XX 00 XX 0000 (letters-digits-letters-digits)
        # Only correct digits in letter positions and letters in digit positions
        corrected = cleaned.replace(' ', '')
        if len(corrected) >= 6:
            # First 2 chars should be letters
            head = corrected[:2].replace('0','O').replace('1','I').replace('5','S').replace('8','B').replace('2','Z')
            # Rest: try matching as-is
            tail = corrected[2:]
            corrected = head + tail
            match = self.plate_pattern.search(corrected)
            if match:
                return match.group()

        # Return cleaned text even without strict format match (partial plates are useful)
        return cleaned if len(cleaned) >= 4 else ""


class LicensePlateRecognizer:
    """Complete license plate pipeline: detect → enhance → OCR."""

    def __init__(self, enhancer=None):
        self.detector = PlateDetector()
        self.ocr = PlateOCR()
        self.enhancer = enhancer  # AdaptivePreprocessor instance

    def recognize(self, image: np.ndarray, vehicle_bbox: tuple = None) -> list:
        """
        Recognize plates in image or within a vehicle bounding box.
        Returns list of PlateResult.
        """
        # If vehicle bbox given, search only within that region
        if vehicle_bbox:
            x1, y1, x2, y2 = [int(v) for v in vehicle_bbox]
            search_region = image[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            search_region = image
            offset = (0, 0)

        if search_region.size == 0:
            return []

        # Detect plate candidates
        plate_bboxes = self.detector.detect_plates(search_region)
        results = []

        for bbox in plate_bboxes:
            bx1, by1, bx2, by2 = bbox
            crop = search_region[by1:by2, bx1:bx2]

            if crop.size == 0:
                continue

            # Enhance crop for OCR
            if self.enhancer:
                enhanced = self.enhancer.enhance_plate_crop(crop)
            else:
                enhanced = crop

            # Run OCR
            text, confidence = self.ocr.read_plate(enhanced)

            if text:
                # Convert bbox back to full image coordinates
                abs_bbox = (
                    bx1 + offset[0], by1 + offset[1],
                    bx2 + offset[0], by2 + offset[1]
                )
                results.append(PlateResult(
                    bbox=abs_bbox,
                    text=text,
                    confidence=confidence,
                    enhanced_crop=enhanced,
                ))

        return results
