"""
Vehicle and Road User Detection using YOLOv8.

Detects: cars, motorcycles, buses, trucks, autos, persons, bicycles.
Also uses a dedicated helmet detection model for helmet violations.
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass
class Detection:
    bbox: tuple  # (x1, y1, x2, y2)
    class_name: str
    confidence: float
    class_id: int = 0
    # True when class_name was assigned by a heuristic rather than the detector
    # itself (e.g. auto-rickshaw inferred from bounding-box size). Downstream
    # scoring treats these as lower-trust.
    inferred: bool = False

    @property
    def center(self) -> tuple:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class DetectionResult:
    vehicles: list = field(default_factory=list)
    persons: list = field(default_factory=list)
    helmets: list = field(default_factory=list)       # "With helmet"
    no_helmets: list = field(default_factory=list)    # "Without helmet"
    traffic_lights: list = field(default_factory=list)
    all_detections: list = field(default_factory=list)


# COCO class mapping relevant to traffic
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle"}
PERSON_CLASS = 0
TRAFFIC_LIGHT_CLASS = 9


class VehicleDetector:
    """YOLO-based vehicle and road user detection."""

    def __init__(self, model_path: str = "yolov8n.pt", helmet_model_path: str = None, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        if YOLO is None:
            raise ImportError("ultralytics package required. Install via: pip install ultralytics")
        self.model = YOLO(model_path)
        self.helmet_model = None
        if helmet_model_path and Path(helmet_model_path).exists():
            self.helmet_model = YOLO(helmet_model_path)

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Run detection on an image."""
        results = self.model(image, conf=self.confidence_threshold, verbose=False, max_det=300, imgsz=1280)
        result = DetectionResult()

        if not results or len(results) == 0:
            return result

        boxes = results[0].boxes
        if boxes is None:
            return result

        for i in range(len(boxes)):
            bbox = tuple(int(v) for v in boxes.xyxy[i].cpu().numpy())
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())

            if cls_id in VEHICLE_CLASSES:
                det = Detection(bbox=bbox, class_name=VEHICLE_CLASSES[cls_id], confidence=conf, class_id=cls_id)
                result.vehicles.append(det)
                result.all_detections.append(det)

            elif cls_id == PERSON_CLASS:
                det = Detection(bbox=bbox, class_name="person", confidence=conf, class_id=cls_id)
                result.persons.append(det)
                result.all_detections.append(det)

            elif cls_id == TRAFFIC_LIGHT_CLASS:
                det = Detection(bbox=bbox, class_name="traffic_light", confidence=conf, class_id=cls_id)
                result.traffic_lights.append(det)
                result.all_detections.append(det)

        # Reclassify auto-rickshaws (size between motorcycle and car)
        self._classify_autos(result)

        # Run helmet detection if model available
        if self.helmet_model is not None:
            self._detect_helmets(image, result)

        return result

    def _classify_autos(self, result: DetectionResult):
        """Heuristically relabel some 'car' detections as 'auto' (auto-rickshaw).

        COCO/YOLOv8 has no auto-rickshaw class, so we *infer* it from bounding-box
        size relative to the motorcycles and cars in the SAME image. This is an
        honest best-effort heuristic, not a trained classifier: perspective and
        occlusion can fool it. Reclassified detections are therefore flagged
        `inferred=True` and have their confidence discounted so that detection
        metrics and the Evidence Defensibility Score treat them as lower-trust
        rather than presenting a guess as a certain class.
        """
        motorcycles = [v for v in result.vehicles if v.class_name == "motorcycle"]
        cars = [v for v in result.vehicles if v.class_name == "car"]

        if not motorcycles or not cars:
            return  # Need reference sizes for both to interpolate

        avg_moto_area = np.mean([v.area for v in motorcycles])
        avg_car_area = np.mean([v.area for v in cars])

        if avg_car_area <= avg_moto_area:
            return

        # Auto size range: between bike and car
        low = avg_moto_area * 1.3
        high = avg_car_area * 0.8

        for v in result.vehicles:
            # Only reclassify cars that look like autos — never reclassify motorcycles
            if v.class_name == "car":
                aspect = v.width / max(v.height, 1)
                if low < v.area < high and 0.8 < aspect < 1.6:
                    v.class_name = "auto"
                    v.inferred = True
                    # Discount confidence to reflect heuristic (not model) origin.
                    v.confidence = round(float(v.confidence) * 0.6, 3)

    def _detect_helmets(self, image: np.ndarray, result: DetectionResult):
        """Run helmet model in TWO passes for maximum recall:
          Pass 1: Full image (catches everything the model can see)
          Pass 2: Per-motorcycle crops (catches small/far riders)
        """
        h_img, w_img = image.shape[:2]

        # Pass 1: Run on FULL image — catches most visible helmets/no-helmets
        full_results = self.helmet_model(image, conf=0.2, verbose=False)
        seen_boxes = set()

        if full_results and len(full_results) > 0:
            boxes = full_results[0].boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    bbox = tuple(int(v) for v in boxes.xyxy[i].cpu().numpy())
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    cls_name = self.helmet_model.names.get(cls_id, "").strip()

                    det = Detection(bbox=bbox, class_name=cls_name, confidence=conf, class_id=cls_id)
                    seen_boxes.add(bbox)

                    if "without" in cls_name.lower() or "no" in cls_name.lower():
                        result.no_helmets.append(det)
                    else:
                        result.helmets.append(det)
                    result.all_detections.append(det)

        # Pass 2: Run on expanded motorcycle crops (catches small/distant riders missed by full-image)
        motorcycles = [v for v in result.vehicles if v.class_name in ("motorcycle", "bicycle")]
        for bike in motorcycles:
            x1, y1, x2, y2 = [int(v) for v in bike.bbox]
            bike_h = y2 - y1
            bike_w = x2 - x1
            # Expand generously — rider's head can be far above bike bbox
            expand_top = int(bike_h * 1.5)
            expand_side = int(bike_w * 0.3)
            cx1 = max(0, x1 - expand_side)
            cy1 = max(0, y1 - expand_top)
            cx2 = min(w_img, x2 + expand_side)
            cy2 = min(h_img, y2 + int(bike_h * 0.1))

            crop = image[cy1:cy2, cx1:cx2]
            if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
                continue

            h_results = self.helmet_model(crop, conf=0.15, verbose=False)
            if not h_results or len(h_results) == 0:
                continue

            boxes = h_results[0].boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                bx1, by1, bx2, by2 = [int(v) for v in boxes.xyxy[i].cpu().numpy()]
                bbox = (bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1)

                # Skip if already detected in full-image pass (dedup by overlap)
                duplicate = False
                for sb in seen_boxes:
                    # Check IoU
                    ix1 = max(bbox[0], sb[0])
                    iy1 = max(bbox[1], sb[1])
                    ix2 = min(bbox[2], sb[2])
                    iy2 = min(bbox[3], sb[3])
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        area1 = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
                        area2 = (sb[2]-sb[0]) * (sb[3]-sb[1])
                        iou = inter / max(area1 + area2 - inter, 1)
                        if iou > 0.4:
                            duplicate = True
                            break
                if duplicate:
                    continue

                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = self.helmet_model.names.get(cls_id, "").strip()

                det = Detection(bbox=bbox, class_name=cls_name, confidence=conf, class_id=cls_id)
                seen_boxes.add(bbox)

                if "without" in cls_name.lower() or "no" in cls_name.lower():
                    result.no_helmets.append(det)
                else:
                    result.helmets.append(det)
                result.all_detections.append(det)
