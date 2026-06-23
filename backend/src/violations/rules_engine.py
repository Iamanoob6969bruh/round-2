"""
Scene Analysis & Violation Rules Engine.

Computes spatial relationships between detected objects,
then applies declarative rules to identify violations.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from ..detection.detector import Detection, DetectionResult
from .helmet_analyzer import HelmetHeadAnalyzer


@dataclass
class SpatialRelation:
    subject: Detection
    relation: str  # "on", "near", "inside", "above", "beyond"
    obj: Detection
    confidence: float


@dataclass
class Violation:
    violation_type: str
    confidence: float
    involved_objects: list
    description: str
    severity: str  # "low", "medium", "high"
    bbox: tuple = (0, 0, 0, 0)  # bounding box enclosing the violation


class SceneAnalyzer:
    """Compute spatial relationships between detections."""

    def __init__(self):
        self.iou_threshold = 0.1
        self.proximity_threshold = 50  # pixels

    def is_person_on_vehicle(self, person: Detection, vehicle: Detection) -> float:
        """Check if person is riding a vehicle. Returns confidence 0-1."""
        px1, py1, px2, py2 = person.bbox
        vx1, vy1, vx2, vy2 = vehicle.bbox

        # Horizontal overlap: person's center should be within vehicle's horizontal span
        px_center = (px1 + px2) / 2
        h_overlap = vx1 - 30 <= px_center <= vx2 + 30

        # Vertical: person should be above or overlapping with vehicle top
        # Person's bottom half should overlap with vehicle's top half
        person_bottom = py2
        vehicle_top = vy1
        vehicle_center_y = (vy1 + vy2) / 2

        # Person is "on" vehicle if their bottom is near vehicle top area
        v_close = person_bottom >= vehicle_top - vehicle.height * 0.3 and py1 < vehicle_center_y

        if h_overlap and v_close:
            # Compute IoU-like overlap
            ix1 = max(px1, vx1)
            iy1 = max(py1, vy1)
            ix2 = min(px2, vx2)
            iy2 = min(py2, vy2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                person_area = person.area
                ratio = inter / max(person_area, 1)
                return min(0.5 + ratio, 1.0)
            return 0.5
        return 0.0

    def is_above(self, obj_a: Detection, obj_b: Detection) -> float:
        """Check if obj_a is above obj_b."""
        a_bottom = obj_a.bbox[3]
        b_top = obj_b.bbox[1]
        h_overlap = self._horizontal_overlap(obj_a, obj_b)

        if a_bottom <= b_top + 20 and h_overlap > 0.3:
            return 1.0
        return 0.0

    def is_near(self, obj_a: Detection, obj_b: Detection, threshold: float = None) -> float:
        """Check proximity between two objects."""
        threshold = threshold or self.proximity_threshold
        dist = np.sqrt((obj_a.center[0] - obj_b.center[0])**2 + (obj_a.center[1] - obj_b.center[1])**2)
        if dist < threshold:
            return 1.0 - (dist / threshold)
        return 0.0

    def is_beyond_line(self, vehicle: Detection, line_y: int) -> float:
        """Check if vehicle has crossed a stop line.
        
        In a typical traffic camera (looking down the road):
        - Stop line is a horizontal line at some y coordinate
        - Vehicles CLOSER to camera have LARGER y values
        - A vehicle that crossed the stop line has moved closer to camera
          than the line, so its TOP edge (y1) has gone past (below) the line
        
        We check if the vehicle's top edge (front-facing side toward camera)
        is below the stop line y coordinate.
        """
        # Vehicle top edge = the front of the vehicle (closest to camera)
        vehicle_top = vehicle.bbox[1]
        vehicle_bottom = vehicle.bbox[3]
        
        # The vehicle must be IN the zone near the stop line
        # Its top must be below the line AND its center must be near the line
        # (not a car way behind or way in front)
        vehicle_center_y = (vehicle_top + vehicle_bottom) / 2
        
        # Vehicle is "beyond" if its center is below (past) the stop line
        if vehicle_center_y > line_y:
            # How far past the line (relative to vehicle height)
            overshoot = (vehicle_center_y - line_y) / max(vehicle.height, 1)
            # Only flag if clearly past (not just touching)
            if overshoot > 0.3:
                return min(overshoot, 1.0)
        return 0.0

    def count_persons_on_vehicle(self, vehicle: Detection, persons: list) -> list:
        """Count how many persons are on/associated with a vehicle."""
        riders = []
        for person in persons:
            conf = self.is_person_on_vehicle(person, vehicle)
            if conf > 0.3:
                riders.append((person, conf))
        return riders

    def _compute_vertical_overlap(self, a: Detection, b: Detection) -> float:
        y_top = max(a.bbox[1], b.bbox[1])
        y_bottom = min(a.bbox[3], b.bbox[3])
        if y_bottom <= y_top:
            return 0.0
        overlap = y_bottom - y_top
        return overlap / min(a.height, b.height)

    def _horizontal_overlap(self, a: Detection, b: Detection) -> float:
        x_left = max(a.bbox[0], b.bbox[0])
        x_right = min(a.bbox[2], b.bbox[2])
        if x_right <= x_left:
            return 0.0
        overlap = x_right - x_left
        return overlap / min(a.width, b.width)

    def _enclosing_bbox(self, objects: list) -> tuple:
        if not objects:
            return (0, 0, 0, 0)
        x1 = min(o.bbox[0] for o in objects)
        y1 = min(o.bbox[1] for o in objects)
        x2 = max(o.bbox[2] for o in objects)
        y2 = max(o.bbox[3] for o in objects)
        return (x1, y1, x2, y2)


class StopLineDetector:
    """Detect stop line = ONLY a zebra crossing visible on the road.
    
    Zebra crossing signature: alternating white and dark horizontal bands
    in the lower portion of the image. Nothing else counts.
    If no zebra crossing found → None → no stop line violation.
    """

    def detect(self, image: np.ndarray) -> Optional[int]:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Look in bottom 50% of image ONLY (where road+zebra actually is)
        roi_top = h // 2
        roi = gray[roi_top:, :]
        roi_h, roi_w = roi.shape

        # Zebra crossing = multiple horizontal white bands separated by dark gaps
        # For each row, compute mean brightness across the row.
        row_means = np.mean(roi, axis=1)

        # Need variation in row means (zebra has bright and dark rows)
        row_max = np.max(row_means)
        row_min = np.min(row_means)
        row_range = row_max - row_min
        if row_range < 15:
            return None  # Completely uniform, no markings

        # Bright rows = above midpoint between min and max
        mid = (row_max + row_min) / 2
        bright_rows = row_means > mid

        # Find clusters of consecutive bright rows (each stripe is a few pixels thick)
        stripes = []
        in_stripe = False
        start = 0
        for i in range(roi_h):
            if bright_rows[i] and not in_stripe:
                start = i
                in_stripe = True
            elif not bright_rows[i] and in_stripe:
                if 2 <= (i - start) <= 30:  # reasonable stripe thickness
                    stripes.append((start, i))
                in_stripe = False
        if in_stripe and 2 <= (roi_h - start) <= 30:
            stripes.append((start, roi_h))

        # Zebra = 3+ stripes close together with gaps between them
        if len(stripes) < 3:
            return None

        # Check that stripes are evenly spaced (zebra pattern)
        # and span a significant vertical range
        stripe_tops = [s[0] for s in stripes]
        gaps = [stripe_tops[i+1] - stripe_tops[i] for i in range(len(stripe_tops)-1)]
        
        if len(gaps) < 2:
            return None

        # Gaps should be somewhat uniform (within 2x of each other)
        median_gap = np.median(gaps)
        uniform_gaps = sum(1 for g in gaps if 0.4 * median_gap < g < 2.5 * median_gap)
        
        if uniform_gaps < len(gaps) * 0.6:
            return None  # Not uniform enough to be a zebra

        # The stop line = top edge of the zebra crossing
        return int(stripes[0][0]) + roi_top


class ViolationRulesEngine:
    """Declarative rules for traffic violation detection."""

    def __init__(self, scene_config: dict = None):
        """scene_config: optional overrides (usually empty — everything auto-detected)."""
        self.analyzer = SceneAnalyzer()
        self.helmet_analyzer = HelmetHeadAnalyzer()
        self.stop_line_detector = StopLineDetector()
        # `config` holds only STATIC, site-level overrides (e.g. surveyed
        # no-parking zones). It is NEVER mutated per-image: this engine is a
        # singleton reused across requests, so any per-image state must be
        # passed as a local parameter to avoid leaking between uploads.
        self.config = scene_config or {}

    def detect_all_violations(self, detections: DetectionResult, image_shape: tuple = None, image: np.ndarray = None) -> list:
        """Run the FULL set of theme violation rules on a single image.

        All seven violation classes from the challenge are active:
          1. Helmet non-compliance     5. Stop-line violation
          2. Triple riding             6. Red-light violation
          3. Seatbelt non-compliance   7. Illegal parking
          4. Wrong-side driving

        Per-image scene state (the auto-detected stop line) is computed locally
        and threaded through as a parameter — the shared instance is never
        mutated, so concurrent / sequential requests stay isolated.
        """
        if image_shape is None and image is not None:
            image_shape = image.shape

        # ── Scene geometry: auto-detect the stop line (zebra crossing) once,
        # then share the result with the rules that need it. A configured
        # override always wins over auto-detection.
        stop_line_y = self.config.get("stop_line_y")
        if stop_line_y is None and image is not None:
            stop_line_y = self.stop_line_detector.detect(image)

        violations = []
        violations.extend(self._check_helmet(detections, image))
        violations.extend(self._check_triple_riding(detections))
        violations.extend(self._check_seatbelt_vision(detections, image))
        violations.extend(self._check_wrong_side(detections, image_shape))
        violations.extend(self._check_stop_line(detections, stop_line_y))
        violations.extend(self._check_red_light(detections, image, stop_line_y))
        violations.extend(self._check_illegal_parking(detections, stop_line_y, image_shape))

        return self._deduplicate(violations)

    def _deduplicate(self, violations: list) -> list:
        """Remove duplicate violations that overlap significantly on the same type."""
        if len(violations) <= 1:
            return violations
        keep = []
        for v in violations:
            is_dup = False
            for k in keep:
                if k.violation_type != v.violation_type:
                    continue
                iou = self._bbox_iou(v.bbox, k.bbox)
                if iou > 0.3:
                    # Keep the higher-confidence one
                    if v.confidence > k.confidence:
                        keep.remove(k)
                        keep.append(v)
                    is_dup = True
                    break
            if not is_dup:
                keep.append(v)
        return keep

    @staticmethod
    def _bbox_iou(a: tuple, b: tuple) -> float:
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / max(area_a + area_b - inter, 1)

    def _check_helmet(self, detections: DetectionResult, image: np.ndarray = None) -> list:
        """Helmet check for two-wheeler riders.

        Logic:
          - If helmet model detected "Without helmet" near any motorcycle → flag directly
          - If no model detection but motorcycle has riders → analyze front rider's
            head shape (spherical top + symmetry = helmet present)
          - If no helmet signature found → flag
        """
        violations = []
        two_wheelers = [v for v in detections.vehicles if v.class_name in ("motorcycle", "bicycle")]

        # METHOD 1: Helmet model detected "Without helmet" near a two-wheeler
        for nh in detections.no_helmets:
            if nh.confidence < 0.2:
                continue

            # SUPPRESS if a "with helmet" detection overlaps the same area
            # (model detected both — trust the helmet-present detection)
            suppressed = False
            for h in detections.helmets:
                # Check overlap between this no-helmet and any helmet detection
                ix1 = max(nh.bbox[0], h.bbox[0])
                iy1 = max(nh.bbox[1], h.bbox[1])
                ix2 = min(nh.bbox[2], h.bbox[2])
                iy2 = min(nh.bbox[3], h.bbox[3])
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    min_area = min(nh.area, h.area)
                    if inter / max(min_area, 1) > 0.3:
                        suppressed = True
                        break
                # Also suppress if centers are very close (same head region)
                dist = ((nh.center[0] - h.center[0])**2 + (nh.center[1] - h.center[1])**2)**0.5
                if dist < max(nh.width, nh.height) * 0.8:
                    suppressed = True
                    break
            if suppressed:
                continue
            # Find nearest two-wheeler to this detection
            best_bike = None
            best_dist = float('inf')
            for bike in two_wheelers:
                # Check if no-helmet box is above or overlapping the bike
                dist = abs(nh.center[1] - bike.bbox[1])  # vertical proximity
                h_aligned = bike.bbox[0] - 50 < nh.center[0] < bike.bbox[2] + 50
                if h_aligned and dist < best_dist:
                    best_dist = dist
                    best_bike = bike

            if best_bike and best_dist < best_bike.height * 1.5:
                violations.append(Violation(
                    violation_type="helmet_violation",
                    confidence=nh.confidence,
                    involved_objects=[nh, best_bike],
                    description="Rider without helmet (model detection)",
                    severity="high",
                    bbox=self.analyzer._enclosing_bbox([nh, best_bike]),
                ))
            elif nh.confidence >= 0.4:
                # High-confidence no-helmet detection without a matched bike
                # (bike might be out of frame or misclassified) — still flag it
                violations.append(Violation(
                    violation_type="helmet_violation",
                    confidence=nh.confidence * 0.85,
                    involved_objects=[nh],
                    description="Rider without helmet (model detection, no bike matched)",
                    severity="high",
                    bbox=nh.bbox,
                ))

        # METHOD 2: Shape analysis for bikes without model detections
        # Only for motorcycles that weren't already flagged
        flagged_bikes = {id(v.involved_objects[-1]) for v in violations if v.involved_objects}
        
        for vehicle in two_wheelers:
            if id(vehicle) in flagged_bikes:
                continue

            riders = self.analyzer.count_persons_on_vehicle(vehicle, detections.persons)
            if not riders or image is None:
                continue

            # Front rider (highest head = smallest y1)
            front_rider = sorted(riders, key=lambda r: r[0].bbox[1])[0][0]

            # Skip if helmet model already detected "with helmet" on this rider's head
            rider_has_helmet = False
            for h in detections.helmets:
                if self._head_overlaps(h, front_rider):
                    rider_has_helmet = True
                    break
            if rider_has_helmet:
                continue

            head_crop = self._extract_head(front_rider, image)
            has_helmet, confidence, details = self.helmet_analyzer.analyze(head_crop)

            if not has_helmet and confidence > 0.4:
                violations.append(Violation(
                    violation_type="helmet_violation",
                    confidence=confidence * 0.7,
                    involved_objects=[front_rider, vehicle],
                    description=f"No helmet (shape: circ={details.get('circularity',0):.2f}, sym={details.get('symmetry',0):.2f})",
                    severity="high",
                    bbox=self.analyzer._enclosing_bbox([front_rider, vehicle]),
                ))

        return violations

    def _extract_head(self, person: Detection, image: np.ndarray) -> np.ndarray:
        """Extract the head region (top ~30%) of a person bounding box."""
        x1, y1, x2, y2 = [int(v) for v in person.bbox]
        head_h = int((y2 - y1) * 0.32)
        # Slight horizontal inset to focus on head, not shoulders
        inset = int((x2 - x1) * 0.1)
        hx1 = max(0, x1 + inset)
        hx2 = min(image.shape[1], x2 - inset)
        hy1 = max(0, y1)
        hy2 = min(image.shape[0], y1 + head_h)
        return image[hy1:hy2, hx1:hx2]

    def _head_overlaps(self, helmet_det: Detection, person: Detection) -> bool:
        """Check if a helmet/no-helmet detection is on the person's head area."""
        px1, py1, px2, py2 = person.bbox
        head_y2 = py1 + (py2 - py1) * 0.35
        hx, hy = helmet_det.center
        return px1 - 20 <= hx <= px2 + 20 and py1 - 20 <= hy <= head_y2 + 20

    def _check_triple_riding(self, detections: DetectionResult) -> list:
        """More than 2 persons on a two-wheeler — only when tightly clustered on the vehicle."""
        violations = []
        two_wheelers = [v for v in detections.vehicles if v.class_name in ("motorcycle", "bicycle")]

        for vehicle in two_wheelers:
            riders = self.analyzer.count_persons_on_vehicle(vehicle, detections.persons)
            if len(riders) <= 2:
                continue

            # Extra strictness: all riders must be horizontally tight (on the same narrow bike)
            # and vertically stacked close together
            persons = [r[0] for r in riders]
            vx1, vy1, vx2, vy2 = vehicle.bbox
            vehicle_w = vx2 - vx1

            # Check that persons are within the vehicle's horizontal span (tight cluster)
            tight_riders = []
            for p, conf in riders:
                px_center = (p.bbox[0] + p.bbox[2]) / 2
                # Person center must be well within the bike width (not just at edges)
                if vx1 <= px_center <= vx2 and conf > 0.4:
                    tight_riders.append((p, conf))

            if len(tight_riders) > 2:
                # Verify riders are close to each other vertically (stacked on same bike)
                centers_y = sorted([(p.bbox[1] + p.bbox[3]) / 2 for p, _ in tight_riders])
                max_gap = max(centers_y[i+1] - centers_y[i] for i in range(len(centers_y)-1))
                # Gap between any two adjacent riders shouldn't exceed vehicle height
                if max_gap < (vy2 - vy1) * 1.2:
                    persons_involved = [r[0] for r in tight_riders]
                    avg_conf = np.mean([r[1] for r in tight_riders])
                    violations.append(Violation(
                        violation_type="triple_riding",
                        confidence=avg_conf * 0.9,
                        involved_objects=persons_involved + [vehicle],
                        description=f"{len(tight_riders)} persons on {vehicle.class_name}",
                        severity="high",
                        bbox=self.analyzer._enclosing_bbox(persons_involved + [vehicle]),
                    ))

        return violations

    def _check_red_light(self, detections: DetectionResult, image: np.ndarray = None, stop_line_y: int = None) -> list:
        """Vehicle crossing intersection when traffic light is red."""
        violations = []
        if not detections.traffic_lights:
            return violations

        if stop_line_y is None:
            # Estimate: traffic light position suggests stop line nearby
            tl = detections.traffic_lights[0]
            stop_line_y = tl.bbox[3] + 100  # below traffic light

        for tl in detections.traffic_lights:
            signal_state = self._detect_signal_color(tl, image)
            if signal_state != "red":
                continue

            for vehicle in detections.vehicles:
                beyond = self.analyzer.is_beyond_line(vehicle, stop_line_y)
                if beyond > 0.3:
                    violations.append(Violation(
                        violation_type="red_light_violation",
                        confidence=beyond * tl.confidence * 0.8,
                        involved_objects=[vehicle, tl],
                        description=f"{vehicle.class_name} crossed stop line on red signal",
                        severity="high",
                        bbox=vehicle.bbox,
                    ))

        return violations

    def _check_stop_line(self, detections: DetectionResult, stop_line_y: int = None) -> list:
        """Flag vehicles crossing (near and past) the zebra crossing line."""
        violations = []
        if stop_line_y is None:
            return violations

        for vehicle in detections.vehicles:
            # Only flag vehicles whose body straddles or is just past the line
            # (not vehicles far away from the line)
            vehicle_top = vehicle.bbox[1]
            vehicle_bottom = vehicle.bbox[3]

            # Vehicle must be near the stop line: its bbox overlaps or just crossed it
            # Skip vehicles entirely above (before) the line
            if vehicle_bottom < stop_line_y:
                continue
            # Skip vehicles far past the line (more than 1.5x their height beyond)
            overshoot = vehicle_top - stop_line_y
            if overshoot > vehicle.height * 1.5:
                continue
            # Must have actually crossed: top edge past the line
            if vehicle_top > stop_line_y:
                confidence = min(overshoot / max(vehicle.height, 1) + 0.3, 1.0) * vehicle.confidence
                violations.append(Violation(
                    violation_type="stop_line_violation",
                    confidence=round(confidence, 3),
                    involved_objects=[vehicle],
                    description=f"{vehicle.class_name} beyond zebra crossing",
                    severity="medium",
                    bbox=vehicle.bbox,
                ))

        return violations

    def _check_wrong_side(self, detections: DetectionResult, image_shape: tuple = None) -> list:
        """Wrong-side driving detection (single-image proxy).

        True direction of travel cannot be recovered from one still frame, so
        we detect the strongest *single-image* signature of wrong-side driving:
        a vehicle that sits ALONE on one side of the dominant traffic stream
        while essentially every other vehicle occupies the opposite side.

        Conservative by construction — this does NOT fire for ordinary two-lane
        traffic (which has vehicles distributed on both sides). It only fires
        when one vehicle is a clear lateral outlier separated from the pack by a
        wide gap, i.e. it has drifted into the oncoming lane. Anything weaker is
        left for the Evidence Defensibility Score / human review downstream.
        """
        violations = []
        if image_shape is None:
            return violations
        vehicles = detections.vehicles
        if len(vehicles) < 3:
            return violations  # need a "stream" to be an outlier from

        h, w = image_shape[:2]

        for vehicle in vehicles:
            vx = vehicle.center[0]
            others = [o for o in vehicles if o is not vehicle]
            others_x = np.array([o.center[0] for o in others])
            pack_center = float(np.median(others_x))
            pack_spread = float(others_x.max() - others_x.min())

            # (1) The remaining vehicles must form a coherent stream (clustered),
            #     not be scattered across the whole road (which would be ordinary
            #     multi-lane traffic, never a violation).
            if pack_spread > 0.35 * w:
                continue

            # (2) Wide lateral gap to the nearest other vehicle (a real lane
            #     departure, not edge-of-lane jitter).
            nearest_gap = min(abs(vx - ox) for ox in others_x)
            if nearest_gap < 0.22 * w:
                continue

            # (3) The vehicle must sit well away from the pack centre — it has
            #     drifted into the oncoming side.
            if abs(vx - pack_center) < 0.18 * w:
                continue

            conf = min(nearest_gap / (0.5 * w), 1.0) * vehicle.confidence * 0.6
            violations.append(Violation(
                violation_type="wrong_side_driving",
                confidence=round(conf, 3),
                involved_objects=[vehicle],
                description=f"{vehicle.class_name} isolated on the opposite side of the traffic stream",
                severity="high",
                bbox=vehicle.bbox,
            ))

        return violations

    def _check_illegal_parking(self, detections: DetectionResult, stop_line_y: int = None,
                               image_shape: tuple = None) -> list:
        """Illegal parking detection.

        Two complementary signals, both derivable from a single still frame:

          (a) Surveyed no-parking zones (site config) — a vehicle whose centre
              falls inside a declared zone is flagged. Highest confidence.

          (b) Parked-on-crossing heuristic — a vehicle whose body sits squarely
              ON the detected zebra crossing / stop line (straddling it rather
              than just nosing past it) is obstructing a pedestrian crossing,
              which is an illegal stop. This needs no prior site survey.

        Direction/time of day cannot be inferred from one image, so we stay
        conservative and let the Evidence Defensibility Score downstream route
        low-quality flags to human review rather than auto-issuing them.
        """
        violations = []

        # ── (a) Configured no-parking zones ──
        no_parking_zones = self.config.get("no_parking_zones", [])
        flagged = set()
        for zone in no_parking_zones:
            zx1, zy1, zx2, zy2 = zone
            for vehicle in detections.vehicles:
                vx, vy = vehicle.center
                if zx1 <= vx <= zx2 and zy1 <= vy <= zy2 and id(vehicle) not in flagged:
                    flagged.add(id(vehicle))
                    violations.append(Violation(
                        violation_type="illegal_parking",
                        confidence=round(vehicle.confidence * 0.7, 3),
                        involved_objects=[vehicle],
                        description=f"{vehicle.class_name} in declared no-parking zone",
                        severity="medium",
                        bbox=vehicle.bbox,
                    ))

        # ── (b) Parked-on-crossing heuristic ──
        # A vehicle straddles the crossing when its bbox spans the stop line
        # (top above, bottom below) — i.e. it is sitting ON the markings, not
        # merely crossing the leading edge.
        if stop_line_y is not None:
            for vehicle in detections.vehicles:
                if id(vehicle) in flagged:
                    continue
                # cars/buses/trucks parking on a crossing is the typical case
                if vehicle.class_name not in ("car", "bus", "truck", "auto"):
                    continue
                vtop, vbottom = vehicle.bbox[1], vehicle.bbox[3]
                straddle = vtop < stop_line_y < vbottom
                if not straddle:
                    continue
                # How much of the vehicle body sits across the line (0..1):
                # require a substantial overlap so a vehicle merely touching the
                # line isn't mistaken for a parked obstruction.
                inside = (vbottom - stop_line_y) / max(vehicle.height, 1)
                if 0.35 <= inside <= 0.9:
                    flagged.add(id(vehicle))
                    violations.append(Violation(
                        violation_type="illegal_parking",
                        confidence=round(vehicle.confidence * 0.55, 3),
                        involved_objects=[vehicle],
                        description=f"{vehicle.class_name} obstructing pedestrian crossing / stop line",
                        severity="medium",
                        bbox=vehicle.bbox,
                    ))

        return violations

    def _check_seatbelt(self, detections: DetectionResult) -> list:
        """Legacy placeholder — replaced by _check_seatbelt_vision."""
        return []

    def _check_seatbelt_vision(self, detections: DetectionResult, image: np.ndarray) -> list:
        """Seatbelt non-compliance via diagonal-strap detection.

        Logic: for each car with a person overlapping (driver), crop the
        person's torso region and look for a diagonal line (the seatbelt strap).
        Seatbelt = strong diagonal edge from shoulder to opposite hip.
        Absence of this diagonal in the torso → flag.
        """
        violations = []
        if image is None:
            return violations

        cars = [v for v in detections.vehicles if v.class_name in ("car", "bus", "truck")]
        if not cars:
            return violations

        h_img, w_img = image.shape[:2]

        for car in cars:
            # Find persons overlapping with this car (driver/passenger)
            for person in detections.persons:
                # Check if person is inside the car's bbox (visible through windshield)
                px, py = person.center
                cx1, cy1, cx2, cy2 = car.bbox
                if not (cx1 <= px <= cx2 and cy1 <= py <= cy2):
                    continue

                # Torso region: middle 40-75% vertically of the person bbox
                px1, py1, px2, py2 = [int(v) for v in person.bbox]
                torso_y1 = py1 + int((py2 - py1) * 0.30)
                torso_y2 = py1 + int((py2 - py1) * 0.75)
                torso = image[max(0, torso_y1):min(h_img, torso_y2),
                              max(0, px1):min(w_img, px2)]

                if torso.size == 0 or torso.shape[0] < 15 or torso.shape[1] < 15:
                    continue

                has_belt = self._detect_diagonal_strap(torso)
                if not has_belt:
                    conf = person.confidence * car.confidence * 0.7
                    violations.append(Violation(
                        violation_type="seatbelt_violation",
                        confidence=round(conf, 3),
                        involved_objects=[person, car],
                        description=f"No diagonal seatbelt strap detected on occupant in {car.class_name}",
                        severity="medium",
                        bbox=self.analyzer._enclosing_bbox([person]),
                    ))
                break  # one person per car is enough

        return violations

    def _detect_diagonal_strap(self, torso: np.ndarray) -> bool:
        """Detect a diagonal line (seatbelt strap) in the torso crop.

        A seatbelt appears as a strong diagonal edge (~30-60 degrees)
        crossing from one shoulder toward the opposite hip.
        Uses Hough line detection on edges.
        """
        gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY) if len(torso.shape) == 3 else torso
        edges = cv2.Canny(gray, 50, 150)

        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20,
                                minLineLength=int(min(torso.shape[:2]) * 0.4),
                                maxLineGap=10)
        if lines is None:
            return False

        # Check for diagonal lines (30-60 degrees from horizontal)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx < 1:
                continue
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            # Seatbelt strap runs at roughly 30-65 degrees
            if 25 <= angle <= 70:
                # Length must be significant relative to torso
                length = np.sqrt(dx**2 + dy**2)
                if length > min(torso.shape[:2]) * 0.35:
                    return True
        return False

    def _detect_signal_color(self, traffic_light: Detection, image: np.ndarray = None) -> str:
        """Detect traffic light color from the crop using HSV color analysis."""
        if image is None:
            return "unknown"
        x1, y1, x2, y2 = [int(v) for v in traffic_light.bbox]
        h_img, w_img = image.shape[:2]
        crop = image[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return "unknown"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Red in HSV: hue 0-10 or 160-180, high saturation, high value
        mask_r1 = cv2.inRange(hsv, (0, 80, 80), (12, 255, 255))
        mask_r2 = cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
        red_pixels = cv2.countNonZero(mask_r1) + cv2.countNonZero(mask_r2)
        # Green: hue 40-85
        mask_g = cv2.inRange(hsv, (40, 60, 60), (85, 255, 255))
        green_pixels = cv2.countNonZero(mask_g)
        # Yellow: hue 15-35
        mask_y = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
        yellow_pixels = cv2.countNonZero(mask_y)

        total = crop.shape[0] * crop.shape[1]
        threshold = total * 0.05  # at least 5% of the crop should be that color

        if red_pixels > threshold and red_pixels > green_pixels and red_pixels > yellow_pixels:
            return "red"
        elif green_pixels > threshold and green_pixels > red_pixels:
            return "green"
        elif yellow_pixels > threshold:
            return "yellow"
        return "unknown"
