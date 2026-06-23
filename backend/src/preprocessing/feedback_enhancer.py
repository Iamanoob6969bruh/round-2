"""
Feedback-Loop Image Enhancement.

Control-loop approach:
  1. Measure image metrics (resolution, brightness/dimness, overexposure, contrast, blur)
  2. Compare against predefined TARGET metrics (ideal values)
  3. Apply adjustments proportional to the error (toward target)
  4. Re-measure → repeat until metrics converge or max iterations reached

This is NOT one-shot enhancement — it iterates until the image matches the
standard reference profile as closely as possible.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field


# ---- TARGET METRICS (the "standard" well-exposed, sharp image) ----
TARGET = {
    "brightness": 120.0,      # mean luminance (0-255), mid-bright
    "contrast": 60.0,         # std-dev of luminance — good spread
    "overexposure": 0.02,     # fraction of blown-out (>250) pixels — keep low
    "underexposure": 0.05,    # fraction of crushed (<10) pixels — keep low
    "sharpness": 300.0,       # Laplacian variance — crisp
    "min_resolution": 1000,   # min width in px — upscale if smaller
}

# Convergence tolerances (how close is "close enough")
TOLERANCE = {
    "brightness": 12.0,
    "contrast": 15.0,
    "overexposure": 0.03,
    "underexposure": 0.04,
    "sharpness": 120.0,
}


@dataclass
class Metrics:
    width: int
    height: int
    brightness: float
    contrast: float
    overexposure: float
    underexposure: float
    sharpness: float

    def as_dict(self):
        return {
            "resolution": f"{self.width}x{self.height}",
            "brightness": round(self.brightness, 1),
            "contrast": round(self.contrast, 1),
            "overexposure": round(self.overexposure, 4),
            "underexposure": round(self.underexposure, 4),
            "sharpness": round(self.sharpness, 1),
        }


@dataclass
class EnhancementTrace:
    """Records the feedback loop for transparency/debugging."""
    iterations: int = 0
    history: list = field(default_factory=list)   # list of (action, metrics_dict)
    initial: dict = None
    final: dict = None
    converged: bool = False


class FeedbackLoopEnhancer:
    """Iteratively enhance an image to match TARGET metrics."""

    def __init__(self, target: dict = None, tolerance: dict = None, max_iterations: int = 6):
        self.target = target or TARGET
        self.tolerance = tolerance or TOLERANCE
        self.max_iterations = max_iterations

    # ---------- Measurement ----------

    def measure(self, image: np.ndarray) -> Metrics:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        total = gray.size

        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        overexposure = float(np.sum(gray > 250) / total)
        underexposure = float(np.sum(gray < 10) / total)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        return Metrics(w, h, brightness, contrast, overexposure, underexposure, sharpness)

    def _error(self, m: Metrics) -> dict:
        """Signed error = current - target for each metric."""
        return {
            "brightness": m.brightness - self.target["brightness"],
            "contrast": m.contrast - self.target["contrast"],
            "overexposure": m.overexposure - self.target["overexposure"],
            "underexposure": m.underexposure - self.target["underexposure"],
            "sharpness": m.sharpness - self.target["sharpness"],
        }

    def _is_converged(self, m: Metrics) -> bool:
        err = self._error(m)
        return (abs(err["brightness"]) <= self.tolerance["brightness"] and
                abs(err["contrast"]) <= self.tolerance["contrast"] and
                err["overexposure"] <= self.tolerance["overexposure"] and
                err["underexposure"] <= self.tolerance["underexposure"] and
                err["sharpness"] >= -self.tolerance["sharpness"])  # sharper is fine

    def _score(self, m: Metrics) -> float:
        """Lower = closer to target. Weighted normalized distance."""
        err = self._error(m)
        return (abs(err["brightness"]) / 128 +
                abs(err["contrast"]) / 60 +
                max(err["overexposure"], 0) * 5 +
                max(err["underexposure"], 0) * 5 +
                max(-err["sharpness"], 0) / 300)

    # ---------- One correction step ----------

    def _apply_correction(self, image: np.ndarray, m: Metrics) -> tuple:
        """Apply ONE proportional correction toward the target. Returns (image, action)."""
        err = self._error(m)
        actions = []
        result = image

        # Priority 1: Fix overexposure (reduce highlights)
        if err["overexposure"] > self.tolerance["overexposure"]:
            gamma = 1.0 + min(err["overexposure"] * 5, 0.5)  # dampened to avoid oscillation
            result = self._gamma(result, gamma)
            actions.append(f"reduce_highlights(γ={gamma:.2f})")

        # Priority 2: Fix brightness (gamma toward target)
        elif abs(err["brightness"]) > self.tolerance["brightness"]:
            # gamma < 1 brightens, > 1 darkens
            ratio = self.target["brightness"] / max(m.brightness, 1)
            gamma = np.clip(1.0 / ratio, 0.4, 2.5)
            result = self._gamma(result, gamma)
            actions.append(f"brightness(γ={gamma:.2f})")

        # Priority 3: Fix low contrast (CLAHE)
        if err["contrast"] < -self.tolerance["contrast"]:
            clip = np.clip(abs(err["contrast"]) / 15, 1.5, 5.0)
            result = self._clahe(result, clip)
            actions.append(f"contrast(clip={clip:.1f})")

        # Priority 4: Fix blur — unsharp mask
        if err["sharpness"] < -self.tolerance["sharpness"]:
            strength = np.clip(abs(err["sharpness"]) / 300, 0.5, 2.0)
            result = self._unsharp(result, strength)
            actions.append(f"sharpen(s={strength:.2f})")

        if not actions:
            actions.append("no-op")
        return result, " + ".join(actions)

    # ---------- Main loop ----------

    def enhance(self, image: np.ndarray) -> tuple:
        """Run the feedback loop. Returns (best_image, trace)."""
        if image is None or image.size == 0:
            trace = EnhancementTrace()
            trace.converged = True
            return image, trace
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("enhance() requires a BGR (3-channel) uint8 image")

        trace = EnhancementTrace()
        result = image.copy()

        # Step 0: Upscale if below target resolution (do once, up front)
        m = self.measure(result)
        if m.width < self.target["min_resolution"]:
            scale = self.target["min_resolution"] / m.width
            result = cv2.resize(result, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            trace.history.append(("upscale", self.measure(result).as_dict()))

        m = self.measure(result)
        trace.initial = m.as_dict()

        best_image = result.copy()
        best_score = self._score(m)
        worse_count = 0

        # Iterate
        for i in range(self.max_iterations):
            if self._is_converged(m):
                trace.converged = True
                break

            result, action = self._apply_correction(result, m)
            m = self.measure(result)
            trace.iterations = i + 1
            trace.history.append((action, m.as_dict()))

            score = self._score(m)
            if score < best_score:
                best_score = score
                best_image = result.copy()
                worse_count = 0
            else:
                worse_count += 1
                if worse_count >= 2:
                    break

        trace.final = self.measure(best_image).as_dict()
        if self._is_converged(self.measure(best_image)):
            trace.converged = True

        return best_image, trace

    # ---------- Primitive operations ----------

    def _gamma(self, image, gamma):
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    def _clahe(self, image, clip):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def _unsharp(self, image, strength):
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        amount = 1.0 + strength
        result = cv2.addWeighted(image, amount, gaussian, 1.0 - amount, 0)
        return np.clip(result, 0, 255).astype(np.uint8)


class PlateEnhancer:
    """Iterative enhancement specifically for a license plate crop.
    Repeats until OCR-readable target (sharp + high contrast) is reached."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations

    def enhance(self, crop: np.ndarray) -> np.ndarray:
        """Iteratively enhance plate crop for maximum readability."""
        if crop.size == 0:
            return crop

        result = crop.copy()
        h, w = result.shape[:2]

        # Upscale aggressively (plates need big pixels for OCR)
        target_w = 400
        if w < target_w:
            scale = target_w / w
            result = cv2.resize(result, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        for _ in range(self.max_iterations):
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
            brightness = np.mean(gray)
            contrast = np.std(gray)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            changed = False

            # Brighten/darken toward 130
            if abs(brightness - 130) > 20:
                gamma = np.clip(np.log(130/255) / np.log(max(brightness,1)/255), 0.4, 2.5)
                result = self._gamma(result, gamma)
                changed = True

            # Boost contrast if low
            if contrast < 50:
                result = self._clahe(result, 3.0)
                changed = True

            # Sharpen if soft
            if sharpness < 200:
                gaussian = cv2.GaussianBlur(result, (0, 0), 1.5)
                result = cv2.addWeighted(result, 2.0, gaussian, -1.0, 0)
                changed = True

            if not changed:
                break

        # Final denoise to clean up
        if len(result.shape) == 3:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        else:
            gray = result
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _gamma(self, image, gamma):
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    def _clahe(self, image, clip):
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(4, 4)).apply(l)
            return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return cv2.createCLAHE(clipLimit=clip, tileGridSize=(4, 4)).apply(image)
