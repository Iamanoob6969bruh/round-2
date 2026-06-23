"""
Helmet detection via head-region shape analysis.

A riding helmet has a distinctive signature:
  - Large, bulky relative to a bare head
  - Very SPHERICAL / rounded top (smooth circular arc)
  - SYMMETRICAL (left-right mirror symmetry)
  - Often a single smooth color/texture (covers hair/face)

A bare head: less round top, asymmetric (hair, face features), more texture variation.

This module analyzes the top portion of a rider's head region and scores
how "helmet-like" it is, combining:
  1. Top-contour circularity (how well the top fits a circle/arc)
  2. Left-right symmetry of the region
  3. Texture smoothness (helmets are smoother than hair/face)
"""

import cv2
import numpy as np


class HelmetHeadAnalyzer:
    """Analyze a rider's head crop to decide helmet vs no-helmet."""

    def analyze(self, head_crop: np.ndarray) -> tuple:
        """
        Returns (has_helmet: bool, confidence: float, details: dict).
        head_crop should be the upper region (head) of the front rider.
        """
        if head_crop.size == 0 or head_crop.shape[0] < 10 or head_crop.shape[1] < 10:
            return False, 0.0, {"reason": "crop too small"}

        gray = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY) if len(head_crop.shape) == 3 else head_crop

        circularity = self._top_circularity(gray)
        symmetry = self._symmetry(gray)
        smoothness = self._smoothness(gray)

        # Helmet score: weighted combination
        # Spherical top is the strongest signal (user emphasized this)
        helmet_score = 0.45 * circularity + 0.35 * symmetry + 0.20 * smoothness

        details = {
            "circularity": round(float(circularity), 3),
            "symmetry": round(float(symmetry), 3),
            "smoothness": round(float(smoothness), 3),
            "helmet_score": round(float(helmet_score), 3),
        }

        has_helmet = helmet_score >= 0.55
        # Confidence is distance from the 0.55 decision boundary, scaled
        confidence = min(abs(helmet_score - 0.55) * 2 + 0.5, 1.0)

        return has_helmet, float(confidence), details

    def _top_circularity(self, gray: np.ndarray) -> float:
        """Measure how spherical/rounded the top of the head region is.
        Fits the top silhouette to a circular arc — helmets are very round."""
        h, w = gray.shape

        # Segment foreground (head) from background using Otsu
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # The head might be darker or lighter than bg — try both polarities,
        # pick the one whose foreground is more centered/top-heavy
        candidates = [thresh, cv2.bitwise_not(thresh)]
        best_circularity = 0.0

        for mask in candidates:
            # Find the top edge profile: for each column, the first foreground row
            top_edge = []
            for col in range(w):
                rows = np.where(mask[:, col] > 0)[0]
                if len(rows) > 0:
                    top_edge.append((col, rows[0]))

            if len(top_edge) < w * 0.4:
                continue

            xs = np.array([p[0] for p in top_edge], dtype=np.float64)
            ys = np.array([p[1] for p in top_edge], dtype=np.float64)

            # Fit a circle to the top-edge points (algebraic circle fit)
            circ = self._circle_fit_quality(xs, ys)
            best_circularity = max(best_circularity, circ)

        return best_circularity

    def _circle_fit_quality(self, xs: np.ndarray, ys: np.ndarray) -> float:
        """Fit points to a circle, return goodness-of-fit 0-1.
        A helmet top fits a circle well; a bare/irregular head does not."""
        if len(xs) < 5:
            return 0.0

        # Algebraic circle fit: solve for center (a,b) and radius r
        # x^2 + y^2 + D*x + E*y + F = 0
        A = np.column_stack([xs, ys, np.ones_like(xs)])
        b = -(xs**2 + ys**2)
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0

        D, E, F = sol
        cx = -D / 2
        cy = -E / 2
        r_sq = cx**2 + cy**2 - F
        if r_sq <= 0:
            return 0.0
        r = np.sqrt(r_sq)

        # Residuals: distance of each point from the fitted circle
        dists = np.sqrt((xs - cx)**2 + (ys - cy)**2)
        residual = np.mean(np.abs(dists - r))

        # Normalize: small residual relative to radius = good circle fit
        if r < 1:
            return 0.0
        fit_quality = max(0.0, 1.0 - residual / (r * 0.3))
        return min(fit_quality, 1.0)

    def _symmetry(self, gray: np.ndarray) -> float:
        """Left-right mirror symmetry. Helmets are highly symmetric."""
        h, w = gray.shape
        # Focus on the top 60% (the helmet dome area)
        top = gray[:int(h * 0.6), :]
        left = top[:, :w // 2]
        right = top[:, w - w // 2:]
        right_flipped = cv2.flip(right, 1)

        # Match sizes
        min_w = min(left.shape[1], right_flipped.shape[1])
        left = left[:, :min_w].astype(np.float64)
        right_flipped = right_flipped[:, :min_w].astype(np.float64)

        # Normalized correlation
        diff = np.abs(left - right_flipped)
        symmetry = 1.0 - np.mean(diff) / 255.0
        return float(np.clip(symmetry, 0, 1))

    def _smoothness(self, gray: np.ndarray) -> float:
        """Texture smoothness. Helmets are smoother than hair/face."""
        # Top portion (helmet dome)
        h = gray.shape[0]
        top = gray[:int(h * 0.6), :]
        # Edge density: helmets have few internal edges, hair/face have many
        edges = cv2.Canny(top, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        # Low edge density = smooth = helmet-like
        smoothness = 1.0 - min(edge_density * 5, 1.0)
        return float(smoothness)
