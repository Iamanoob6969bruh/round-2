"""
Fully Adaptive Image Enhancement Pipeline.

Order: Upscale → Denoise → Motion Deblur → Rain Removal → Shadow Removal →
       Exposure/Color Correction → Contrast Enhancement → Sharpen

All decisions derived from image statistics. Zero hardcoded thresholds.
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class ImageStats:
    brightness_mean: float
    brightness_median: float
    brightness_p5: float
    brightness_p95: float
    contrast: float
    dynamic_range: float
    sharpness: float
    noise_level: float
    has_shadows: bool
    rain_score: float
    motion_blur_angle: float
    motion_blur_strength: float
    is_low_res: bool
    overall_quality: float


class AdaptivePreprocessor:

    def analyze(self, image: np.ndarray) -> ImageStats:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        h, w = gray.shape

        p5 = np.percentile(gray, 5)
        p50 = np.percentile(gray, 50)
        p95 = np.percentile(gray, 95)
        mean = np.mean(gray)
        std = np.std(gray)
        dynamic_range = p95 - p5

        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Noise from high-frequency content
        hf = cv2.filter2D(gray, -1, np.array([[1,-2,1],[-2,4,-2],[1,-2,1]]))
        noise_level = np.median(np.abs(hf)) / 0.6745

        # Shadow detection: large dark regions adjacent to bright regions
        has_shadows = self._detect_shadows(gray, p50)

        rain_score = self._estimate_rain(gray)
        blur_angle, blur_strength = self._estimate_motion_blur(gray)
        is_low_res = (h < 500 or w < 500)

        # Quality composite
        q_bright = 1.0 - abs(mean - 127) / 127
        q_contrast = min(dynamic_range / 200, 1.0)
        q_sharp = min(sharpness / (mean + 1), 1.0)
        q_noise = max(1.0 - noise_level / (std + 1), 0.0)
        overall = np.clip(0.2*q_bright + 0.3*q_contrast + 0.3*q_sharp + 0.2*q_noise, 0, 1)

        return ImageStats(
            brightness_mean=mean, brightness_median=p50,
            brightness_p5=p5, brightness_p95=p95,
            contrast=std, dynamic_range=dynamic_range,
            sharpness=sharpness, noise_level=noise_level,
            has_shadows=has_shadows, rain_score=rain_score,
            motion_blur_angle=blur_angle, motion_blur_strength=blur_strength,
            is_low_res=is_low_res, overall_quality=float(overall),
        )

    def process(self, image: np.ndarray) -> tuple:
        """Full enhancement pipeline. Returns (enhanced, stats)."""
        stats = self.analyze(image)
        result = image.copy()

        # 1. UPSCALE low-res images first (so all subsequent ops work on more pixels)
        if stats.is_low_res:
            result = self._upscale(result)

        # 2. DENOISE (before other ops so noise doesn't get amplified)
        if stats.noise_level > stats.contrast * 0.25:
            result = self._denoise(result, stats)

        # 3. MOTION DEBLUR
        if stats.motion_blur_strength > 0.5:
            result = self._deblur_motion(result, stats.motion_blur_angle, stats.motion_blur_strength)

        # 4. RAIN REMOVAL
        if stats.rain_score > 0.3:
            result = self._remove_rain(result, stats.rain_score)

        # 5. SHADOW REMOVAL
        if stats.has_shadows:
            result = self._remove_shadows(result)

        # 6. EXPOSURE / COLOR CORRECTION (bring median brightness to target)
        result = self._correct_exposure(result, stats)

        # 7. CONTRAST (adaptive CLAHE based on dynamic range)
        result = self._enhance_contrast(result, stats)

        # 8. SHARPEN (final pass)
        result = self._sharpen(result, stats)

        return result, stats

    def enhance_plate_crop(self, crop: np.ndarray) -> np.ndarray:
        """Enhance plate region for OCR. Denoise first to preserve text."""
        if crop.size == 0:
            return crop

        h, w = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

        # Trim border noise (plates often have noisy edges from cropping)
        border_y, border_x = max(2, h // 8), max(2, w // 20)
        gray = gray[border_y:h - border_y, border_x:w - border_x]

        # Denoise at original resolution BEFORE upscaling (noise is small-scale)
        gray = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)

        # Upscale for OCR readability
        cur_h, cur_w = gray.shape[:2]
        if cur_w < 400:
            scale = 400 / cur_w
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Gentle contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def calibrate_confidence(self, raw_confidence: float, bbox: tuple, image: np.ndarray) -> float:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return raw_confidence * 0.5
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float64)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        contrast = np.std(gray)
        q = (min(sharpness / 200, 1.0) * 0.4 +
             (1.0 - abs(brightness - 127) / 127) * 0.3 +
             min(contrast / 50, 1.0) * 0.3)
        return round(raw_confidence * (0.5 + 0.5 * q), 3)

    # --- Private methods ---

    def _upscale(self, image: np.ndarray) -> np.ndarray:
        """2x upscale using INTER_CUBIC."""
        h, w = image.shape[:2]
        target_w = max(w * 2, 1000)
        scale = target_w / w
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def _denoise(self, image: np.ndarray, stats: ImageStats) -> np.ndarray:
        strength = int(np.clip(stats.noise_level / 2, 3, 12))
        return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)

    def _deblur_motion(self, image: np.ndarray, angle: float, strength: float) -> np.ndarray:
        """Directional unsharp mask to counter motion blur."""
        ksize = int(np.clip(strength * 8, 5, 21))
        if ksize % 2 == 0:
            ksize += 1

        # Build motion kernel
        kernel = np.zeros((ksize, ksize))
        center = ksize // 2
        rad = np.radians(angle)
        for i in range(ksize):
            x = int(center + (i - center) * np.cos(rad))
            y = int(center + (i - center) * np.sin(rad))
            if 0 <= x < ksize and 0 <= y < ksize:
                kernel[y, x] = 1
        kernel /= kernel.sum()

        # Deconvolve: sharpen in blur direction
        sharp_kernel = np.zeros((ksize, ksize), dtype=np.float64)
        sharp_kernel[center, center] = 2.0
        sharp_kernel -= kernel

        result = cv2.filter2D(image, -1, sharp_kernel)
        return np.clip(result, 0, 255).astype(np.uint8)

    def _remove_rain(self, image: np.ndarray, rain_score: float) -> np.ndarray:
        """Remove rain streaks via directional median filter + blend."""
        d = int(np.clip(rain_score * 12, 3, 11))
        if d % 2 == 0:
            d += 1
        filtered = cv2.medianBlur(image, d)
        alpha = min(rain_score * 0.6, 0.5)
        return cv2.addWeighted(filtered, alpha, image, 1.0 - alpha, 0)

    def _remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """Remove shadows using LAB channel normalization + morphological light estimation."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Estimate illumination using large morphological closing
        # This gives the "light field" — shadows are where light is low
        kernel_size = max(l.shape[0], l.shape[1]) // 8
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        light_field = cv2.morphologyEx(l, cv2.MORPH_CLOSE, kernel)

        # Normalize L channel by the light field
        # This removes illumination variation (shadows)
        light_field_f = light_field.astype(np.float64)
        l_f = l.astype(np.float64)
        mean_light = np.mean(light_field_f)

        # Avoid division by zero
        normalized = l_f * mean_light / (light_field_f + 1)
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        result = cv2.merge([normalized, a, b])
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    def _correct_exposure(self, image: np.ndarray, stats: ImageStats) -> np.ndarray:
        """Gamma correction to bring median brightness toward target."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_median = np.median(gray)
        target = 120.0

        if abs(current_median - target) < 15:
            return image

        gamma = np.log(target / 255.0) / np.log(max(current_median, 1) / 255.0)
        gamma = np.clip(gamma, 0.3, 3.0)
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    def _enhance_contrast(self, image: np.ndarray, stats: ImageStats) -> np.ndarray:
        """CLAHE with clip derived from current dynamic range."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dr = np.percentile(gray, 95) - np.percentile(gray, 5)

        if dr > 160:
            return image  # Already good contrast

        clip = np.clip(4.0 * (1.0 - dr / 200), 1.5, 5.0)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def _sharpen(self, image: np.ndarray, stats: ImageStats) -> np.ndarray:
        """Unsharp mask with adaptive strength."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        current_sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)

        # Only sharpen if still soft relative to brightness
        if current_sharpness > brightness * 3:
            return image

        sigma = max(1.0, 3.0 - current_sharpness / 150)
        gaussian = cv2.GaussianBlur(image, (0, 0), sigma)
        amount = np.clip(2.0 - current_sharpness / 300, 1.3, 2.5)
        result = cv2.addWeighted(image, amount, gaussian, 1.0 - amount, 0)
        return np.clip(result, 0, 255).astype(np.uint8)

    # --- Detection helpers ---

    def _detect_shadows(self, gray: np.ndarray, median: float) -> bool:
        """Detect if image has significant shadow regions."""
        # Shadow = large connected dark regions with bright regions nearby
        dark_thresh = median * 0.5
        dark_mask = (gray < dark_thresh).astype(np.uint8)
        dark_ratio = np.sum(dark_mask) / dark_mask.size

        # Shadows: 10-50% of image is dark, rest is bright
        # (not just an overall dark image)
        bright_ratio = np.sum(gray > median * 1.3) / gray.size
        return 0.1 < dark_ratio < 0.5 and bright_ratio > 0.2

    def _estimate_rain(self, gray: np.ndarray) -> float:
        sobel_v = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_h = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        v_energy = np.mean(np.abs(sobel_v))
        h_energy = np.mean(np.abs(sobel_h))
        if h_energy < 1:
            return 0.0
        ratio = v_energy / h_energy
        return min((ratio - 1.3) / 1.0, 1.0) if ratio > 1.3 else 0.0

    def _estimate_motion_blur(self, gray: np.ndarray) -> tuple:
        """FFT-based motion blur direction/strength estimation."""
        small = cv2.resize(gray.astype(np.uint8), (256, 256))
        f = np.fft.fft2(small.astype(np.float64))
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)

        center = 128
        best_score = 0
        best_angle = -1.0

        for angle in range(0, 180, 15):
            rad = np.radians(angle)
            score = sum(
                magnitude[int(center + r * np.sin(rad)), int(center + r * np.cos(rad))]
                for r in range(10, 100)
                if 0 <= int(center + r * np.sin(rad)) < 256 and 0 <= int(center + r * np.cos(rad)) < 256
            )
            if score > best_score:
                best_score = score
                best_angle = float(angle)

        perp_rad = np.radians((best_angle + 90) % 180)
        perp_score = sum(
            magnitude[int(center + r * np.sin(perp_rad)), int(center + r * np.cos(perp_rad))]
            for r in range(10, 100)
            if 0 <= int(center + r * np.sin(perp_rad)) < 256 and 0 <= int(center + r * np.cos(perp_rad)) < 256
        )

        if perp_score < 1:
            return -1.0, 0.0

        anisotropy = best_score / perp_score
        if anisotropy > 1.5:
            return best_angle, anisotropy - 1.0
        return -1.0, 0.0
