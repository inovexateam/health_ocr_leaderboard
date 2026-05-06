"""
Health Image OCR Engine
Extracts fitness metrics from any health app screenshot using
Tesseract OCR + OpenCV preprocessing. Zero paid APIs.
"""

import re
import cv2
import numpy as np
import pytesseract
from PIL import Image
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class HealthData:
    steps: Optional[int] = None
    calories: Optional[int] = None
    distance_km: Optional[float] = None
    active_minutes: Optional[int] = None
    sleep_hours: Optional[float] = None
    heart_rate: Optional[int] = None
    floors: Optional[int] = None
    raw_text: str = ""
    confidence: str = "low"   # low / medium / high
    source_app: str = "unknown"

    def to_dict(self):
        return asdict(self)

    def fields_found(self):
        checks = [self.steps, self.calories, self.distance_km,
                  self.active_minutes, self.sleep_hours, self.heart_rate, self.floors]
        return sum(1 for c in checks if c is not None)


# ── Preprocessing variants ────────────────────────────────────────────────────

def _preprocess_variants(img_path: str):
    """Return multiple OpenCV-processed versions of the image for OCR."""
    img = cv2.imread(img_path)
    if img is None:
        img = np.array(Image.open(img_path).convert("RGB"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    # Upscale small images for better OCR
    if max(h, w) < 1000:
        scale = 1000 / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    variants = {}

    # 1. Plain grayscale
    variants["gray"] = gray

    # 2. Otsu thresholding (great for dark-bg apps like Apple Health / Fitbit)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants["otsu"] = otsu

    # 3. Inverted Otsu (for white-on-dark text)
    variants["otsu_inv"] = cv2.bitwise_not(otsu)

    # 4. Adaptive threshold - fine for uneven lighting
    ada = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 10)
    variants["adaptive"] = ada

    # 5. CLAHE + denoise for compressed/blurry screenshots
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=15)
    variants["clahe"] = denoised

    # 6. Sharpened
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, kernel)
    variants["sharp"] = sharp

    return variants


# ── OCR runner ────────────────────────────────────────────────────────────────

TESS_CONFIGS = [
    "--psm 6 -l eng",   # uniform block of text
    "--psm 11 -l eng",  # sparse text
    "--psm 3 -l eng",   # fully auto page segmentation
]


def _run_ocr(img_array) -> str:
    texts = []
    pil = Image.fromarray(img_array)
    for cfg in TESS_CONFIGS:
        try:
            t = pytesseract.image_to_string(pil, config=cfg)
            if t.strip():
                texts.append(t)
        except Exception:
            pass
    # Return longest result (most text captured)
    return max(texts, key=len) if texts else ""


def extract_text_from_image(img_path: str) -> str:
    """Run OCR across all preprocessing variants, return best combined text."""
    variants = _preprocess_variants(img_path)
    all_texts = []
    for name, arr in variants.items():
        t = _run_ocr(arr)
        if t.strip():
            all_texts.append(t)

    if not all_texts:
        return ""

    # Combine unique lines from all variants
    seen = set()
    combined = []
    for text in all_texts:
        for line in text.splitlines():
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                combined.append(line)

    return "\n".join(combined)


# ── Number parsing helpers ────────────────────────────────────────────────────

def _clean_num(s: str) -> str:
    """Remove thousand separators, keep decimal point."""
    return re.sub(r"[,_\s]", "", s)


def _parse_int(s: str) -> Optional[int]:
    try:
        return int(_clean_num(s))
    except Exception:
        return None


def _parse_float(s: str) -> Optional[float]:
    try:
        return float(_clean_num(s))
    except Exception:
        return None


# ── Pattern library ───────────────────────────────────────────────────────────

STEP_PATTERNS = [
    r"(\d[\d,\s]*\d|\d)\s*steps",
    r"steps[:\s]+(\d[\d,\s]*)",
    r"step\s*count[:\s]+(\d[\d,\s]*)",
    r"👟\s*steps?\s*[:\-]?\s*([\d,]+)",   # emoji prefix
    r"daily\s*steps[:\s]*([\d,]+)",
    # Standalone 4-6 digit number on its own line (likely step count)
    # Only as last resort — handled separately below
]

CALORIE_PATTERNS = [
    r"(\d[\d,]*)\s*(?:kcal|cal(?:ories)?)",
    r"(?:calories?|cal)\s*(?:burned|burnt)?\s*[:\-]?\s*(\d[\d,]*)",
    r"active\s*cal(?:ories?)?\s*[:\-]?\s*(\d[\d,]*)",
    r"move[:\s]+(\d[\d,]*)\s*/?\s*\d*\s*cal",
    r"(\d[\d,]*)\s*/\s*\d+\s*cal",   # "487 / 600 cal"
]

DISTANCE_PATTERNS = [
    r"(\d+\.?\d*)\s*km",
    r"(\d+\.?\d*)\s*miles?",
    r"distance[:\s]+(\d+\.?\d*)\s*(km|mi)",
    r"(\d+\.?\d*)\s*mi\b",
]

ACTIVE_MIN_PATTERNS = [
    # Explicit "active time" / "exercise" labels — must match those keywords
    r"active\s*time[:\s]+(\d+)h\s*(\d+)m",
    r"active\s*time[:\s]+(\d+)\s*h(?:r|ours?)?\s*(\d+)\s*m(?:in)?",
    r"exercise[:\s]+(\d+)\s*/?\s*\d*\s*min",
    r"(\d+)\s*min(?:utes?)?\s*(?:active|exercise|workout)",
    r"active[:\s]+(\d+)\s*min",
    # hh:mm style active time with label nearby
    r"active.*?(\d+)h\s*(\d+)m",
]

SLEEP_PATTERNS = [
    r"(\d+)\s*h(?:r|ours?)?\s*(\d+)?\s*m(?:in)?",     # 7h 23m
    r"sleep[:\s]+(\d+)\.?(\d*)\s*(?:h|hrs?|hours?)",
    r"(\d+):(\d+)\s*(?:hrs?|hours?)?\s*(?:sleep|asleep)",
]

HR_PATTERNS = [
    r"(\d{2,3})\s*bpm",
    r"heart\s*rate[:\s]+(\d{2,3})",
    r"avg\s*hr[:\s]+(\d{2,3})",
    r"hr[:\s]+(\d{2,3})",
]

FLOOR_PATTERNS = [
    r"(\d+)\s*floors?",
    r"floors?\s*climbed[:\s]+(\d+)",
]

APP_SIGNATURES = {
    "Apple Health": ["activity", "move", "exercise", "stand", "ring"],
    "Google Fit": ["google fit", "heart points", "google"],
    "Fitbit": ["fitbit", "sleep score", "floors climbed"],
    "Samsung Health": ["samsung health", "samsung"],
    "Garmin": ["garmin", "connect"],
    "Strava": ["strava"],
    "Manual Post": ["good morning", "update", "team", "everyone"],
}


# ── Main extraction function ──────────────────────────────────────────────────

def extract_health_data(img_path: str) -> HealthData:
    raw_text = extract_text_from_image(img_path)
    text_lower = raw_text.lower()
    data = HealthData(raw_text=raw_text[:2000])

    # Detect source app
    for app, keywords in APP_SIGNATURES.items():
        if any(kw in text_lower for kw in keywords):
            data.source_app = app
            break

    # ── Steps ──
    for pat in STEP_PATTERNS:
        m = re.search(pat, text_lower, re.IGNORECASE | re.MULTILINE)
        if m:
            val = _parse_int(m.group(1))
            if val and 100 < val < 100_000:
                data.steps = val
                break

    # Fallback: standalone 4-6 digit number on its own line
    # Only if no step was found yet and "step" appears somewhere in the text
    if not data.steps and "step" in text_lower:
        for line in text_lower.splitlines():
            line = line.strip().replace(",", "").replace(" ", "")
            if re.match(r"^\d{4,6}$", line):
                val = _parse_int(line)
                if val and 100 < val < 100_000:
                    data.steps = val
                    break
    best_cal = None
    for pat in CALORIE_PATTERNS:
        for m in re.finditer(pat, text_lower, re.IGNORECASE):
            val = _parse_int(m.group(1))
            if val and 50 < val < 10_000:
                # Prefer active/burned calories over total
                if best_cal is None or val < best_cal:
                    best_cal = val
    data.calories = best_cal

    # ── Distance ──
    for pat in DISTANCE_PATTERNS:
        m = re.search(pat, text_lower, re.IGNORECASE)
        if m:
            val = _parse_float(m.group(1))
            if val:
                # Convert miles to km if needed
                if len(m.groups()) > 1 and m.group(2) and "mi" in m.group(2):
                    val = round(val * 1.60934, 2)
                if 0.1 < val < 200:
                    data.distance_km = val
                    break

    # ── Active minutes ──
    for pat in ACTIVE_MIN_PATTERNS:
        m = re.search(pat, text_lower, re.IGNORECASE)
        if m:
            groups = [g for g in m.groups() if g is not None]
            if len(groups) >= 2:
                hrs, mins = _parse_int(groups[0]) or 0, _parse_int(groups[1]) or 0
                total = hrs * 60 + mins
                if 0 < total < 1440:
                    data.active_minutes = total
                    break
            elif groups:
                mins = _parse_int(groups[0])
                if mins and 0 < mins < 1440:
                    data.active_minutes = mins
                    break

    # ── Sleep ──
    sleep_m = re.search(r"sleep.*?(\d+)h\s*(\d+)?m?|(\d+)h\s*(\d+)m.*?sleep",
                         text_lower, re.IGNORECASE)
    if sleep_m:
        gs = [g for g in sleep_m.groups() if g is not None]
        if len(gs) >= 2:
            h, m_ = int(gs[0]), int(gs[1])
            data.sleep_hours = round(h + m_ / 60, 2)
        elif gs:
            data.sleep_hours = float(gs[0])

    # ── Heart rate ──
    for pat in HR_PATTERNS:
        m = re.search(pat, text_lower, re.IGNORECASE)
        if m:
            val = _parse_int(m.group(1))
            if val and 30 < val < 250:
                data.heart_rate = val
                break

    # ── Floors ──
    for pat in FLOOR_PATTERNS:
        m = re.search(pat, text_lower, re.IGNORECASE)
        if m:
            val = _parse_int(m.group(1))
            if val and 0 < val < 200:
                data.floors = val
                break

    # ── Confidence scoring ──
    n = data.fields_found()
    if n >= 3:
        data.confidence = "high"
    elif n >= 1:
        data.confidence = "medium"
    else:
        data.confidence = "low"

    return data
