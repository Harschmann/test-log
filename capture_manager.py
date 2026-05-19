"""Capture / save management."""
from datetime import datetime
from typing import Tuple
import numpy as np
import cv2

from config import RAW_DIR, NORMAL_DIR, JPEG_QUALITY


class CaptureManager:
    """Handles saving captured frames to disk, with separate folders for raw/normal."""

    _RAW_EXTS = ("*.tiff", "*.tif", "*.png", "*.bmp")
    _NORMAL_EXTS = ("*.jpg", "*.jpeg", "*.png")

    def __init__(self):
        self._counter = {"raw": 0, "normal": 0}
        self._refresh_counters()

    def _refresh_counters(self):
        try:
            self._counter["raw"] = sum(len(list(RAW_DIR.glob(p))) for p in self._RAW_EXTS)
            self._counter["normal"] = sum(len(list(NORMAL_DIR.glob(p))) for p in self._NORMAL_EXTS)
        except Exception:
            pass

    @property
    def raw_count(self) -> int:
        return self._counter["raw"]

    @property
    def normal_count(self) -> int:
        return self._counter["normal"]

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    def save_raw(self, arr: np.ndarray, camera_name: str = "") -> Tuple[bool, str]:
        """
        Save a raw frame as a lossless TIFF.
        - Basler 12-bit Bayer: stored as 16-bit single-channel TIFF (12 bits used).
        - Other: stored as-is.
        """
        if arr is None:
            return False, "No data"
        try:
            path = RAW_DIR / f"raw_{self._timestamp()}.tiff"
            ok = cv2.imwrite(str(path), arr, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            if ok:
                self._counter["raw"] += 1
                return True, str(path)
            return False, "imwrite returned False"
        except Exception as e:
            return False, str(e)

    def save_normal(self, bgr: np.ndarray, camera_name: str = "",
                    fmt: str = "jpg") -> Tuple[bool, str]:
        """Save processed BGR frame as JPEG (default) or PNG."""
        if bgr is None:
            return False, "No data"
        try:
            ext = fmt.lower().strip(".")
            path = NORMAL_DIR / f"img_{self._timestamp()}.{ext}"
            if ext in ("jpg", "jpeg"):
                ok = cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            else:
                ok = cv2.imwrite(str(path), bgr)
            if ok:
                self._counter["normal"] += 1
                return True, str(path)
            return False, "imwrite returned False"
        except Exception as e:
            return False, str(e)
