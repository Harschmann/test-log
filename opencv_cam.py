"""Generic OpenCV camera driver (USB webcams, UVC-compliant industrial cameras)."""
from typing import Optional, List, Tuple
import numpy as np
import cv2

from .base import CameraBase


class OpenCVCamera(CameraBase):
    """OpenCV-based camera. Works for any UVC/DirectShow device."""

    def __init__(self, device_index: int = 0, name_hint: str = "USB Camera"):
        super().__init__()
        self._device_index = device_index
        self._name = name_hint
        self._cap: Optional[cv2.VideoCapture] = None

    # ----- lifecycle -----

    def connect(self) -> bool:
        try:
            # Try DirectShow first (Windows), then any backend
            for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
                self._cap = cv2.VideoCapture(self._device_index, backend)
                if self._cap.isOpened():
                    break
                self._cap.release()
                self._cap = None

            if self._cap is None or not self._cap.isOpened():
                return False

            # Request highest reasonable resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            # Confirm we can actually read
            ret, _ = self._cap.read()
            if not ret:
                self._cap.release()
                self._cap = None
                return False

            self._name = f"USB Camera (device {self._device_index})"
            self._connected = True
            return True
        except Exception as e:
            print(f"[OpenCV] connect failed: {e}")
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
            return False

    def disconnect(self) -> None:
        self._connected = False
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    # ----- frame access -----

    def grab_frame(self) -> Optional[np.ndarray]:
        if not self._connected or self._cap is None:
            return None
        try:
            ret, frame = self._cap.read()
            if not ret:
                return None
            return frame
        except Exception as e:
            print(f"[OpenCV] grab error: {e}")
            return None

    def grab_raw(self) -> Optional[np.ndarray]:
        # USB cameras give already-processed 8-bit BGR; "raw" == frame
        return self.grab_frame()

    # ----- info -----

    def get_info(self) -> dict:
        info = super().get_info()
        if self._cap is not None and self._connected:
            try:
                info["width"] = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                info["height"] = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                info["fps"] = float(self._cap.get(cv2.CAP_PROP_FPS))
            except Exception:
                pass
        return info

    # ----- detection -----

    @staticmethod
    def detect() -> bool:
        """Probe device 0."""
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            ok = cap.isOpened()
            if ok:
                ret, _ = cap.read()
                ok = ret
            cap.release()
            return ok
        except Exception:
            return False

    @staticmethod
    def enumerate_devices(max_check: int = 5) -> List[Tuple[int, str]]:
        """Return list of (index, name) for available USB cameras."""
        devices = []
        for i in range(max_check):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        devices.append((i, f"USB Camera {i}"))
                cap.release()
            except Exception:
                pass
        return devices
