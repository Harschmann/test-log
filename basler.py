"""Basler industrial camera driver (pypylon)."""
from typing import Optional
import numpy as np
import cv2

try:
    from pypylon import pylon
    PYPYLON_AVAILABLE = True
except ImportError:
    PYPYLON_AVAILABLE = False

from .base import CameraBase


# Preferred pixel formats in order: prefer 12-bit raw for lossless capture.
_PIXEL_FORMAT_PRIORITY = (
    "BayerRG12", "BayerBG12", "BayerGR12", "BayerGB12",
    "BayerRG8",  "BayerBG8",  "BayerGR8",  "BayerGB8",
    "Mono12",    "Mono8",
)


class BaslerCamera(CameraBase):
    """Basler camera via pypylon. Supports auto pixel-format and exposure control."""

    def __init__(self):
        super().__init__()
        self._name = "Basler"
        self._camera = None
        self._pixel_format = None

    # ----- lifecycle -----

    def connect(self) -> bool:
        if not PYPYLON_AVAILABLE:
            return False
        try:
            self._camera = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateFirstDevice()
            )
            self._camera.Open()

            # Pick the best supported pixel format
            for fmt in _PIXEL_FORMAT_PRIORITY:
                try:
                    self._camera.PixelFormat.SetValue(fmt)
                    self._pixel_format = fmt
                    break
                except Exception:
                    continue

            # Update name with model info
            try:
                model = self._camera.GetDeviceInfo().GetModelName()
                self._name = f"Basler {model}"
            except Exception:
                pass

            # Sensible defaults (best-effort, ignore unsupported)
            for setter, val in (("Gain", 0.0), ("Gamma", 1.0)):
                try:
                    getattr(self._camera, setter).SetValue(val)
                except Exception:
                    pass

            self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self._connected = True
            return True
        except Exception as e:
            print(f"[Basler] connect failed: {e}")
            self._cleanup()
            return False

    def disconnect(self) -> None:
        self._cleanup()

    def _cleanup(self):
        self._connected = False
        if self._camera is None:
            return
        try:
            if self._camera.IsGrabbing():
                self._camera.StopGrabbing()
            if self._camera.IsOpen():
                self._camera.Close()
        except Exception as e:
            print(f"[Basler] cleanup error: {e}")
        finally:
            self._camera = None

    # ----- frame access -----

    def _retrieve_array(self, timeout_ms: int = 1000) -> Optional[np.ndarray]:
        if not self._connected or self._camera is None:
            return None
        try:
            res = self._camera.RetrieveResult(timeout_ms, pylon.TimeoutHandling_Return)
            if res is None:
                return None
            try:
                if not res.GrabSucceeded():
                    return None
                # Copy out of pylon's buffer so we can release it immediately
                return np.array(res.Array, copy=True)
            finally:
                res.Release()
        except Exception as e:
            print(f"[Basler] retrieve error: {e}")
            return None

    def grab_frame(self) -> Optional[np.ndarray]:
        arr = self._retrieve_array(1000)
        if arr is None:
            return None
        return self._to_bgr_preview(arr)

    def grab_raw(self) -> Optional[np.ndarray]:
        return self._retrieve_array(5000)

    def _to_bgr_preview(self, arr: np.ndarray) -> np.ndarray:
        """Convert any camera output to BGR uint8 for display."""
        fmt = self._pixel_format or ""

        # 12/16-bit -> 8-bit for preview
        if arr.dtype == np.uint16:
            arr8 = (arr >> 4).astype(np.uint8) if "12" in fmt else (arr >> 8).astype(np.uint8)
        else:
            arr8 = arr

        # Debayer based on pattern
        if "BayerRG" in fmt:
            return cv2.cvtColor(arr8, cv2.COLOR_BayerRG2BGR)
        if "BayerBG" in fmt:
            return cv2.cvtColor(arr8, cv2.COLOR_BayerBG2BGR)
        if "BayerGR" in fmt:
            return cv2.cvtColor(arr8, cv2.COLOR_BayerGR2BGR)
        if "BayerGB" in fmt:
            return cv2.cvtColor(arr8, cv2.COLOR_BayerGB2BGR)
        if "Mono" in fmt:
            return cv2.cvtColor(arr8, cv2.COLOR_GRAY2BGR)

        # Unknown format fallback
        if arr8.ndim == 2:
            return cv2.cvtColor(arr8, cv2.COLOR_GRAY2BGR)
        return arr8

    # ----- info / controls -----

    def get_info(self) -> dict:
        info = super().get_info()
        info["pixel_format"] = self._pixel_format
        if self._camera is not None and self._connected:
            try:
                info["width"] = int(self._camera.Width.GetValue())
                info["height"] = int(self._camera.Height.GetValue())
            except Exception:
                pass
        return info

    def supports_exposure(self) -> bool:
        return self._connected and self._camera is not None

    def _exposure_node(self):
        if self._camera is None:
            return None
        # Different pylon versions use different node names
        for attr in ("ExposureTime", "ExposureTimeAbs"):
            try:
                if hasattr(self._camera, attr):
                    return getattr(self._camera, attr)
            except Exception:
                pass
        return None

    def set_exposure(self, microseconds: float) -> bool:
        node = self._exposure_node()
        if node is None:
            return False
        try:
            node.SetValue(float(microseconds))
            return True
        except Exception as e:
            print(f"[Basler] set_exposure failed: {e}")
            return False

    def get_exposure_range(self) -> tuple:
        node = self._exposure_node()
        if node is None:
            return (1000.0, 100000.0)
        try:
            return (float(node.GetMin()), float(node.GetMax()))
        except Exception:
            return (1000.0, 100000.0)

    # ----- detection -----

    @staticmethod
    def detect() -> bool:
        if not PYPYLON_AVAILABLE:
            return False
        try:
            devices = pylon.TlFactory.GetInstance().EnumerateDevices()
            return len(devices) > 0
        except Exception:
            return False
