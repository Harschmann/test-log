"""Basler industrial camera driver (pypylon) with raw/normal mode support."""
from typing import Optional
import numpy as np
import cv2

try:
    from pypylon import pylon
    PYPYLON_AVAILABLE = True
except ImportError:
    PYPYLON_AVAILABLE = False

from .base import CameraBase


# Preferred pixel formats in order of priority.
# Bayer 12-bit gives us lossless sensor data for raw mode AND maximum quality
# input for the converter in normal mode.
_PIXEL_FORMAT_PRIORITY = (
    "BayerRG12", "BayerBG12", "BayerGR12", "BayerGB12",
    "BayerRG8",  "BayerBG8",  "BayerGR8",  "BayerGB8",
    "Mono12",    "Mono8",
)


class BaslerCamera(CameraBase):
    """
    Basler camera via pypylon.

    Mode behaviour:
      - normal: BalanceWhiteAuto ON, BGR8 output via pylon's ImageFormatConverter
                (proper colors, ready-to-use).
      - raw   : All color processing OFF, returns sensor-native Bayer/Mono data
                (uint16 for 12-bit formats) for lossless saves.
    """

    def __init__(self):
        super().__init__()
        self._name = "Basler"
        self._camera = None
        self._pixel_format = None
        self._converter = None         # pylon.ImageFormatConverter

    # ----- lifecycle -----

    def connect(self) -> bool:
        if not PYPYLON_AVAILABLE:
            return False
        try:
            self._camera = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateFirstDevice()
            )
            self._camera.Open()

            # Pick best supported sensor-native pixel format
            for fmt in _PIXEL_FORMAT_PRIORITY:
                try:
                    self._camera.PixelFormat.SetValue(fmt)
                    self._pixel_format = fmt
                    break
                except Exception:
                    continue

            # Identify model
            try:
                model = self._camera.GetDeviceInfo().GetModelName()
                self._name = f"Basler {model}"
            except Exception:
                pass

            # Set up pylon's image converter for normal-mode output
            self._converter = pylon.ImageFormatConverter()
            self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            # Default Gain=0; gamma is set per-mode below
            try:
                self._camera.Gain.SetValue(0.0)
            except Exception:
                pass

            # Apply current mode's settings (default is "normal")
            self._apply_mode_settings()

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
            self._converter = None

    # ----- mode handling -----

    def set_mode(self, mode: str) -> None:
        if mode not in ("raw", "normal") or mode == self._mode:
            super().set_mode(mode)
            return
        super().set_mode(mode)
        self._apply_mode_settings()

    def _apply_mode_settings(self):
        """Configure the camera for the current mode."""
        if self._camera is None:
            return
        if self._mode == "normal":
            # Auto white balance for natural colors
            for value in ("Continuous", "Once"):
                try:
                    self._camera.BalanceWhiteAuto.SetValue(value)
                    break
                except Exception:
                    continue
            # Default gamma -- some cameras default to 1.0 which looks flat;
            # leave it alone, the converter handles the rest.
        else:
            # Raw mode: disable any in-camera color processing
            try:
                self._camera.BalanceWhiteAuto.SetValue("Off")
            except Exception:
                pass
            try:
                self._camera.Gamma.SetValue(1.0)
            except Exception:
                pass

    # ----- frame access -----

    def _retrieve(self, timeout_ms: int = 1000):
        """Retrieve a pylon GrabResult. Caller is responsible for Release()."""
        if not self._connected or self._camera is None:
            return None
        try:
            res = self._camera.RetrieveResult(timeout_ms, pylon.TimeoutHandling_Return)
            if res is None:
                return None
            if not res.GrabSucceeded():
                res.Release()
                return None
            return res
        except Exception as e:
            print(f"[Basler] retrieve error: {e}")
            return None

    def grab_frame(self) -> Optional[np.ndarray]:
        """
        Live-preview / normal-save frame. In normal mode, uses pylon's
        ImageFormatConverter to apply white balance + debayer and return
        proper BGR8. In raw mode, does a basic debayer of the raw array
        so the preview still shows something.
        """
        res = self._retrieve(1000)
        if res is None:
            return None
        try:
            if self._mode == "normal" and self._converter is not None:
                try:
                    converted = self._converter.Convert(res)
                    bgr = converted.GetArray()        # uint8 (H, W, 3) BGR
                    return np.array(bgr, copy=True)
                except Exception as e:
                    print(f"[Basler] converter failed, falling back: {e}")
            # Raw mode (or converter fallback): basic debayer of sensor data
            arr = np.array(res.Array, copy=True)
            return self._basic_debayer(arr)
        finally:
            res.Release()

    def grab_raw(self) -> Optional[np.ndarray]:
        """Sensor-native frame (uint16 for 12-bit Bayer, uint8 for 8-bit)."""
        res = self._retrieve(5000)
        if res is None:
            return None
        try:
            return np.array(res.Array, copy=True)
        finally:
            res.Release()

    def _basic_debayer(self, arr: np.ndarray) -> np.ndarray:
        """Convert raw sensor array to BGR uint8 with NO white balance."""
        fmt = self._pixel_format or ""

        if arr.dtype == np.uint16:
            arr8 = (arr >> 4).astype(np.uint8) if "12" in fmt else (arr >> 8).astype(np.uint8)
        else:
            arr8 = arr

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
