"""
GenICam/GenTL camera driver via the Harvester library.

This covers a wide range of manufacturing cameras when their GenTL producer
(.cti file) is installed: IMI Tech (NeptuneSDK), Allied Vision (Vimba),
FLIR/Teledyne (Spinnaker), IDS (uEye), Daheng, Hikrobot MVS, etc.
"""
from typing import Optional, List
import os
import numpy as np
import cv2

try:
    from harvesters.core import Harvester
    HARVESTER_AVAILABLE = True
except Exception:
    HARVESTER_AVAILABLE = False

from .base import CameraBase


_COMMON_CTI_PATHS = (
    r"C:\Program Files\Basler\pylon 7\Runtime\x64",
    r"C:\Program Files\Basler\pylon 6\Runtime\x64",
    r"C:\Program Files\MATRIX VISION\mvIMPACT Acquire\bin\x64",
    r"C:\Program Files\Allied Vision\Vimba_6.0\VimbaGigETL\Bin\Win64",
    r"C:\Program Files\Allied Vision\Vimba_6.0\VimbaUSBTL\Bin\Win64",
    r"C:\Program Files\STEMMER IMAGING\Common Vision Blox\GenICam\bin\Win64_x64",
    r"C:\Program Files\Common Files\Daheng\Imaging\GenTL\Win64",
    r"C:\Program Files\IMI Technology\NeptuneCaptureSDK\bin\Win64_x64",
    r"C:\Program Files\MVS\Runtime\Win64_x64",  # Hikrobot
    r"C:\Program Files\Teledyne DALSA\Sapera\Bin\Win64",
)


def _find_cti_files() -> List[str]:
    paths = []
    for k in ("GENICAM_GENTL64_PATH", "GENICAM_GENTL32_PATH"):
        v = os.environ.get(k, "")
        for p in v.split(os.pathsep):
            if p and os.path.isdir(p):
                paths.append(p)
    for p in _COMMON_CTI_PATHS:
        if os.path.isdir(p):
            paths.append(p)

    cti_files = []
    seen = set()
    for directory in paths:
        try:
            for fn in os.listdir(directory):
                if fn.lower().endswith(".cti"):
                    full = os.path.join(directory, fn)
                    if full not in seen:
                        cti_files.append(full)
                        seen.add(full)
        except Exception:
            pass
    return cti_files


def _gray_world_wb(bgr: np.ndarray) -> np.ndarray:
    """Simple gray-world white balance. Software fallback when the camera
    doesn't expose BalanceWhiteAuto via GenICam."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        return bgr
    try:
        b, g, r = cv2.split(bgr.astype(np.float32))
        eps = 1e-6
        b_mean = max(b.mean(), eps)
        g_mean = max(g.mean(), eps)
        r_mean = max(r.mean(), eps)
        avg = (b_mean + g_mean + r_mean) / 3.0
        b = np.clip(b * (avg / b_mean), 0, 255)
        g = np.clip(g * (avg / g_mean), 0, 255)
        r = np.clip(r * (avg / r_mean), 0, 255)
        return cv2.merge((b, g, r)).astype(np.uint8)
    except Exception:
        return bgr


class HarvesterCamera(CameraBase):
    """Generic GenICam camera with raw/normal mode support."""

    def __init__(self):
        super().__init__()
        self._name = "GenICam Camera"
        self._harvester = None
        self._acquirer = None
        self._pixel_format = None
        self._wb_supported = False

    # ----- lifecycle -----

    def connect(self) -> bool:
        if not HARVESTER_AVAILABLE:
            return False
        try:
            self._harvester = Harvester()
            ctis = _find_cti_files()
            if not ctis:
                return False
            for cti in ctis:
                try:
                    self._harvester.add_file(cti)
                except Exception:
                    pass
            self._harvester.update()
            if not self._harvester.device_info_list:
                return False

            self._acquirer = self._harvester.create_image_acquirer(0)

            try:
                info = self._harvester.device_info_list[0]
                vendor = getattr(info, "vendor", "") or ""
                model = getattr(info, "model", "") or ""
                self._name = f"{vendor} {model}".strip() or "GenICam Camera"
            except Exception:
                pass

            # Pick the best pixel format
            try:
                node_map = self._acquirer.remote_device.node_map
                for fmt in ("BayerRG12", "BayerRG8", "Mono12", "Mono8", "RGB8", "BGR8"):
                    try:
                        node_map.PixelFormat.value = fmt
                        self._pixel_format = fmt
                        break
                    except Exception:
                        continue
            except Exception:
                pass

            # Probe for hardware white balance support
            self._wb_supported = self._probe_wb_support()

            # Apply initial mode settings
            self._apply_mode_settings()

            self._acquirer.start()
            self._connected = True
            return True
        except Exception as e:
            print(f"[Harvester] connect failed: {e}")
            self._cleanup()
            return False

    def disconnect(self) -> None:
        self._cleanup()

    def _cleanup(self):
        self._connected = False
        try:
            if self._acquirer is not None:
                try:
                    self._acquirer.stop()
                except Exception:
                    pass
                try:
                    self._acquirer.destroy()
                except Exception:
                    pass
            if self._harvester is not None:
                try:
                    self._harvester.reset()
                except Exception:
                    pass
        finally:
            self._acquirer = None
            self._harvester = None

    # ----- mode handling -----

    def set_mode(self, mode: str) -> None:
        if mode not in ("raw", "normal"):
            return
        super().set_mode(mode)
        self._apply_mode_settings()

    def _probe_wb_support(self) -> bool:
        if self._acquirer is None:
            return False
        try:
            nm = self._acquirer.remote_device.node_map
            _ = nm.BalanceWhiteAuto       # raises if missing
            return True
        except Exception:
            return False

    def _apply_mode_settings(self):
        if self._acquirer is None or not self._wb_supported:
            return
        try:
            nm = self._acquirer.remote_device.node_map
            if self._mode == "normal":
                for value in ("Continuous", "Once"):
                    try:
                        nm.BalanceWhiteAuto.value = value
                        break
                    except Exception:
                        continue
            else:
                try:
                    nm.BalanceWhiteAuto.value = "Off"
                except Exception:
                    pass
        except Exception:
            pass

    # ----- frame access -----

    def _fetch(self) -> Optional[np.ndarray]:
        if not self._connected or self._acquirer is None:
            return None
        try:
            with self._acquirer.fetch(timeout=2.0) as buffer:
                comp = buffer.payload.components[0]
                w, h = comp.width, comp.height
                arr = np.array(comp.data, copy=True).reshape(h, w)
                return arr
        except Exception as e:
            print(f"[Harvester] fetch error: {e}")
            return None

    def grab_frame(self) -> Optional[np.ndarray]:
        arr = self._fetch()
        if arr is None:
            return None
        bgr = self._to_bgr(arr)
        # Software white balance only when the camera doesn't have it
        # AND we're in normal mode AND we got a 3-channel result
        if (self._mode == "normal" and not self._wb_supported
                and bgr.ndim == 3 and bgr.shape[2] == 3):
            bgr = _gray_world_wb(bgr)
        return bgr

    def grab_raw(self) -> Optional[np.ndarray]:
        arr = self._fetch()
        if arr is None:
            return None
        fmt = self._pixel_format or ""
        # MSB-align 12-bit data so saved TIFFs display correctly
        # (see comment in BaslerCamera.grab_raw)
        if arr.dtype == np.uint16 and "12" in fmt:
            arr = arr << 4
        return arr

    def _to_bgr(self, arr: np.ndarray) -> np.ndarray:
        fmt = self._pixel_format or ""
        if arr.dtype == np.uint16:
            arr = (arr >> 4).astype(np.uint8) if "12" in fmt else (arr >> 8).astype(np.uint8)
        if "BayerRG" in fmt:
            return cv2.cvtColor(arr, cv2.COLOR_BayerRG2BGR)
        if "BayerBG" in fmt:
            return cv2.cvtColor(arr, cv2.COLOR_BayerBG2BGR)
        if "Mono" in fmt or arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return arr

    def get_info(self) -> dict:
        info = super().get_info()
        info["pixel_format"] = self._pixel_format
        info["hw_white_balance"] = self._wb_supported
        return info

    @staticmethod
    def detect() -> bool:
        if not HARVESTER_AVAILABLE:
            return False
        if not _find_cti_files():
            return False
        try:
            h = Harvester()
            for cti in _find_cti_files():
                try:
                    h.add_file(cti)
                except Exception:
                    pass
            h.update()
            found = len(h.device_info_list) > 0
            try:
                h.reset()
            except Exception:
                pass
            return found
        except Exception:
            return False
