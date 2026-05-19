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


# Common Windows install locations of GenTL producer .cti files
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
    """Scan environment + common install paths for GenTL producer (.cti) files."""
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


class HarvesterCamera(CameraBase):
    """Generic GenICam camera using the Harvester library."""

    def __init__(self):
        super().__init__()
        self._name = "GenICam Camera"
        self._harvester = None
        self._acquirer = None
        self._pixel_format = None

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

            # Identify camera
            try:
                info = self._harvester.device_info_list[0]
                vendor = getattr(info, "vendor", "") or ""
                model = getattr(info, "model", "") or ""
                self._name = f"{vendor} {model}".strip() or "GenICam Camera"
            except Exception:
                pass

            # Try preferred pixel format (best effort)
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
        return self._to_bgr(arr)

    def grab_raw(self) -> Optional[np.ndarray]:
        return self._fetch()

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
