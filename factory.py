"""Factory + auto-detection for camera drivers."""
from typing import Optional, List
from .base import CameraBase
from .basler import BaslerCamera
from .opencv_cam import OpenCVCamera

try:
    from .harvester_cam import HarvesterCamera, HARVESTER_AVAILABLE
except Exception:
    HARVESTER_AVAILABLE = False
    HarvesterCamera = None  # type: ignore


class CameraFactory:
    """
    Factory pattern: try cameras in priority order.

    Order matters - we try purpose-built industrial drivers first
    (Basler -> GenICam fallback for IMI Tech, Allied Vision, FLIR, etc.)
    and finally fall back to OpenCV for USB webcams.
    """

    PRIORITY = ("Basler", "GenICam", "USB")

    @staticmethod
    def list_available() -> List[str]:
        """Quick detection - returns names of detected camera types."""
        available = []
        if BaslerCamera.detect():
            available.append("Basler")
        if HARVESTER_AVAILABLE and HarvesterCamera is not None and HarvesterCamera.detect():
            available.append("GenICam")
        if OpenCVCamera.detect():
            available.append("USB")
        return available

    @staticmethod
    def create(camera_type: str) -> Optional[CameraBase]:
        """Create a camera of the given type without connecting."""
        t = camera_type.lower().strip()
        if t == "basler":
            return BaslerCamera()
        if t in ("genicam", "imi", "imi tech", "imitech", "harvester",
                 "allied vision", "flir", "ids", "hikrobot", "daheng"):
            if HARVESTER_AVAILABLE and HarvesterCamera is not None:
                return HarvesterCamera()
            return None
        if t in ("usb", "webcam", "opencv", "uvc"):
            return OpenCVCamera()
        return None

    @staticmethod
    def auto_detect_and_create() -> Optional[CameraBase]:
        """
        Try each camera type in priority order; return the first one
        that connects successfully.
        """
        for cam_type in CameraFactory.PRIORITY:
            cam = CameraFactory.create(cam_type)
            if cam is None:
                continue
            if cam.connect():
                return cam
            # Connect failed; ensure cleanup before trying next
            try:
                cam.disconnect()
            except Exception:
                pass
        return None
