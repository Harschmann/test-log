"""Abstract base class for all camera drivers (Strategy pattern)."""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class CameraBase(ABC):
    """All camera implementations conform to this interface."""

    def __init__(self):
        self._connected = False
        self._name = "Unknown"
        self._mode = "normal"   # "raw" or "normal"

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def mode(self) -> str:
        return self._mode

    # ----- lifecycle -----

    @abstractmethod
    def connect(self) -> bool:
        """Open the camera and start streaming. Returns True on success."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Stop streaming and release the camera."""
        ...

    # ----- mode -----

    def set_mode(self, mode: str) -> None:
        """
        Switch between 'raw' (no processing, sensor-native) and 'normal'
        (white-balanced, debayered, ready for viewing). Default just stores
        the value; subclasses override to actually reconfigure the camera.
        """
        if mode in ("raw", "normal"):
            self._mode = mode

    # ----- frame access -----

    @abstractmethod
    def grab_frame(self) -> Optional[np.ndarray]:
        """
        Live-preview frame as BGR uint8. In normal mode the camera-native
        processing (white balance, gamma, debayer) is applied; in raw mode
        a basic debayer of the sensor data is returned.
        """
        ...

    @abstractmethod
    def grab_raw(self) -> Optional[np.ndarray]:
        """
        Raw sensor frame (highest bit depth, no debayering). For cameras
        without a true raw stream, may equal grab_frame().
        """
        ...

    # ----- info / control -----

    def get_info(self) -> dict:
        return {"name": self._name, "connected": self._connected, "mode": self._mode}

    def set_exposure(self, microseconds: float) -> bool:
        return False

    def get_exposure_range(self) -> tuple:
        return (1000.0, 100000.0)

    def supports_exposure(self) -> bool:
        return False

    # ----- detection -----

    @staticmethod
    @abstractmethod
    def detect() -> bool:
        ...
