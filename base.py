"""Abstract base class for all camera drivers (Strategy pattern)."""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class CameraBase(ABC):
    """All camera implementations conform to this interface."""

    def __init__(self):
        self._connected = False
        self._name = "Unknown"

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ----- lifecycle -----

    @abstractmethod
    def connect(self) -> bool:
        """Open the camera and start streaming. Returns True on success."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Stop streaming and release the camera."""
        ...

    # ----- frame access -----

    @abstractmethod
    def grab_frame(self) -> Optional[np.ndarray]:
        """
        Grab the latest frame for live preview.
        Returns BGR uint8 array suitable for display, or None on failure.
        """
        ...

    @abstractmethod
    def grab_raw(self) -> Optional[np.ndarray]:
        """
        Grab a raw frame (highest bit depth, no debayering).
        Returns the raw sensor array, or None on failure.
        For cameras without true raw support, this may equal grab_frame().
        """
        ...

    # ----- info / control -----

    def get_info(self) -> dict:
        """Return camera info dict (overridable)."""
        return {"name": self._name, "connected": self._connected}

    def set_exposure(self, microseconds: float) -> bool:
        """Set exposure time in microseconds. Returns True if supported."""
        return False

    def get_exposure_range(self) -> tuple:
        """Return (min, max) supported exposure time. Default range if not supported."""
        return (1000.0, 100000.0)

    def supports_exposure(self) -> bool:
        return False

    # ----- detection -----

    @staticmethod
    @abstractmethod
    def detect() -> bool:
        """Check if any camera of this type is plugged in and reachable."""
        ...
