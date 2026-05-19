"""Application configuration and constants."""
from pathlib import Path

# Application info
APP_NAME = "VisionPro"
APP_VERSION = "2.0.0"

# Paths -- everything saved under Desktop/VisionPro_Captures
BASE_DIR = Path.home() / "Desktop" / "VisionPro_Captures"
RAW_DIR = BASE_DIR / "raw"
NORMAL_DIR = BASE_DIR / "normal"

# Main window
WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 950
PREVIEW_WIDTH = 1200
PREVIEW_HEIGHT = 750

# Theme
APPEARANCE_MODE = "dark"        # "dark" or "light"
COLOR_THEME = "blue"            # "blue", "green", "dark-blue"

# Capture
DEFAULT_MODE = "normal"         # "raw" or "normal"
JPEG_QUALITY = 95
THUMBNAIL_SIZE = (220, 165)

# Burst capture
BURST_COUNT = 5
BURST_INTERVAL_MS = 200

# Live update target (ms between frames)
LIVE_INTERVAL_MS = 30


def ensure_dirs():
    """Create capture folders on first run."""
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    NORMAL_DIR.mkdir(parents=True, exist_ok=True)
