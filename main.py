"""VisionPro — Industrial camera capture application.

Run with:  python main.py
"""
import sys
import traceback

from config import ensure_dirs
from ui import MainWindow


def main():
    try:
        ensure_dirs()
        app = MainWindow()
        app.mainloop()
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
