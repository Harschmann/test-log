"""Image viewer window with tabs for raw vs normal captures."""
import os
import subprocess
import sys
from pathlib import Path
from typing import List
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np

from config import RAW_DIR, NORMAL_DIR, THUMBNAIL_SIZE


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp")


def _open_in_explorer(path: Path):
    """Open a directory in the OS file manager."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        print(f"Failed to open folder: {e}")


def _load_image_for_display(filepath: Path, is_raw_folder: bool) -> np.ndarray:
    """Load an image as a BGR uint8 numpy array suitable for display."""
    arr = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Could not read {filepath}")
    # 16-bit -> 8-bit
    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)
    # Single-channel: assume Bayer if from raw folder, else gray
    if arr.ndim == 2:
        if is_raw_folder:
            arr = cv2.cvtColor(arr, cv2.COLOR_BayerRG2BGR)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return arr


class ViewerWindow(ctk.CTkToplevel):
    """Tabbed gallery: Normal Images / Raw Images. Click thumbnail to view full size."""

    def __init__(self, master):
        super().__init__(master)
        self.title("Captured Images Viewer")
        self.geometry("1150x780")
        self.minsize(800, 500)

        # Tabbed interface
        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_normal = self.tabs.add("📷  Normal Images")
        self.tab_raw = self.tabs.add("⚙  Raw Images")

        self._scrolls = {}
        self._dirs = {"normal": NORMAL_DIR, "raw": RAW_DIR}
        self._thumb_refs = {"normal": [], "raw": []}

        self._build_tab(self.tab_normal, "normal")
        self._build_tab(self.tab_raw, "raw")

        # Bottom action bar
        bottom = ctk.CTkFrame(self, height=50)
        bottom.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkButton(bottom, text="🔄  Refresh", width=130, height=34,
                      command=self._refresh_all).pack(side="left", padx=5, pady=8)
        ctk.CTkButton(bottom, text="📁  Open Folder", width=140, height=34,
                      command=self._open_current_folder).pack(side="left", padx=5, pady=8)
        self.count_lbl = ctk.CTkLabel(bottom, text="", font=ctk.CTkFont(size=12))
        self.count_lbl.pack(side="left", padx=20)

        ctk.CTkButton(bottom, text="Close", width=100, height=34,
                      fg_color="#6e6e6e", hover_color="#525252",
                      command=self.destroy).pack(side="right", padx=5, pady=8)

        self._refresh_all()

    def _build_tab(self, tab, key: str):
        directory = self._dirs[key]

        # Header showing the directory path
        header = ctk.CTkLabel(
            tab, text=f"📂  {directory}",
            font=ctk.CTkFont(size=11), text_color="gray70", anchor="w",
        )
        header.pack(anchor="w", padx=10, pady=(6, 4), fill="x")

        scroll = ctk.CTkScrollableFrame(tab)
        scroll.pack(fill="both", expand=True, padx=5, pady=5)
        self._scrolls[key] = scroll

    def _list_files(self, directory: Path) -> List[Path]:
        if not directory.exists():
            return []
        try:
            files = [f for f in directory.iterdir()
                     if f.is_file() and f.suffix.lower() in _IMAGE_EXTS]
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            return files
        except Exception as e:
            print(f"List files error: {e}")
            return []

    def _refresh_all(self):
        self._populate("normal")
        self._populate("raw")
        n = len(self._list_files(NORMAL_DIR))
        r = len(self._list_files(RAW_DIR))
        self.count_lbl.configure(text=f"Total — Normal: {n}   Raw: {r}")

    def _populate(self, key: str):
        scroll = self._scrolls[key]
        directory = self._dirs[key]

        # Clear existing thumbnails
        for child in scroll.winfo_children():
            child.destroy()
        self._thumb_refs[key] = []

        files = self._list_files(directory)
        if not files:
            ctk.CTkLabel(
                scroll, text="No images yet.\nCapture some from the main window.",
                font=ctk.CTkFont(size=13), text_color="gray60",
            ).pack(pady=40)
            return

        cols = 4
        for i, f in enumerate(files):
            row, col = divmod(i, cols)
            self._add_thumbnail(scroll, f, row, col, key)

    def _add_thumbnail(self, parent, filepath: Path, row: int, col: int, key: str):
        try:
            is_raw = (key == "raw")
            arr = _load_image_for_display(filepath, is_raw)
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img.thumbnail(THUMBNAIL_SIZE)
            photo = ImageTk.PhotoImage(img)
            self._thumb_refs[key].append(photo)

            frame = ctk.CTkFrame(parent)
            frame.grid(row=row, column=col, padx=8, pady=8)

            btn = ctk.CTkButton(
                frame, image=photo, text="",
                width=THUMBNAIL_SIZE[0] + 12, height=THUMBNAIL_SIZE[1] + 12,
                fg_color="transparent", hover_color=("gray80", "gray20"),
                command=lambda p=filepath, r=is_raw: self._open_full(p, r),
            )
            btn.pack(padx=3, pady=3)

            name = filepath.name
            if len(name) > 28:
                name = name[:25] + "..."
            label = ctk.CTkLabel(
                frame, text=name, font=ctk.CTkFont(size=10),
                wraplength=THUMBNAIL_SIZE[0],
            )
            label.pack(padx=3, pady=(0, 6))
        except Exception as e:
            print(f"Thumbnail error for {filepath.name}: {e}")

    def _open_full(self, filepath: Path, is_raw: bool):
        FullImageWindow(self, filepath, is_raw)

    def _open_current_folder(self):
        current = self.tabs.get()
        folder = RAW_DIR if "Raw" in current else NORMAL_DIR
        _open_in_explorer(folder)


class FullImageWindow(ctk.CTkToplevel):
    """Single full-size image viewer."""

    def __init__(self, master, filepath: Path, is_raw: bool):
        super().__init__(master)
        self.title(filepath.name)
        self.geometry("1050x800")
        self._photo = None

        try:
            arr = _load_image_for_display(filepath, is_raw)
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img.thumbnail((1000, 720))
            self._photo = ImageTk.PhotoImage(img)

            label = ctk.CTkLabel(self, image=self._photo, text="")
            label.pack(expand=True, padx=10, pady=10)

            try:
                size_kb = filepath.stat().st_size // 1024
            except Exception:
                size_kb = 0
            info_text = (f"{filepath.name}   |   "
                         f"{arr.shape[1]} x {arr.shape[0]}   |   "
                         f"{size_kb:,} KB")
            ctk.CTkLabel(self, text=info_text, font=ctk.CTkFont(size=11)).pack(pady=(0, 10))
        except Exception as e:
            ctk.CTkLabel(self, text=f"Error loading image:\n{e}",
                         font=ctk.CTkFont(size=14)).pack(expand=True, padx=20, pady=20)
