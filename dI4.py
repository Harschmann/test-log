"""
Generalized circular-region detector with interactive tuning GUI.

Pipeline (no model-specific assumptions):
    grayscale -> blur -> threshold (Otsu or manual) -> morphology close
    -> find contours -> filter by area + circularity -> annotate

Three panels: Original | Binary mask | Detected
A "Show crops" button opens a popup with each detected region cropped out.

Install:
    pip install opencv-python numpy Pillow
    # Linux may also need: sudo apt install python3-tk

Run:
    python detect_phone_cameras.py img29.png
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk


DISPLAY_MAX = (420, 460)   # per-panel max thumbnail size (w, h)

# Resampling filter (PIL 9.1+ uses Resampling enum; older uses module attr)
RESAMPLE = (Image.Resampling.LANCZOS
            if hasattr(Image, "Resampling") else Image.LANCZOS)


# ----------------------------------------------------------------------------
# Pure CV functions — no GUI dependencies
# ----------------------------------------------------------------------------

def preprocess(img_bgr, *, blur, threshold, use_otsu, invert, morph_close):
    """Return binary mask: grayscale -> blur -> threshold -> morph close."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    k = max(1, int(blur))
    if k % 2 == 0:
        k += 1
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if use_otsu:
        _, binary = cv2.threshold(gray, 0, 255, flag | cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, int(threshold), 255, flag)

    mc = int(morph_close)
    if mc > 0:
        ks = mc * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


def find_circular_blobs(binary, *, min_area_pct, max_area_pct, min_circularity):
    """Find blobs that pass area + circularity filters.

    Returns list of (x, y, r, circularity) sorted top-to-bottom, left-to-right.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)

    H, W = binary.shape
    img_area = float(H * W)
    a_min = img_area * float(min_area_pct) / 100.0
    a_max = img_area * float(max_area_pct) / 100.0
    c_min = float(min_circularity)

    found = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < a_min or area > a_max:
            continue
        perim = cv2.arcLength(c, True)
        if perim <= 0:
            continue
        circ = 4.0 * np.pi * area / (perim * perim)
        if circ < c_min:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        found.append((int(x), int(y), int(r), float(circ)))

    found.sort(key=lambda v: (v[1] // 30, v[0]))
    return found


def crop_circles(img_bgr, circles, padding_pct=0.15):
    """Crop a square region around each circle with `padding_pct` extra space."""
    H, W = img_bgr.shape[:2]
    crops = []
    for x, y, r, _ in circles:
        pad = int(r * padding_pct)
        x1, y1 = max(0, x - r - pad), max(0, y - r - pad)
        x2, y2 = min(W, x + r + pad), min(H, y + r + pad)
        crops.append(img_bgr[y1:y2, x1:x2])
    return crops


def annotate(img_bgr, circles):
    """Draw detected circles + indices on a copy of the image."""
    out = img_bgr.copy()
    for i, (x, y, r, _) in enumerate(circles, 1):
        cv2.circle(out, (x, y), r, (0, 255, 0), 3)
        cv2.circle(out, (x, y), 3, (0, 0, 255), 4)
        cv2.putText(out, str(i), (x - 12, y - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return out


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------

class DetectorGUI:
    # (key, label, min, max, default, is_int)
    SLIDER_CONFIGS = [
        ('blur',            'blur kernel (odd; 1 = none)',                1,    21,   5,     True),
        ('threshold',       'threshold (used only if Otsu off)',          0,    255,  80,    True),
        ('morph_close',     'morphology close (0 = off)',                 0,    15,   2,     True),
        ('min_area_pct',    'min blob area  (% of image)',                0.001, 5.0, 0.01,  False),
        ('max_area_pct',    'max blob area  (% of image)',                0.05,  20.0, 5.0,  False),
        ('min_circularity', 'min circularity (1.0 = perfect circle)',     0.30,  1.00, 0.70, False),
    ]

    def __init__(self, root, image_path=None):
        self.root = root
        self.root.title("Generalized Circle Detector")
        self.root.geometry("1450x940")

        self.original = None         # full-res BGR ndarray
        self.last_circles = []
        self.params = {}             # slider tk Vars by key
        self.value_labels = {}       # (label_widget, is_int) by key
        self.crops_window = None

        self._build_ui()
        if image_path and os.path.exists(image_path):
            self._load(image_path)

    # ---- UI construction ---------------------------------------------------
    def _build_ui(self):
        # File row
        top = ttk.Frame(self.root, padding=(10, 10, 10, 4))
        top.pack(fill=tk.X)
        ttk.Button(top, text="Load image…", command=self._browse).pack(side=tk.LEFT)
        self.file_label = ttk.Label(top, text="(no image)", foreground="#555")
        self.file_label.pack(side=tk.LEFT, padx=10)

        # Parameter sliders
        sliders_box = ttk.LabelFrame(self.root, text="Parameters", padding=10)
        sliders_box.pack(fill=tk.X, padx=10, pady=6)
        for cfg in self.SLIDER_CONFIGS:
            self._add_slider(sliders_box, *cfg)

        # Checkboxes
        cb = ttk.Frame(sliders_box)
        cb.pack(fill=tk.X, pady=(8, 2))
        self.use_otsu = tk.BooleanVar(value=True)
        self.invert = tk.BooleanVar(value=True)
        ttk.Checkbutton(cb, text="Use Otsu auto-threshold",
                        variable=self.use_otsu).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(cb, text="Invert  (target dark regions)",
                        variable=self.invert).pack(side=tk.LEFT, padx=20)

        # Action buttons
        actions = ttk.Frame(self.root, padding=(10, 0))
        actions.pack(fill=tk.X)
        tk.Button(actions, text="▶  Run Detection", command=self.run,
                  bg="#2563eb", fg="white",
                  font=("Helvetica", 11, "bold"),
                  relief=tk.FLAT, padx=18, pady=8).pack(side=tk.LEFT)
        ttk.Button(actions, text="Reset", command=self._reset).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="Print params", command=self._print).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Show crops", command=self._show_crops).pack(side=tk.LEFT, padx=4)
        self.status = ttk.Label(actions, text="Load an image to begin.",
                                foreground="#666")
        self.status.pack(side=tk.LEFT, padx=20)

        # Three image panels
        panels = ttk.Frame(self.root, padding=10)
        panels.pack(fill=tk.BOTH, expand=True)
        self.orig_lbl = self._make_panel(panels, "Original")
        self.bin_lbl  = self._make_panel(panels, "Binary (after threshold)")
        self.out_lbl  = self._make_panel(panels, "Detected")

    def _make_panel(self, parent, title):
        box = ttk.LabelFrame(parent, text=title, padding=4)
        box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        lbl = tk.Label(box, bg="#1f2937")
        lbl.pack(fill=tk.BOTH, expand=True)
        return lbl

    def _add_slider(self, parent, key, label, mn, mx, default, is_int):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)

        ttk.Label(row, text=label, width=36).pack(side=tk.LEFT)

        var = tk.IntVar(value=int(default)) if is_int else tk.DoubleVar(value=float(default))
        self.params[key] = var

        value_lbl = ttk.Label(row, width=8, anchor="e",
                              text=self._fmt(default, is_int))
        value_lbl.pack(side=tk.RIGHT, padx=6)
        self.value_labels[key] = (value_lbl, is_int)

        def on_change(v, lbl=value_lbl, integer=is_int):
            lbl.config(text=self._fmt(float(v), integer))

        ttk.Scale(row, from_=mn, to=mx, variable=var,
                  orient=tk.HORIZONTAL, command=on_change
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

    @staticmethod
    def _fmt(value, is_int):
        if is_int:
            return str(int(value))
        return f"{float(value):.4g}"

    # ---- Actions -----------------------------------------------------------
    def _browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"),
                       ("All files", "*.*")])
        if path:
            self._load(path)

    def _load(self, path):
        img = cv2.imread(path)
        if img is None:
            self.status.config(text=f"Failed to read {path}", foreground="red")
            return
        self.original = img
        self.file_label.config(
            text=f"{os.path.basename(path)}  ({img.shape[1]}×{img.shape[0]})")
        self.status.config(text="Tune sliders → click Run Detection.",
                           foreground="#111")
        self._show(img, self.orig_lbl)
        for lbl in (self.bin_lbl, self.out_lbl):
            lbl.config(image=""); lbl.image = None
        self.last_circles = []

    def _gather(self):
        return dict(
            blur=int(self.params['blur'].get()),
            threshold=int(self.params['threshold'].get()),
            use_otsu=bool(self.use_otsu.get()),
            invert=bool(self.invert.get()),
            morph_close=int(self.params['morph_close'].get()),
            min_area_pct=float(self.params['min_area_pct'].get()),
            max_area_pct=float(self.params['max_area_pct'].get()),
            min_circularity=float(self.params['min_circularity'].get()),
        )

    def run(self):
        if self.original is None:
            self.status.config(text="Load an image first.", foreground="red")
            return

        p = self._gather()
        binary = preprocess(self.original,
                            blur=p['blur'], threshold=p['threshold'],
                            use_otsu=p['use_otsu'], invert=p['invert'],
                            morph_close=p['morph_close'])
        circles = find_circular_blobs(binary,
                                      min_area_pct=p['min_area_pct'],
                                      max_area_pct=p['max_area_pct'],
                                      min_circularity=p['min_circularity'])
        annotated = annotate(self.original, circles)
        self.last_circles = circles

        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self._show(self.original, self.orig_lbl)
        self._show(binary_bgr, self.bin_lbl)
        self._show(annotated, self.out_lbl)

        n = len(circles)
        color = "green" if 0 < n <= 12 else ("orange" if n else "red")
        self.status.config(text=f"Detected {n} circle(s)", foreground=color)

    def _reset(self):
        for cfg in self.SLIDER_CONFIGS:
            key, _label, _mn, _mx, default, is_int = cfg
            self.params[key].set(int(default) if is_int else float(default))
            lbl, _ = self.value_labels[key]
            lbl.config(text=self._fmt(default, is_int))
        self.use_otsu.set(True)
        self.invert.set(True)

    def _print(self):
        p = self._gather()
        print("Current params:")
        for k, v in p.items():
            print(f"  {k:>17} = {v}")
        if self.last_circles:
            print(f"Detected circles ({len(self.last_circles)}):")
            for i, (x, y, r, c) in enumerate(self.last_circles, 1):
                print(f"  #{i}: center=({x},{y})  r={r}  circularity={c:.3f}")

    def _show_crops(self):
        if self.original is None or not self.last_circles:
            self.status.config(text="No detections to crop.", foreground="red")
            return

        crops = crop_circles(self.original, self.last_circles)

        if self.crops_window is not None and self.crops_window.winfo_exists():
            for w in self.crops_window.winfo_children():
                w.destroy()
        else:
            self.crops_window = tk.Toplevel(self.root)
            self.crops_window.title("Cropped regions")

        cols = min(max(len(crops), 1), 6)
        for i, crop in enumerate(crops):
            r, c = divmod(i, cols)
            frame = ttk.LabelFrame(self.crops_window, text=f"#{i+1}", padding=3)
            frame.grid(row=r, column=c, padx=4, pady=4)

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            pil.thumbnail((200, 200), RESAMPLE)
            tk_img = ImageTk.PhotoImage(pil)
            lbl = tk.Label(frame, image=tk_img)
            lbl.image = tk_img
            lbl.pack()

    # ---- Image helper ------------------------------------------------------
    def _show(self, bgr, widget):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail(DISPLAY_MAX, RESAMPLE)
        tk_img = ImageTk.PhotoImage(pil)
        widget.config(image=tk_img)
        widget.image = tk_img  # keep reference


if __name__ == "__main__":
    initial = sys.argv[1] if len(sys.argv) > 1 else "img29.png"
    root = tk.Tk()
    DetectorGUI(root, initial if os.path.exists(initial) else None)
    root.mainloop()
