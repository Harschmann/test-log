"""
Interactive GUI for tuning phone-camera circle detection.

Layout:
  - Top:    sliders for every Hough parameter + a Run Detection button.
  - Bottom: ORIGINAL | DETECTED side by side.

Adjust sliders -> click "Run Detection" -> see the result. Repeat.

Install:
    pip install opencv-python numpy Pillow
    # On Linux you may also need: sudo apt install python3-tk

Run:
    python detect_phone_cameras.py img29.png
or just:
    python detect_phone_cameras.py        # opens with img29.png if present
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk


DISPLAY_MAX = (620, 560)   # max thumbnail size for each panel (w, h)


class CameraDetectorGUI:
    def __init__(self, root, image_path=None):
        self.root = root
        self.root.title("Phone Camera Circle Detector")
        self.root.geometry("1380x920")

        self.original_img = None      # full-res BGR numpy array
        self.params = {}              # tk vars keyed by name
        self.value_labels = {}        # ttk.Labels showing current values

        self._build_ui()

        if image_path and os.path.exists(image_path):
            self._load_image(image_path)

    # ---------- UI construction ---------------------------------------------
    def _build_ui(self):
        # ---- File row -----
        top = ttk.Frame(self.root, padding=(10, 10, 10, 0))
        top.pack(fill=tk.X)

        ttk.Button(top, text="Load image…", command=self._browse).pack(side=tk.LEFT)
        self.file_label = ttk.Label(top, text="(no image)", foreground="#555")
        self.file_label.pack(side=tk.LEFT, padx=10)

        # ---- Sliders -----
        sliders = ttk.LabelFrame(self.root, text="Parameters", padding=10)
        sliders.pack(fill=tk.X, padx=10, pady=8)

        # (key, label, min, max, default, is_int)
        configs = [
            ('param1',          'param1 (Canny upper)',                    10, 250, 60,  True),
            ('param2',          'param2 (accumulator: lower→more circles)', 1, 200, 60,  True),
            ('min_radius_pct',  'min radius  (% of short side)',          0.5,  20, 3.0, False),
            ('max_radius_pct',  'max radius  (% of short side)',          1.0,  40, 12.0, False),
            ('min_dist_pct',    'min distance between circles (%)',         1,  50, 8,   True),
            ('dp_x10',          'dp ×10 (resolution; 12 = 1.2)',            8,  30, 12,  True),
        ]
        for cfg in configs:
            self._add_slider(sliders, *cfg)

        # ---- Mode toggle -----
        mode_row = ttk.Frame(sliders)
        mode_row.pack(fill=tk.X, pady=(8, 2))
        ttk.Label(mode_row, text="Phone body", width=32).pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="black")
        ttk.Radiobutton(mode_row, text="Black (S26 / Ultra)",
                        variable=self.mode_var, value="black").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(mode_row, text="Silver (S25 Edge)",
                        variable=self.mode_var, value="silver").pack(side=tk.LEFT, padx=4)

        # filter toggle
        self.filter_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(mode_row, text="Drop circles with bright interior",
                        variable=self.filter_var).pack(side=tk.LEFT, padx=20)

        # ---- Action row -----
        actions = ttk.Frame(self.root, padding=(10, 0))
        actions.pack(fill=tk.X)

        run_btn = tk.Button(actions, text="▶  Run Detection",
                            command=self.run_detection,
                            bg="#2563eb", fg="white",
                            font=("Helvetica", 11, "bold"),
                            relief=tk.FLAT, padx=18, pady=8)
        run_btn.pack(side=tk.LEFT)

        ttk.Button(actions, text="Reset params",
                   command=self._reset_params).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="Print params",
                   command=self._print_params).pack(side=tk.LEFT, padx=4)

        self.status = ttk.Label(actions, text="Load an image to begin.",
                                foreground="#666")
        self.status.pack(side=tk.LEFT, padx=20)

        # ---- Image panels -----
        panels = ttk.Frame(self.root, padding=10)
        panels.pack(fill=tk.BOTH, expand=True)

        orig_box = ttk.LabelFrame(panels, text="Original", padding=4)
        orig_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        self.orig_label = tk.Label(orig_box, bg="#1f2937")
        self.orig_label.pack(fill=tk.BOTH, expand=True)

        out_box = ttk.LabelFrame(panels, text="Detected", padding=4)
        out_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        self.out_label = tk.Label(out_box, bg="#1f2937")
        self.out_label.pack(fill=tk.BOTH, expand=True)

    def _add_slider(self, parent, key, label, mn, mx, default, is_int):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)

        ttk.Label(row, text=label, width=32).pack(side=tk.LEFT)

        var = tk.IntVar(value=int(default)) if is_int else tk.DoubleVar(value=float(default))
        self.params[key] = var

        value_lbl = ttk.Label(row, width=7, anchor="e",
                              text=(str(int(default)) if is_int else f"{default:.1f}"))
        value_lbl.pack(side=tk.RIGHT, padx=6)
        self.value_labels[key] = value_lbl

        def on_change(v, k=key, integer=is_int, lbl=value_lbl):
            lbl.config(text=str(int(float(v))) if integer else f"{float(v):.1f}")

        ttk.Scale(row, from_=mn, to=mx, variable=var,
                  orient=tk.HORIZONTAL, command=on_change
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

    # ---------- Actions ------------------------------------------------------
    def _browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"),
                       ("All files", "*.*")])
        if path:
            self._load_image(path)

    def _load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            self.status.config(text=f"Failed to read: {path}", foreground="red")
            return
        self.original_img = img
        self.file_label.config(text=f"{os.path.basename(path)}  "
                                    f"({img.shape[1]}×{img.shape[0]})")
        self.status.config(text="Image loaded. Tune sliders and click Run.",
                           foreground="#111")
        self._show(img, self.orig_label)
        self.out_label.config(image="")
        self.out_label.image = None

    def _reset_params(self):
        defaults = {'param1': 60, 'param2': 60, 'min_radius_pct': 3.0,
                    'max_radius_pct': 12.0, 'min_dist_pct': 8, 'dp_x10': 12}
        for k, v in defaults.items():
            self.params[k].set(v)
            lbl = self.value_labels[k]
            lbl.config(text=str(v) if isinstance(v, int) else f"{v:.1f}")
        self.mode_var.set("black")
        self.filter_var.set(True)

    def _print_params(self):
        vals = {k: v.get() for k, v in self.params.items()}
        vals['mode'] = self.mode_var.get()
        vals['filter_bright'] = self.filter_var.get()
        print("Current params:", vals)

    # ---------- Detection ----------------------------------------------------
    def _detect(self):
        img = self.original_img
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.mode_var.get() == "silver":
            proc = cv2.GaussianBlur(gray, (9, 9), 2)
            _, dark = cv2.threshold(proc, 90, 255, cv2.THRESH_BINARY_INV)
            if np.any(dark):
                proc = cv2.bitwise_and(proc, proc, mask=dark)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            proc = cv2.GaussianBlur(clahe.apply(gray), (9, 9), 2)

        short = min(h, w)
        p1 = max(1, int(self.params['param1'].get()))
        p2 = max(1, int(self.params['param2'].get()))
        dp = max(1.0, self.params['dp_x10'].get() / 10.0)
        min_r = max(3, int(short * float(self.params['min_radius_pct'].get()) / 100))
        max_r = max(min_r + 2, int(short * float(self.params['max_radius_pct'].get()) / 100))
        min_d = max(5, int(short * float(self.params['min_dist_pct'].get()) / 100))

        circles = cv2.HoughCircles(
            proc, cv2.HOUGH_GRADIENT, dp=dp,
            minDist=min_d, param1=p1, param2=p2,
            minRadius=min_r, maxRadius=max_r,
        )
        if circles is None:
            return np.empty((0, 3), dtype=int)

        circles = np.round(circles[0, :]).astype(int)

        if self.filter_var.get():
            keep = []
            for x, y, r in circles:
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), max(2, int(r * 0.6)), 255, -1)
                if cv2.mean(gray, mask=mask)[0] < 160:
                    keep.append((x, y, r))
            circles = np.array(keep, dtype=int) if keep else np.empty((0, 3), dtype=int)

        if len(circles):
            circles = np.array(sorted(circles.tolist(),
                                      key=lambda c: (c[1] // 30, c[0])))
        return circles

    def run_detection(self):
        if self.original_img is None:
            self.status.config(text="Load an image first.", foreground="red")
            return

        circles = self._detect()
        annotated = self.original_img.copy()
        for i, (x, y, r) in enumerate(circles):
            cv2.circle(annotated, (x, y), r, (0, 255, 0), 3)
            cv2.circle(annotated, (x, y), 3, (0, 0, 255), 4)
            cv2.putText(annotated, str(i + 1), (x - 12, y - r - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        self._show(annotated, self.out_label)
        n = len(circles)
        color = "green" if 0 < n <= 8 else ("orange" if n else "red")
        self.status.config(text=f"Detected {n} circle(s)", foreground=color)

    # ---------- Helpers ------------------------------------------------------
    def _show(self, bgr, widget):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail(DISPLAY_MAX, Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil)
        widget.config(image=tk_img)
        widget.image = tk_img  # keep reference so it isn't garbage-collected


if __name__ == "__main__":
    initial = sys.argv[1] if len(sys.argv) > 1 else "img29.png"
    root = tk.Tk()
    CameraDetectorGUI(root, initial if os.path.exists(initial) else None)
    root.mainloop()
