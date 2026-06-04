"""
Dust Detection App
------------------
Workflow:
  1. Load Image
  2. Type a radius, then CLICK on each camera center (left panel).
     - Each click adds a center using the current radius value.
     - Change the radius box anytime before clicking a different-sized camera.
  3. Undo Last / Clear All to fix mistakes.
  4. Detect -> result shows on the right (before vs after).
  5. Save Result -> writes a before|after image to the results folder.
  6. Open Folder -> opens that folder so you don't dig through files.

Detection logic is the two-pass method (bright dust + faint dust via tophat).
Tune the Detection Settings row if a new phone model needs it.

Requires: pip install opencv-python pillow numpy
"""

import os
import sys
import math
import subprocess
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk


CANVAS_W = 520
CANVAS_H = 560
SAVE_DIR = "dust_results"


def run_detection(img, centers, p):
    """centers: list of (x, y, r). p: dict of params. Returns (result_img, count)."""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for (x, y, r) in centers:
        cv2.circle(mask, (x, y), r, 255, -1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    result = img.copy()
    count = 0
    big_detected = np.zeros(img.shape[:2], dtype=np.uint8)

    # ---- Pass A: bright / big dust (no shape filter) ----
    _, bb = cv2.threshold(blurred, p["bright"], 255, cv2.THRESH_BINARY)
    bb = cv2.bitwise_and(bb, mask)
    cont, _ = cv2.findContours(bb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cont:
        a = cv2.contourArea(c)
        if a < p["min_area"]:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        cv2.circle(result, (int(x), int(y)), max(10, int(r) + 4), (0, 0, 255), 3)
        cv2.circle(big_detected, (int(x), int(y)), int(r) + 20, 255, -1)
        count += 1

    # ---- Pass B: faint dust via tophat, rings killed by circularity ----
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["kernel"], p["kernel"]))
    th = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, k)
    th = cv2.bitwise_and(th, mask)
    _, bf = cv2.threshold(th, p["faint"], 255, cv2.THRESH_BINARY)
    cont2, _ = cv2.findContours(bf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cont2:
        a = cv2.contourArea(c)
        if a < p["min_area"] or a > p["max_area"]:
            continue
        per = cv2.arcLength(c, True)
        if per == 0:
            continue
        circ = (4 * math.pi * a) / (per * per)
        if circ < p["circ"]:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if big_detected[int(y), int(x)] == 255:
            continue
        cv2.circle(result, (int(x), int(y)), max(10, int(r) + 4), (0, 0, 255), 3)
        count += 1

    return result, count


class DustApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dust Detection App")

        self.original = None        # BGR original
        self.result = None          # BGR detection result
        self.centers = []           # list of (x, y, r)

        # display mapping for the input canvas
        self.scale = 1.0
        self.off_x = 0
        self.off_y = 0
        self.input_photo = None
        self.result_photo = None

        self.radius_var = tk.StringVar(value="450")

        # detection params
        self.bright_var = tk.StringVar(value="200")
        self.faint_var  = tk.StringVar(value="15")
        self.kernel_var = tk.StringVar(value="41")
        self.minarea_var = tk.StringVar(value="3")
        self.maxarea_var = tk.StringVar(value="500")
        self.circ_var   = tk.StringVar(value="0.4")

        self.status_var = tk.StringVar(value="Load an image to start.")
        self.count_var  = tk.StringVar(value="Centers: 0    Dust: -")

        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        # Toolbar
        bar = ttk.Frame(self.root, padding=6)
        bar.grid(row=0, column=0, sticky="ew")
        ttk.Button(bar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="Detect", command=self.detect).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="Save Result", command=self.save_result).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="Open Folder", command=self.open_folder).pack(side=tk.LEFT, padx=3)

        # Controls row 1
        c1 = ttk.Frame(self.root, padding=(6, 0))
        c1.grid(row=1, column=0, sticky="ew")
        ttk.Label(c1, text="Radius:").pack(side=tk.LEFT)
        ttk.Entry(c1, textvariable=self.radius_var, width=6).pack(side=tk.LEFT, padx=(2, 10))
        ttk.Button(c1, text="Undo Last", command=self.undo).pack(side=tk.LEFT, padx=3)
        ttk.Button(c1, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=3)
        ttk.Label(c1, textvariable=self.count_var).pack(side=tk.LEFT, padx=15)

        # Controls row 2 - detection settings
        c2 = ttk.LabelFrame(self.root, text="Detection Settings (tune if needed)", padding=6)
        c2.grid(row=2, column=0, sticky="ew", padx=6, pady=4)
        self._setting(c2, "Bright thr", self.bright_var)
        self._setting(c2, "Faint thr", self.faint_var)
        self._setting(c2, "TopHat kernel", self.kernel_var)
        self._setting(c2, "Min area", self.minarea_var)
        self._setting(c2, "Max area", self.maxarea_var)
        self._setting(c2, "Circularity", self.circ_var)

        # Canvas area
        ca = ttk.Frame(self.root, padding=6)
        ca.grid(row=3, column=0)

        left = ttk.Frame(ca)
        left.grid(row=0, column=0, padx=5)
        ttk.Label(left, text="INPUT  (click camera centers here)").pack()
        self.input_canvas = tk.Canvas(left, width=CANVAS_W, height=CANVAS_H,
                                      bg="gray20", highlightthickness=1,
                                      highlightbackground="gray50")
        self.input_canvas.pack()
        self.input_canvas.bind("<Button-1>", self.on_click)

        right = ttk.Frame(ca)
        right.grid(row=0, column=1, padx=5)
        ttk.Label(right, text="RESULT  (red = detected dust)").pack()
        self.result_canvas = tk.Canvas(right, width=CANVAS_W, height=CANVAS_H,
                                       bg="gray20", highlightthickness=1,
                                       highlightbackground="gray50")
        self.result_canvas.pack()

        # Status
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN,
                  anchor="w", padding=4).grid(row=4, column=0, sticky="ew")

    def _setting(self, parent, label, var):
        f = ttk.Frame(parent)
        f.pack(side=tk.LEFT, padx=6)
        ttk.Label(f, text=label).pack()
        ttk.Entry(f, textvariable=var, width=6).pack()

    # ---------------- helpers ----------------
    def _read_int(self, var, default):
        try:
            return int(float(var.get()))
        except ValueError:
            return default

    def _read_float(self, var, default):
        try:
            return float(var.get())
        except ValueError:
            return default

    def _params(self):
        return {
            "bright": self._read_int(self.bright_var, 200),
            "faint": self._read_int(self.faint_var, 15),
            "kernel": self._read_int(self.kernel_var, 41),
            "min_area": self._read_int(self.minarea_var, 3),
            "max_area": self._read_int(self.maxarea_var, 500),
            "circ": self._read_float(self.circ_var, 0.4),
        }

    def _build_before(self):
        """Original with ROI circles + numbered center dots."""
        disp = self.original.copy()
        for i, (x, y, r) in enumerate(self.centers):
            cv2.circle(disp, (x, y), r, (0, 255, 0), 4)
            cv2.circle(disp, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(disp, str(i + 1), (x + 12, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        return disp

    def _show(self, canvas, bgr, store_map=False):
        h, w = bgr.shape[:2]
        scale = min(CANVAS_W / w, CANVAS_H / h)
        sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(bgr, (sw, sh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        ox = (CANVAS_W - sw) // 2
        oy = (CANVAS_H - sh) // 2
        canvas.delete("all")
        canvas.create_image(ox, oy, anchor="nw", image=photo)
        if store_map:
            self.scale = scale
            self.off_x = ox
            self.off_y = oy
            self.input_photo = photo
        else:
            self.result_photo = photo

    def _refresh_input(self):
        if self.original is None:
            return
        self._show(self.input_canvas, self._build_before(), store_map=True)
        dust = "-" if self.result is None else str(self._last_count)
        self.count_var.set(f"Centers: {len(self.centers)}    Dust: {dust}")

    # ---------------- actions ----------------
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All", "*.*")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not load image.")
            return
        self.original = img
        self.result = None
        self.centers = []
        self._last_count = 0
        self.result_canvas.delete("all")
        self._refresh_input()
        h, w = img.shape[:2]
        self.status_var.set(f"Loaded {os.path.basename(path)} ({w}x{h}). "
                            f"Set radius and click camera centers.")

    def on_click(self, event):
        if self.original is None:
            self.status_var.set("Load an image first.")
            return
        if self.scale <= 0:
            return
        ix = (event.x - self.off_x) / self.scale
        iy = (event.y - self.off_y) / self.scale
        h, w = self.original.shape[:2]
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return
        r = self._read_int(self.radius_var, 450)
        self.centers.append((int(ix), int(iy), r))
        self._refresh_input()
        self.status_var.set(f"Added center {len(self.centers)} at "
                            f"({int(ix)}, {int(iy)}) radius {r}.")

    def undo(self):
        if self.centers:
            self.centers.pop()
            self._refresh_input()
            self.status_var.set("Removed last center.")

    def clear_all(self):
        self.centers = []
        self.result = None
        self.result_canvas.delete("all")
        self._refresh_input()
        self.status_var.set("Cleared all centers.")

    def detect(self):
        if self.original is None:
            messagebox.showwarning("No image", "Load an image first.")
            return
        if not self.centers:
            messagebox.showwarning("No centers", "Click at least one camera center.")
            return
        self.result, count = run_detection(self.original, self.centers, self._params())
        self._last_count = count
        self._show(self.result_canvas, self.result, store_map=False)
        self.count_var.set(f"Centers: {len(self.centers)}    Dust: {count}")
        self.status_var.set(f"Detection done. {count} dust spots found.")

    def save_result(self):
        if self.result is None:
            messagebox.showwarning("Nothing to save", "Run Detect first.")
            return
        os.makedirs(SAVE_DIR, exist_ok=True)
        before = self._build_before()
        # pad to same height just in case, then stitch side by side
        h = max(before.shape[0], self.result.shape[0])
        def pad(im):
            if im.shape[0] == h:
                return im
            out = np.zeros((h, im.shape[1], 3), dtype=np.uint8)
            out[:im.shape[0]] = im
            return out
        divider = np.full((h, 8, 3), 255, dtype=np.uint8)
        combined = cv2.hconcat([pad(before), divider, pad(self.result)])
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"result_{stamp}.png"
        path = os.path.join(SAVE_DIR, fname)
        cv2.imwrite(path, combined)
        self.status_var.set(f"Saved {fname} in '{SAVE_DIR}' folder.")

    def open_folder(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        path = os.path.abspath(SAVE_DIR)
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.run(["open", path])
            else:
                subprocess.run(["xdg-open", path])
            self.status_var.set(f"Opened {path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


def main():
    root = tk.Tk()
    DustApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
