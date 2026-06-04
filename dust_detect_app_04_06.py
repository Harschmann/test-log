"""
Dust Detection App  (with synced zoom/pan)
------------------------------------------
Workflow:
  1. Load Image
  2. Type a radius, then CLICK each camera center on the left panel.
     - Each click uses the current radius. Change radius for different sized cameras.
  3. Undo Last / Clear All to fix mistakes.
  4. Detect -> result appears on the right.
  5. Save Result -> writes a before|after image into the results folder.
  6. Open Folder -> opens that folder.

Controls on the image panels:
  - Mouse wheel  = zoom in / out (both panels stay in sync)
  - Click + drag = pan (both panels move together)
  - Quick click on the LEFT panel (no drag) = add a camera center
  - Zoom In / Zoom Out / Fit buttons also available

Detection logic is the two-pass method (bright dust + faint dust via tophat).

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

        self.original = None
        self.result = None
        self.centers = []
        self._last_count = 0

        # shared view state for BOTH canvases
        self.base_scale = 1.0   # fit-to-canvas scale
        self.zoom = 1.0         # extra zoom on top of fit
        self.view_x = 0.0       # image top-left in canvas coords
        self.view_y = 0.0

        # drag state
        self._dragging = False
        self._drag_start = (0, 0)
        self._last = (0, 0)

        self.input_photo = None
        self.result_photo = None

        self.radius_var = tk.StringVar(value="450")
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
        bar = ttk.Frame(self.root, padding=6)
        bar.grid(row=0, column=0, sticky="ew")
        ttk.Button(bar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="Detect", command=self.detect).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="Save Result", command=self.save_result).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="Open Folder", command=self.open_folder).pack(side=tk.LEFT, padx=3)

        c1 = ttk.Frame(self.root, padding=(6, 0))
        c1.grid(row=1, column=0, sticky="ew")
        ttk.Label(c1, text="Radius:").pack(side=tk.LEFT)
        ttk.Entry(c1, textvariable=self.radius_var, width=6).pack(side=tk.LEFT, padx=(2, 10))
        ttk.Button(c1, text="Undo Last", command=self.undo).pack(side=tk.LEFT, padx=3)
        ttk.Button(c1, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=3)
        ttk.Separator(c1, orient="vertical").pack(side=tk.LEFT, fill="y", padx=8)
        ttk.Button(c1, text="Zoom In", command=lambda: self.zoom_button(1.25)).pack(side=tk.LEFT, padx=3)
        ttk.Button(c1, text="Zoom Out", command=lambda: self.zoom_button(0.8)).pack(side=tk.LEFT, padx=3)
        ttk.Button(c1, text="Fit", command=self.fit_view).pack(side=tk.LEFT, padx=3)
        ttk.Label(c1, textvariable=self.count_var).pack(side=tk.LEFT, padx=15)

        c2 = ttk.LabelFrame(self.root, text="Detection Settings (tune if needed)", padding=6)
        c2.grid(row=2, column=0, sticky="ew", padx=6, pady=4)
        self._setting(c2, "Bright thr", self.bright_var)
        self._setting(c2, "Faint thr", self.faint_var)
        self._setting(c2, "TopHat kernel", self.kernel_var)
        self._setting(c2, "Min area", self.minarea_var)
        self._setting(c2, "Max area", self.maxarea_var)
        self._setting(c2, "Circularity", self.circ_var)

        ca = ttk.Frame(self.root, padding=6)
        ca.grid(row=3, column=0)

        left = ttk.Frame(ca)
        left.grid(row=0, column=0, padx=5)
        ttk.Label(left, text="INPUT  (click = add center, drag = pan, wheel = zoom)").pack()
        self.input_canvas = tk.Canvas(left, width=CANVAS_W, height=CANVAS_H,
                                      bg="gray20", highlightthickness=1,
                                      highlightbackground="gray50")
        self.input_canvas.pack()

        right = ttk.Frame(ca)
        right.grid(row=0, column=1, padx=5)
        ttk.Label(right, text="RESULT  (red = dust, drag = pan, wheel = zoom)").pack()
        self.result_canvas = tk.Canvas(right, width=CANVAS_W, height=CANVAS_H,
                                       bg="gray20", highlightthickness=1,
                                       highlightbackground="gray50")
        self.result_canvas.pack()

        # bind interactions on both canvases
        for cv_ in (self.input_canvas, self.result_canvas):
            cv_.bind("<MouseWheel>", self.on_wheel)      # windows / mac
            cv_.bind("<Button-4>", self.on_wheel)        # linux up
            cv_.bind("<Button-5>", self.on_wheel)        # linux down
            cv_.bind("<ButtonPress-1>", self.on_press)
            cv_.bind("<B1-Motion>", self.on_drag)
        self.input_canvas.bind("<ButtonRelease-1>", self.on_release_input)
        self.result_canvas.bind("<ButtonRelease-1>", self.on_release_other)

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
        disp = self.original.copy()
        for i, (x, y, r) in enumerate(self.centers):
            cv2.circle(disp, (x, y), r, (0, 255, 0), 4)
            cv2.circle(disp, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(disp, str(i + 1), (x + 12, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        return disp

    def _render(self, canvas, bgr, attr):
        """Render bgr into canvas using the shared view state, cropping to
        only the visible region for performance."""
        canvas.delete("all")
        if bgr is None:
            return
        H, W = bgr.shape[:2]
        s = self.base_scale * self.zoom
        vx, vy = self.view_x, self.view_y

        left = max(0, int(-vx / s))
        top = max(0, int(-vy / s))
        right = min(W, int((CANVAS_W - vx) / s) + 1)
        bottom = min(H, int((CANVAS_H - vy) / s) + 1)
        if right <= left or bottom <= top:
            return  # fully off screen

        crop = bgr[top:bottom, left:right]
        cw = max(1, int((right - left) * s))
        ch = max(1, int((bottom - top) * s))
        interp = cv2.INTER_NEAREST if self.zoom > 1.5 else cv2.INTER_AREA
        resized = cv2.resize(crop, (cw, ch), interpolation=interp)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        px = vx + left * s
        py = vy + top * s
        canvas.create_image(px, py, anchor="nw", image=photo)
        setattr(self, attr, photo)

    def _refresh_views(self):
        if self.original is None:
            return
        self._render(self.input_canvas, self._build_before(), "input_photo")
        self._render(self.result_canvas, self.result, "result_photo")
        dust = "-" if self.result is None else str(self._last_count)
        self.count_var.set(f"Centers: {len(self.centers)}    Dust: {dust}")

    # ---------------- view: zoom / pan ----------------
    def fit_view(self):
        if self.original is None:
            return
        H, W = self.original.shape[:2]
        self.base_scale = min(CANVAS_W / W, CANVAS_H / H)
        self.zoom = 1.0
        s = self.base_scale
        self.view_x = (CANVAS_W - W * s) / 2
        self.view_y = (CANVAS_H - H * s) / 2
        self._refresh_views()

    def _apply_zoom(self, factor, cx, cy):
        if self.original is None:
            return
        s_old = self.base_scale * self.zoom
        ix = (cx - self.view_x) / s_old
        iy = (cy - self.view_y) / s_old
        self.zoom = max(0.1, min(self.zoom * factor, 60.0))
        s_new = self.base_scale * self.zoom
        self.view_x = cx - ix * s_new   # keep point under cursor fixed
        self.view_y = cy - iy * s_new
        self._refresh_views()

    def on_wheel(self, event):
        if self.original is None:
            return
        if getattr(event, "delta", 0) > 0 or getattr(event, "num", None) == 4:
            factor = 1.2
        elif getattr(event, "delta", 0) < 0 or getattr(event, "num", None) == 5:
            factor = 1 / 1.2
        else:
            return
        self._apply_zoom(factor, event.x, event.y)

    def zoom_button(self, factor):
        self._apply_zoom(factor, CANVAS_W / 2, CANVAS_H / 2)

    def on_press(self, event):
        self._drag_start = (event.x, event.y)
        self._last = (event.x, event.y)
        self._dragging = False

    def on_drag(self, event):
        if self.original is None:
            return
        if not self._dragging:
            if abs(event.x - self._drag_start[0]) + abs(event.y - self._drag_start[1]) > 4:
                self._dragging = True
        if self._dragging:
            dx = event.x - self._last[0]
            dy = event.y - self._last[1]
            self.view_x += dx
            self.view_y += dy
            self._last = (event.x, event.y)
            self._refresh_views()

    def on_release_input(self, event):
        if not self._dragging:
            self._add_center(event)
        self._dragging = False

    def on_release_other(self, event):
        self._dragging = False

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
        self.fit_view()
        h, w = img.shape[:2]
        self.status_var.set(f"Loaded {os.path.basename(path)} ({w}x{h}). "
                            f"Set radius and click camera centers.")

    def _add_center(self, event):
        if self.original is None:
            self.status_var.set("Load an image first.")
            return
        s = self.base_scale * self.zoom
        ix = (event.x - self.view_x) / s
        iy = (event.y - self.view_y) / s
        h, w = self.original.shape[:2]
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return
        r = self._read_int(self.radius_var, 450)
        self.centers.append((int(ix), int(iy), r))
        self._refresh_views()
        self.status_var.set(f"Added center {len(self.centers)} at "
                            f"({int(ix)}, {int(iy)}) radius {r}.")

    def undo(self):
        if self.centers:
            self.centers.pop()
            self._refresh_views()
            self.status_var.set("Removed last center.")

    def clear_all(self):
        self.centers = []
        self.result = None
        self.result_canvas.delete("all")
        self._refresh_views()
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
        self._refresh_views()
        self.status_var.set(f"Detection done. {count} dust spots found.")

    def save_result(self):
        if self.result is None:
            messagebox.showwarning("Nothing to save", "Run Detect first.")
            return
        os.makedirs(SAVE_DIR, exist_ok=True)
        before = self._build_before()
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
