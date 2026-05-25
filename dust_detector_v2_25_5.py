"""
Camera Dust Detector v2
-----------------------
Detect white dust particles inside mobile phone camera modules.

Approach (no Hough circles):
  1. Split image with horizontal/vertical lines so each cell has ONE camera.
  2. In each cell, find the camera CENTER by locating the largest dark
     connected component (Otsu threshold so it adapts to image brightness)
     and taking its centroid. Lens reflection in the middle is closed up
     with morphology so the ring still counts as one component.
  3. Build an ANNULUS around that center using two manual radii:
        - inner radius: excludes the lens body + center reflection
        - outer radius: limits the search to the anti-reflective coating
  4. Threshold bright pixels ONLY inside that annulus.
  5. Filter blobs by size to get final dust spots.

After the first detection, the cameras stay cached. Dragging any slider
(inner / outer / threshold / size) recomputes dust live, no need to
re-click Detect. Click on the image to manually move the nearest center.

Requirements:
    pip install opencv-python pillow numpy
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk


class DustDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Dust Detector v2")
        self.root.geometry("1320x840")

        # Image state
        self.original_image = None
        self.image_path = None
        self.tk_image = None
        self.scale = 1.0

        # Grid line variables
        self.h_slider_vars = []
        self.v_slider_vars = []

        # Cached detected cameras: list of {gx, gy} in GLOBAL image coords
        self.cameras = []
        # Cached dust: list of {x, y, r} in GLOBAL image coords
        self.dust_spots = []

        # Detection parameters
        self.dust_threshold = tk.IntVar(value=180)
        self.min_dust_size = tk.IntVar(value=2)
        self.max_dust_size = tk.IntVar(value=400)
        self.inner_radius = tk.IntVar(value=40)
        self.outer_radius = tk.IntVar(value=140)

        # Debounce handle
        self._recompute_pending = None

        self._build_ui()

    # =============================== UI =============================== #

    def _build_ui(self):
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        ttk.Button(toolbar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Detect", command=self.detect).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Save Result", command=self.save_result).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Reset", command=self.reset_all).pack(side=tk.LEFT, padx=3)

        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # ---------- Canvas (left) ----------
        canvas_frame = ttk.Frame(paned)
        paned.add(canvas_frame, weight=4)
        self.canvas = tk.Canvas(canvas_frame, bg="gray18", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # ---------- Scrollable controls (right) ----------
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        ctrl_canvas = tk.Canvas(right, width=320, highlightthickness=0)
        scrollbar = ttk.Scrollbar(right, orient="vertical", command=ctrl_canvas.yview)
        self.controls = ttk.Frame(ctrl_canvas)
        self.controls.bind(
            "<Configure>",
            lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all")),
        )
        ctrl_canvas.create_window((0, 0), window=self.controls, anchor="nw")
        ctrl_canvas.configure(yscrollcommand=scrollbar.set)
        ctrl_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        ctrl_canvas.bind_all(
            "<MouseWheel>",
            lambda e: ctrl_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"),
        )

        # ---------- Grid setup ----------
        ttk.Label(self.controls, text="Grid Setup", font=("Segoe UI", 11, "bold")).pack(
            anchor=tk.W, pady=(6, 2)
        )
        cnt = ttk.Frame(self.controls)
        cnt.pack(fill=tk.X, pady=2)
        ttk.Label(cnt, text="Horizontal lines:").grid(row=0, column=0, sticky=tk.W)
        self.h_spin = ttk.Spinbox(cnt, from_=0, to=10, width=5, command=self.rebuild_sliders)
        self.h_spin.set(0)
        self.h_spin.grid(row=0, column=1, padx=6)
        ttk.Label(cnt, text="Vertical lines:").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        self.v_spin = ttk.Spinbox(cnt, from_=0, to=10, width=5, command=self.rebuild_sliders)
        self.v_spin.set(0)
        self.v_spin.grid(row=1, column=1, padx=6, pady=(6, 0))

        ttk.Separator(self.controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(self.controls, text="Line Positions", font=("Segoe UI", 11, "bold")).pack(
            anchor=tk.W
        )
        self.lines_container = ttk.Frame(self.controls)
        self.lines_container.pack(fill=tk.X, pady=2)

        ttk.Separator(self.controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # ---------- Ring (annulus) ----------
        ttk.Label(self.controls, text="Ring (ROI) Settings", font=("Segoe UI", 11, "bold")).pack(
            anchor=tk.W
        )
        ttk.Label(self.controls, text="Inner radius (exclude lens):").pack(anchor=tk.W, pady=(4, 0))
        ttk.Scale(
            self.controls, from_=0, to=1500, variable=self.inner_radius,
            orient=tk.HORIZONTAL, command=self._on_param_change,
        ).pack(fill=tk.X)
        ttk.Label(self.controls, textvariable=self.inner_radius).pack(anchor=tk.W)

        ttk.Label(self.controls, text="Outer radius (camera body):").pack(anchor=tk.W, pady=(6, 0))
        ttk.Scale(
            self.controls, from_=10, to=1500, variable=self.outer_radius,
            orient=tk.HORIZONTAL, command=self._on_param_change,
        ).pack(fill=tk.X)
        ttk.Label(self.controls, textvariable=self.outer_radius).pack(anchor=tk.W)

        ttk.Separator(self.controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # ---------- Dust detection ----------
        ttk.Label(self.controls, text="Dust Detection", font=("Segoe UI", 11, "bold")).pack(
            anchor=tk.W
        )
        ttk.Label(self.controls, text="Brightness threshold (0-255):").pack(anchor=tk.W, pady=(4, 0))
        ttk.Scale(
            self.controls, from_=100, to=255, variable=self.dust_threshold,
            orient=tk.HORIZONTAL, command=self._on_param_change,
        ).pack(fill=tk.X)
        ttk.Label(self.controls, textvariable=self.dust_threshold).pack(anchor=tk.W)

        ttk.Label(self.controls, text="Min dust size (px):").pack(anchor=tk.W, pady=(6, 0))
        ttk.Scale(
            self.controls, from_=1, to=50, variable=self.min_dust_size,
            orient=tk.HORIZONTAL, command=self._on_param_change,
        ).pack(fill=tk.X)
        ttk.Label(self.controls, textvariable=self.min_dust_size).pack(anchor=tk.W)

        ttk.Label(self.controls, text="Max dust size (px):").pack(anchor=tk.W, pady=(6, 0))
        ttk.Scale(
            self.controls, from_=50, to=3000, variable=self.max_dust_size,
            orient=tk.HORIZONTAL, command=self._on_param_change,
        ).pack(fill=tk.X)
        ttk.Label(self.controls, textvariable=self.max_dust_size).pack(anchor=tk.W)

        ttk.Separator(self.controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # ---------- Legend ----------
        legend = ttk.Frame(self.controls)
        legend.pack(fill=tk.X)
        ttk.Label(legend, text="Legend:", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(legend, text="- Yellow box  = grid cell").pack(anchor=tk.W)
        ttk.Label(legend, text="- Green dot   = camera center").pack(anchor=tk.W)
        ttk.Label(legend, text="- Cyan ring   = inner (excluded)").pack(anchor=tk.W)
        ttk.Label(legend, text="- Green ring  = outer (limit)").pack(anchor=tk.W)
        ttk.Label(legend, text="- Red circle  = dust spot").pack(anchor=tk.W)

        ttk.Separator(self.controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        tip = ttk.Label(
            self.controls,
            text="Tip: Click on the image to move the nearest camera center.",
            wraplength=290, justify=tk.LEFT,
        )
        tip.pack(anchor=tk.W)

        # Status bar
        self.status_var = tk.StringVar(value="Load an image to begin.")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(
            side=tk.BOTTOM, fill=tk.X
        )

    # =============================== File ops =============================== #

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not load image.")
            return
        self.image_path = path
        self.original_image = img
        self.cameras = []
        self.dust_spots = []
        self.rebuild_sliders()

        # Sensible default radii based on image size
        h, w = img.shape[:2]
        est_outer = max(60, min(h, w) // 8)
        self.outer_radius.set(est_outer)
        self.inner_radius.set(max(20, est_outer // 3))

        self.status_var.set(f"Loaded: {os.path.basename(path)} ({w}x{h})")

    def save_result(self):
        if self.original_image is None:
            messagebox.showwarning("Nothing to save", "Load an image first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All", "*.*")],
        )
        if not path:
            return
        composite = self._build_composite()
        cv2.imwrite(path, composite)
        self.status_var.set(f"Saved: {os.path.basename(path)}")

    def reset_all(self):
        self.h_spin.set(0)
        self.v_spin.set(0)
        self.cameras = []
        self.dust_spots = []
        self.rebuild_sliders()
        self.status_var.set("Reset.")

    # =============================== Slider build =============================== #

    def rebuild_sliders(self):
        for w in self.lines_container.winfo_children():
            w.destroy()
        self.h_slider_vars = []
        self.v_slider_vars = []

        if self.original_image is None:
            return

        h, w = self.original_image.shape[:2]
        try:
            n_h = int(self.h_spin.get())
            n_v = int(self.v_spin.get())
        except ValueError:
            return

        for i in range(n_h):
            f = ttk.Frame(self.lines_container)
            f.pack(fill=tk.X, pady=1)
            ttk.Label(f, text=f"H{i+1}", width=4).pack(side=tk.LEFT)
            initial = int(h * (i + 1) / (n_h + 1))
            var = tk.IntVar(value=initial)
            ttk.Scale(
                f, from_=0, to=h, variable=var, orient=tk.HORIZONTAL,
                command=lambda *_: self.redraw(),
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            ttk.Label(f, textvariable=var, width=5).pack(side=tk.LEFT)
            self.h_slider_vars.append(var)

        for i in range(n_v):
            f = ttk.Frame(self.lines_container)
            f.pack(fill=tk.X, pady=1)
            ttk.Label(f, text=f"V{i+1}", width=4).pack(side=tk.LEFT)
            initial = int(w * (i + 1) / (n_v + 1))
            var = tk.IntVar(value=initial)
            ttk.Scale(
                f, from_=0, to=w, variable=var, orient=tk.HORIZONTAL,
                command=lambda *_: self.redraw(),
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            ttk.Label(f, textvariable=var, width=5).pack(side=tk.LEFT)
            self.v_slider_vars.append(var)

        self.redraw()

    # =============================== Param changes =============================== #

    def _on_param_change(self, *args):
        """Debounce: recompute dust 120ms after the last slider tick."""
        if self._recompute_pending is not None:
            self.root.after_cancel(self._recompute_pending)
        self._recompute_pending = self.root.after(120, self._recompute_dust)

    def _recompute_dust(self):
        self._recompute_pending = None
        if self.original_image is None:
            return
        if not self.cameras:
            # No cameras yet, just redraw the rings preview around nothing.
            self.redraw()
            return

        inner_r = self.inner_radius.get()
        outer_r = self.outer_radius.get()
        threshold = self.dust_threshold.get()
        min_sz = self.min_dust_size.get()
        max_sz = self.max_dust_size.get()

        if outer_r <= inner_r:
            self.dust_spots = []
            self.status_var.set(
                f"{len(self.cameras)} cameras, 0 dust spots. Outer must be > inner."
            )
            self.redraw()
            return

        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, bright = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        kernel = np.ones((2, 2), np.uint8)

        all_dust = []
        for cam in self.cameras:
            cx, cy = cam["gx"], cam["gy"]
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (cx, cy), outer_r, 255, -1)
            cv2.circle(mask, (cx, cy), inner_r, 0, -1)

            in_ring = cv2.bitwise_and(bright, mask)
            in_ring = cv2.morphologyEx(in_ring, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(in_ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if min_sz <= area <= max_sz:
                    (x, y), r = cv2.minEnclosingCircle(cnt)
                    all_dust.append({"x": int(x), "y": int(y), "r": int(r)})

        self.dust_spots = all_dust
        self.status_var.set(
            f"{len(self.cameras)} cameras, {len(all_dust)} dust spots "
            f"(inner={inner_r}, outer={outer_r}, thr={threshold})."
        )
        self.redraw()

    # =============================== Canvas click =============================== #

    def on_canvas_click(self, event):
        """Move the nearest camera center to the clicked location."""
        if not self.cameras or self.original_image is None or self.scale <= 0:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        h, w = self.original_image.shape[:2]
        new_w = int(w * self.scale)
        new_h = int(h * self.scale)
        ox = max(0, (cw - new_w) // 2)
        oy = max(0, (ch - new_h) // 2)
        ix = (event.x - ox) / self.scale
        iy = (event.y - oy) / self.scale
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return

        best_i = -1
        best_dist = float("inf")
        for i, cam in enumerate(self.cameras):
            d = np.hypot(cam["gx"] - ix, cam["gy"] - iy)
            if d < best_dist:
                best_dist = d
                best_i = i
        if best_i >= 0:
            self.cameras[best_i]["gx"] = int(ix)
            self.cameras[best_i]["gy"] = int(iy)
            self._recompute_dust()

    # =============================== Draw =============================== #

    def _compute_scale(self):
        if self.original_image is None:
            return 1.0
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        h, w = self.original_image.shape[:2]
        return min(cw / w, ch / h, 1.0)

    def _build_composite(self):
        """Return BGR image with grid lines, centers, rings, and dust drawn."""
        if self.original_image is None:
            return None
        disp = self.original_image.copy()
        h, w = disp.shape[:2]

        inner_r = self.inner_radius.get()
        outer_r = self.outer_radius.get()

        for var in self.h_slider_vars:
            y = int(var.get())
            cv2.line(disp, (0, y), (w, y), (0, 255, 255), 2)
        for var in self.v_slider_vars:
            x = int(var.get())
            cv2.line(disp, (x, 0), (x, h), (0, 255, 255), 2)

        for cam in self.cameras:
            cx, cy = cam["gx"], cam["gy"]
            cv2.circle(disp, (cx, cy), outer_r, (0, 200, 0), 2)
            cv2.circle(disp, (cx, cy), inner_r, (255, 200, 0), 2)
            cv2.circle(disp, (cx, cy), 5, (0, 255, 0), -1)

        for spot in self.dust_spots:
            cv2.circle(disp, (spot["x"], spot["y"]), max(8, spot["r"] + 6), (0, 0, 230), 2)

        return disp

    def redraw(self):
        if self.original_image is None:
            return

        self.scale = self._compute_scale()
        disp = self._build_composite()
        h, w = disp.shape[:2]

        new_w = max(1, int(w * self.scale))
        new_h = max(1, int(h * self.scale))
        resized = cv2.resize(disp, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.tk_image = ImageTk.PhotoImage(Image.fromarray(rgb))

        self.canvas.delete("all")
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        ox = max(0, (cw - new_w) // 2)
        oy = max(0, (ch - new_h) // 2)
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self.tk_image)

    # =============================== Detection =============================== #

    def _get_cells(self):
        h, w = self.original_image.shape[:2]
        x_lines = sorted(set([0, w] + [int(v.get()) for v in self.v_slider_vars]))
        y_lines = sorted(set([0, h] + [int(v.get()) for v in self.h_slider_vars]))
        cells = []
        for i in range(len(y_lines) - 1):
            for j in range(len(x_lines) - 1):
                cells.append((x_lines[j], y_lines[i], x_lines[j + 1], y_lines[i + 1]))
        return cells

    def _find_camera_center(self, cell_bgr):
        """Return (cx, cy) in cell-local coords, or None."""
        h, w = cell_bgr.shape[:2]
        gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Otsu inverse: adapt to whatever brightness the image is.
        _, dark = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Close gaps from lens reflection so the dark module is one blob.
        kernel = np.ones((9, 9), np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dark, connectivity=8
        )
        if num_labels < 2:
            return None

        cell_cx, cell_cy = w / 2.0, h / 2.0
        candidates = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 0.02 * h * w:
                continue
            cx, cy = centroids[i]
            dist = float(np.hypot(cx - cell_cx, cy - cell_cy))
            candidates.append((dist, int(area), float(cx), float(cy)))

        if not candidates:
            return None

        # Prefer near cell center; bigger area breaks ties.
        candidates.sort(key=lambda c: (c[0], -c[1]))
        _, _, cx, cy = candidates[0]
        return int(cx), int(cy)

    def detect(self):
        if self.original_image is None:
            messagebox.showwarning("No image", "Load an image first.")
            return

        cells = self._get_cells()
        self.cameras = []

        for (x1, y1, x2, y2) in cells:
            cell = self.original_image[y1:y2, x1:x2]
            if cell.size == 0 or cell.shape[0] < 30 or cell.shape[1] < 30:
                continue
            center = self._find_camera_center(cell)
            if center is None:
                continue
            cx, cy = center
            self.cameras.append({"gx": x1 + cx, "gy": y1 + cy})

        if not self.cameras:
            messagebox.showinfo(
                "No cameras found",
                "Could not detect a dark blob in any cell. Adjust grid lines or "
                "verify the image contains a camera module per cell.",
            )
            self.dust_spots = []
            self.redraw()
            return

        # Now apply dust detection using current ring + threshold.
        self._recompute_dust()


def main():
    root = tk.Tk()
    DustDetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
