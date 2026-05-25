"""
Camera Dust Detector
--------------------
GUI to detect white dust spots inside mobile phone camera lenses.

Workflow:
  1. Load a back-of-phone image (camera section).
  2. Choose number of horizontal / vertical grid lines so each cell
     contains exactly ONE camera module.
  3. Drag the sliders to position the lines between camera modules.
  4. Tune dust threshold & size sliders if needed.
  5. Hit "Detect Dust".

Output:
  - Green circle  = detected camera lens
  - Red circle    = dust spot
  - Yellow box    = grid cell

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
        self.root.title("Camera Dust Detector")
        self.root.geometry("1280x820")

        # Image state
        self.original_image = None  # BGR numpy array
        self.result_image = None    # BGR with annotations after detection
        self.image_path = None
        self.tk_image = None        # keep ref so PhotoImage isn't GC'd
        self.scale = 1.0

        # Line variables (IntVar list, holds position in image coords)
        self.h_slider_vars = []
        self.v_slider_vars = []

        # Detection params
        self.dust_threshold = tk.IntVar(value=180)
        self.min_dust_size = tk.IntVar(value=2)
        self.max_dust_size = tk.IntVar(value=400)

        self._build_ui()

    # ----------------------- UI ----------------------- #

    def _build_ui(self):
        # Toolbar
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        ttk.Button(toolbar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Detect Dust", command=self.detect).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Save Result", command=self.save_result).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Reset", command=self.reset_all).pack(side=tk.LEFT, padx=3)

        # Split: canvas left, controls right
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        canvas_frame = ttk.Frame(paned)
        paned.add(canvas_frame, weight=4)
        self.canvas = tk.Canvas(canvas_frame, bg="gray18", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

        # Scrollable controls panel
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

        # mouse-wheel scroll on controls
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

        # ---------- Line positions ----------
        ttk.Label(self.controls, text="Line Positions", font=("Segoe UI", 11, "bold")).pack(
            anchor=tk.W
        )
        self.lines_container = ttk.Frame(self.controls)
        self.lines_container.pack(fill=tk.X, pady=2)

        ttk.Separator(self.controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # ---------- Detection settings ----------
        ttk.Label(self.controls, text="Detection Settings", font=("Segoe UI", 11, "bold")).pack(
            anchor=tk.W
        )

        ttk.Label(self.controls, text="Dust brightness threshold (0-255):").pack(
            anchor=tk.W, pady=(4, 0)
        )
        ttk.Scale(
            self.controls, from_=100, to=255, variable=self.dust_threshold, orient=tk.HORIZONTAL
        ).pack(fill=tk.X)
        ttk.Label(self.controls, textvariable=self.dust_threshold).pack(anchor=tk.W)

        ttk.Label(self.controls, text="Min dust size (px):").pack(anchor=tk.W, pady=(6, 0))
        ttk.Scale(
            self.controls, from_=1, to=50, variable=self.min_dust_size, orient=tk.HORIZONTAL
        ).pack(fill=tk.X)
        ttk.Label(self.controls, textvariable=self.min_dust_size).pack(anchor=tk.W)

        ttk.Label(self.controls, text="Max dust size (px):").pack(anchor=tk.W, pady=(6, 0))
        ttk.Scale(
            self.controls, from_=50, to=3000, variable=self.max_dust_size, orient=tk.HORIZONTAL
        ).pack(fill=tk.X)
        ttk.Label(self.controls, textvariable=self.max_dust_size).pack(anchor=tk.W)

        ttk.Separator(self.controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # Legend
        legend = ttk.Frame(self.controls)
        legend.pack(fill=tk.X)
        ttk.Label(legend, text="Legend:", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(legend, text="• Yellow box = grid cell", foreground="#aaaa00").pack(anchor=tk.W)
        ttk.Label(legend, text="• Green circle = camera lens", foreground="#008800").pack(anchor=tk.W)
        ttk.Label(legend, text="• Red circle = dust spot", foreground="#cc0000").pack(anchor=tk.W)

        # Status bar
        self.status_var = tk.StringVar(value="Load an image to begin.")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(
            side=tk.BOTTOM, fill=tk.X
        )

    # ----------------------- File ops ----------------------- #

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
        self.result_image = None
        self.rebuild_sliders()
        self.status_var.set(f"Loaded: {os.path.basename(path)} ({img.shape[1]}x{img.shape[0]})")

    def save_result(self):
        if self.result_image is None:
            messagebox.showwarning("Nothing to save", "Run detection first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All", "*.*")],
        )
        if not path:
            return
        cv2.imwrite(path, self.result_image)
        self.status_var.set(f"Saved: {os.path.basename(path)}")

    def reset_all(self):
        self.h_spin.set(0)
        self.v_spin.set(0)
        self.result_image = None
        self.rebuild_sliders()
        self.status_var.set("Reset.")

    # ----------------------- Slider build ----------------------- #

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
                command=lambda *_: self.redraw()
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
                command=lambda *_: self.redraw()
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            ttk.Label(f, textvariable=var, width=5).pack(side=tk.LEFT)
            self.v_slider_vars.append(var)

        self.redraw()

    # ----------------------- Draw ----------------------- #

    def _compute_scale(self):
        if self.original_image is None:
            return 1.0
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        h, w = self.original_image.shape[:2]
        return min(cw / w, ch / h, 1.0)

    def redraw(self):
        if self.original_image is None:
            return

        self.scale = self._compute_scale()
        base = self.result_image if self.result_image is not None else self.original_image
        disp = base.copy()
        h, w = disp.shape[:2]
        thick = max(2, int(2 / max(self.scale, 0.01)))

        for var in self.h_slider_vars:
            y = int(var.get())
            cv2.line(disp, (0, y), (w, y), (0, 255, 255), thick)
        for var in self.v_slider_vars:
            x = int(var.get())
            cv2.line(disp, (x, 0), (x, h), (0, 255, 255), thick)

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

    # ----------------------- Detection ----------------------- #

    def _get_grid_cells(self):
        h, w = self.original_image.shape[:2]
        x_lines = sorted(set([0, w] + [int(v.get()) for v in self.v_slider_vars]))
        y_lines = sorted(set([0, h] + [int(v.get()) for v in self.h_slider_vars]))
        cells = []
        for i in range(len(y_lines) - 1):
            for j in range(len(x_lines) - 1):
                cells.append((x_lines[j], y_lines[i], x_lines[j + 1], y_lines[i + 1]))
        return cells

    def detect(self):
        if self.original_image is None:
            messagebox.showwarning("No image", "Load an image first.")
            return

        cells = self._get_grid_cells()
        result = self.original_image.copy()
        threshold = self.dust_threshold.get()
        min_sz = self.min_dust_size.get()
        max_sz = self.max_dust_size.get()

        total_cameras = 0
        total_dust = 0

        for idx, (x1, y1, x2, y2) in enumerate(cells):
            cell = self.original_image[y1:y2, x1:x2]
            if cell.size == 0 or cell.shape[0] < 30 or cell.shape[1] < 30:
                continue

            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            min_dim = min(cell.shape[:2])

            # Find the camera lens — largest circular feature in the cell.
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                minDist=min_dim,
                param1=50, param2=25,
                minRadius=int(min_dim * 0.12),
                maxRadius=int(min_dim * 0.48),
            )

            # Draw cell boundary
            cv2.rectangle(result, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 2)
            cv2.putText(
                result, f"#{idx+1}", (x1 + 6, y1 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )

            if circles is None:
                continue

            circles = np.round(circles[0, :]).astype("int")
            ch_, cw_ = cell.shape[:2]

            # Prefer circles near the cell center; tie-break by radius.
            def score(c):
                cx, cy, r = c
                center_dist = np.hypot(cx - cw_ / 2, cy - ch_ / 2)
                return r - 0.3 * center_dist

            cx, cy, r = max(circles, key=score)
            total_cameras += 1

            # Draw lens boundary
            cv2.circle(result, (x1 + cx, y1 + cy), r, (0, 200, 0), 3)
            cv2.putText(
                result, f"Cam {total_cameras}",
                (x1 + cx - 40, y1 + cy - r - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2,
            )

            # Lens-interior mask (shrunk to avoid edge highlights)
            mask = np.zeros(cell.shape[:2], dtype=np.uint8)
            inner_r = int(r * 0.88)
            cv2.circle(mask, (cx, cy), inner_r, 255, -1)

            # Threshold bright pixels inside the lens
            _, bright = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            dust_mask = cv2.bitwise_and(bright, mask)

            # Slight open to clean salt-noise
            kernel = np.ones((2, 2), np.uint8)
            dust_mask = cv2.morphologyEx(dust_mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(dust_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cell_dust = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if min_sz <= area <= max_sz:
                    cell_dust += 1
                    total_dust += 1
                    (dx, dy), dr = cv2.minEnclosingCircle(cnt)
                    cv2.circle(
                        result, (int(x1 + dx), int(y1 + dy)),
                        max(8, int(dr) + 6), (0, 0, 230), 2,
                    )

            cv2.putText(
                result, f"{cell_dust} dust",
                (x1 + cx - 35, y1 + cy + r + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 230), 2,
            )

        self.result_image = result
        self.redraw()
        self.status_var.set(
            f"Detected {total_cameras} camera(s), {total_dust} dust spot(s). "
            f"Tune threshold/size sliders and re-run if needed."
        )

        if total_cameras == 0:
            messagebox.showinfo(
                "No cameras found",
                "Could not detect any camera lens. Adjust the grid so each cell "
                "contains exactly ONE camera, then try again.",
            )


def main():
    root = tk.Tk()
    DustDetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
