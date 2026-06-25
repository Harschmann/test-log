"""
Z-Score Dust Detection App
---------------------------
Finalized algorithm (replaces tophat which falsely flagged AR-coating rings):

  1. Grayscale the image.
  2. Local mean and local std via cv2.boxFilter with WINDOW size (default 100).
     Window must be bigger than a dust blob, otherwise dust pulls its own
     local mean up and self-suppresses (z-score drops, dust gets missed).
  3. z = (pixel - local_mean) / local_std
     z >= Z_THR  => pixel is abnormally bright vs its neighbourhood.
     Rings / AR-coating / texture are "normal" relative to their own
     neighbourhood, so they stay low-z. Real dust is isolated and spikes.
  4. Absolute brightness gate: gray >= ABS_THR (default 140).
     A smudge/fingerprint can pass the z-score test (locally it looks
     bright) but is not actually bright in absolute terms, so this kills
     smudge false positives.
  5. Area filter (default 10 px) to drop sensor noise specks.
  6. Result = original image with RED circles drawn on each detected dust
     contour (not a black/white mask).

Workflow:
  1. Load Image (tiff/tif/png/jpg supported)
  2. Set Radius, then CLICK each camera lens center on the LEFT panel
     (quick click, no drag -> adds a center; click+drag -> pan instead)
  3. Undo to remove the last added center
  4. Tune Window / Z thr / Abs thr if needed
  5. Run -> right panel shows original + red dust markers

Image panel controls (synced across both panels):
  - Mouse wheel   = zoom, anchored to cursor position
  - Click + drag  = pan
  - Zoom In / Zoom Out / Fit buttons also available
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

CANVAS_W = 600
CANVAS_H = 600


class ZscoreDustApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Z-Score Dust Detection")

        self.original = None
        self.result_img = None
        self.centers = []          # list of (cx, cy, radius)

        # zoom / pan state (shared across both panels)
        self.zoom = 1.0
        self.base_scale = 1.0
        self.view_x = 0.0
        self.view_y = 0.0
        self._dragging = False
        self._drag_start = (0, 0)
        self._last = (0, 0)

        self.input_photo = None
        self.result_photo = None

        # tunable parameters
        self.radius_var = tk.StringVar(value="450")
        self.win_var    = tk.StringVar(value="100")
        self.zthr_var   = tk.StringVar(value="3.0")
        self.absthr_var = tk.StringVar(value="140")
        self.area_var   = tk.StringVar(value="10")
        self.circ_var   = tk.StringVar(value="0.45")   # min circularity to count as dust
        self.aspect_var = tk.StringVar(value="3.0")    # max aspect ratio to count as dust
        self.thread_w_var = tk.StringVar(value="5")    # max minor-axis width (px) -> thread
        self.show_glue_var = tk.BooleanVar(value=True)
        self.status     = tk.StringVar(value="Load an image.")

        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        bar = ttk.Frame(self.root, padding=6)
        bar.grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Button(bar, text="Load Image", command=self.load).pack(side=tk.LEFT, padx=3)

        ttk.Label(bar, text="Radius:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(bar, textvariable=self.radius_var, width=6).pack(side=tk.LEFT)

        ttk.Label(bar, text="Window:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(bar, textvariable=self.win_var, width=5).pack(side=tk.LEFT)

        ttk.Label(bar, text="Z thr:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(bar, textvariable=self.zthr_var, width=5).pack(side=tk.LEFT)

        ttk.Label(bar, text="Abs thr:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(bar, textvariable=self.absthr_var, width=5).pack(side=tk.LEFT)

        ttk.Label(bar, text="Min area:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(bar, textvariable=self.area_var, width=5).pack(side=tk.LEFT)

        ttk.Button(bar, text="Undo", command=self.undo).pack(side=tk.LEFT, padx=(10, 3))
        ttk.Button(bar, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="Run", command=self.run).pack(side=tk.LEFT, padx=(10, 3))

        bar2 = ttk.Frame(self.root, padding=(6, 0))
        bar2.grid(row=1, column=0, columnspan=2, sticky="w")

        ttk.Label(bar2, text="Min circularity (dust):").pack(side=tk.LEFT, padx=(0, 2))
        ttk.Entry(bar2, textvariable=self.circ_var, width=5).pack(side=tk.LEFT)

        ttk.Label(bar2, text="Max aspect ratio (dust):").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(bar2, textvariable=self.aspect_var, width=5).pack(side=tk.LEFT)

        ttk.Label(bar2, text="Thread max width (px):").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(bar2, textvariable=self.thread_w_var, width=5).pack(side=tk.LEFT)

        ttk.Checkbutton(bar2, text="Mark glue/thread too", variable=self.show_glue_var
                         ).pack(side=tk.LEFT, padx=(10, 3))

        zoom_bar = ttk.Frame(self.root, padding=(6, 0))
        zoom_bar.grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Button(zoom_bar, text="Zoom In",  command=lambda: self.zoom_button(1.25)).pack(side=tk.LEFT, padx=3)
        ttk.Button(zoom_bar, text="Zoom Out", command=lambda: self.zoom_button(1/1.25)).pack(side=tk.LEFT, padx=3)
        ttk.Button(zoom_bar, text="Fit",      command=self.fit).pack(side=tk.LEFT, padx=3)

        self.input_canvas = tk.Canvas(self.root, width=CANVAS_W, height=CANVAS_H,
                                       bg="gray20", highlightthickness=1)
        self.input_canvas.grid(row=3, column=0, padx=6, pady=6)

        self.result_canvas = tk.Canvas(self.root, width=CANVAS_W, height=CANVAS_H,
                                        bg="gray20", highlightthickness=1)
        self.result_canvas.grid(row=3, column=1, padx=6, pady=6)

        status_bar = ttk.Label(self.root, textvariable=self.status, anchor="w")
        status_bar.grid(row=4, column=0, columnspan=2, sticky="we", padx=6, pady=(0, 6))

        # mouse bindings (both panels synced)
        for cv in (self.input_canvas, self.result_canvas):
            cv.bind("<MouseWheel>", self.on_wheel)        # Windows
            cv.bind("<Button-4>", self.on_wheel)          # Linux scroll up
            cv.bind("<Button-5>", self.on_wheel)          # Linux scroll down
            cv.bind("<ButtonPress-1>", self.on_press)
            cv.bind("<B1-Motion>", self.on_drag)

        self.input_canvas.bind("<ButtonRelease-1>", self.on_release_input)
        self.result_canvas.bind("<ButtonRelease-1>", self.on_release_other)

    # ---------------- file ----------------
    def load(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.tiff *.tif *.png *.jpg *.jpeg *.bmp"), ("All", "*.*")]
        )
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", "Could not load image.")
            return
        self.original = img
        self.centers = []
        self.result_img = None
        self.result_canvas.delete("all")
        self.fit()
        self.status.set("Loaded. Click camera centers then Run.")

    # ---------------- zoom / pan (shared by both panels) ----------------
    def fit(self):
        if self.original is None:
            return
        h, w = self.original.shape[:2]
        self.base_scale = min(CANVAS_W / w, CANVAS_H / h)
        self.zoom = 1.0
        s = self.base_scale
        self.view_x = (CANVAS_W - w * s) / 2
        self.view_y = (CANVAS_H - h * s) / 2
        self._refresh()

    def _apply_zoom(self, factor, cx, cy):
        if self.original is None:
            return
        s_old = self.base_scale * self.zoom
        ix = (cx - self.view_x) / s_old
        iy = (cy - self.view_y) / s_old
        self.zoom = max(0.1, min(self.zoom * factor, 60.0))
        s_new = self.base_scale * self.zoom
        self.view_x = cx - ix * s_new
        self.view_y = cy - iy * s_new
        self._refresh()

    def zoom_button(self, factor):
        self._apply_zoom(factor, CANVAS_W / 2, CANVAS_H / 2)

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
            self._refresh()

    def on_release_input(self, event):
        if not self._dragging:
            self._add_center(event)
        self._dragging = False

    def on_release_other(self, event):
        self._dragging = False

    def _add_center(self, event):
        if self.original is None:
            self.status.set("Load an image first.")
            return
        s = self.base_scale * self.zoom
        ix = (event.x - self.view_x) / s
        iy = (event.y - self.view_y) / s
        h, w = self.original.shape[:2]
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return
        try:
            r = int(self.radius_var.get())
        except ValueError:
            r = 450
        self.centers.append((int(ix), int(iy), r))
        self._refresh()
        self.status.set(f"Added center {len(self.centers)} at ({int(ix)}, {int(iy)}) radius {r}.")

    def undo(self):
        if self.centers:
            self.centers.pop()
            self._refresh()
            self.status.set(f"Undo. {len(self.centers)} center(s) remain.")

    def clear_all(self):
        self.centers = []
        self._refresh()
        self.status.set("Cleared all centers.")

    # ---------------- rendering ----------------
    def _render(self, canvas, base_bgr, photo_attr):
        canvas.delete("all")
        if base_bgr is None:
            return
        h, w = base_bgr.shape[:2]
        s = self.base_scale * self.zoom
        disp = cv2.resize(base_bgr, (max(1, int(w * s)), max(1, int(h * s))),
                           interpolation=cv2.INTER_LINEAR if s >= 1 else cv2.INTER_AREA)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        setattr(self, photo_attr, photo)  # keep a reference, tkinter needs it
        canvas.create_image(self.view_x, self.view_y, anchor="nw", image=photo)

        # draw camera-center circles on the input panel only
        if canvas is self.input_canvas:
            for (cx, cy, r) in self.centers:
                x0, y0 = self.view_x + cx * s, self.view_y + cy * s
                rad = r * s
                canvas.create_oval(x0 - rad, y0 - rad, x0 + rad, y0 + rad,
                                    outline="lime", width=2)
                canvas.create_oval(x0 - 4, y0 - 4, x0 + 4, y0 + 4, fill="lime")

    def _refresh(self):
        self._render(self.input_canvas, self.original, "input_photo")
        self._render(self.result_canvas, self.result_img, "result_photo")

    # ---------------- detection ----------------
    def run(self):
        if self.original is None or not self.centers:
            self.status.set("Load image and add at least one center first.")
            return

        try:
            win = int(self.win_var.get())
            if win % 2 == 0:
                win += 1
        except ValueError:
            win = 101
        try:
            z_thr = float(self.zthr_var.get())
        except ValueError:
            z_thr = 3.0
        try:
            abs_thr = float(self.absthr_var.get())
        except ValueError:
            abs_thr = 140.0
        try:
            min_area = float(self.area_var.get())
        except ValueError:
            min_area = 10.0

        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32)

        # Step 2: local mean / std via box filter (window must be > dust size)
        local_mean = cv2.boxFilter(gray_f, -1, (win, win))
        local_mean_sq = cv2.boxFilter(gray_f * gray_f, -1, (win, win))
        local_var = np.maximum(local_mean_sq - local_mean * local_mean, 0)
        local_std = np.sqrt(local_var)

        # Step 3: z-score
        zscore = np.where(local_std > 1e-5, (gray_f - local_mean) / local_std, 0.0)

        # restrict to camera lens ROI(s)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        for (cx, cy, r) in self.centers:
            cv2.circle(mask, (cx, cy), r, 255, -1)

        # Step 3 + 4: z-score AND absolute brightness gate
        binary = np.where(
            (zscore >= z_thr) & (gray_f >= abs_thr) & (mask == 255),
            255, 0
        ).astype(np.uint8)

        try:
            min_circ = float(self.circ_var.get())
        except ValueError:
            min_circ = 0.45
        try:
            max_aspect = float(self.aspect_var.get())
        except ValueError:
            max_aspect = 3.0
        try:
            thread_max_w = float(self.thread_w_var.get())
        except ValueError:
            thread_max_w = 5.0
        show_glue = bool(self.show_glue_var.get())

        # Step 5: area filter
        # Step 6 (shape gate): both glue residue and stray fiber/thread show up
        # as bright-enough, locally-anomalous, non-round shapes -> they pass
        # the z-score + abs brightness gate same as dust, so we separate them
        # here using shape:
        #   circularity = 4*pi*Area / perimeter^2   -> ~1 for a round blob,
        #                                               low for anything thin
        #   aspect ratio = long side / short side of the minAreaRect ->
        #                                               high for thin shapes
        #   minor_side   = short side of the minAreaRect (actual width) ->
        #                  a thread is much thinner than a glue arc/cord even
        #                  though both are elongated, so width separates them
        # A contour is "dust" only if circularity is high enough AND it is not
        # too elongated. Otherwise it's thin -> classified by width as either
        # "thread" (very thin, width <= thread_max_w) or "glue" (thicker arc).
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = self.original.copy()
        dust_count = 0
        glue_count = 0
        thread_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0

            (rw, rh) = cv2.minAreaRect(cnt)[1]
            minor_side = max(min(rw, rh), 1e-3)
            major_side = max(rw, rh)
            aspect = major_side / minor_side

            is_thin = (circularity < min_circ) or (aspect > max_aspect)

            (x, y), r = cv2.minEnclosingCircle(cnt)
            if not is_thin:
                dust_count += 1
                cv2.circle(result, (int(x), int(y)), max(10, int(r) + 4), (0, 0, 255), 3)
            elif minor_side <= thread_max_w:
                thread_count += 1
                if show_glue:
                    cv2.circle(result, (int(x), int(y)), max(10, int(r) + 4), (0, 255, 255), 2)
            else:
                glue_count += 1
                if show_glue:
                    cv2.circle(result, (int(x), int(y)), max(10, int(r) + 4), (255, 0, 0), 2)

        self.result_img = result

        in_mask = zscore[mask == 255]
        max_z = float(in_mask.max()) if in_mask.size else 0.0

        self._refresh()
        extra = []
        if glue_count:
            extra.append(f"{glue_count} glue arc(s)")
        if thread_count:
            extra.append(f"{thread_count} thread(s)")
        extra_note = f", excluded: {', '.join(extra)}" if extra else ""
        self.status.set(
            f"Done. {dust_count} dust spot(s){extra_note}. win={win}, z_thr={z_thr}, "
            f"abs_thr={abs_thr}, min_area={min_area}, circ>={min_circ}, "
            f"aspect<={max_aspect}, thread_w<={thread_max_w}, max_z={max_z:.2f}"
        )


if __name__ == "__main__":
    root = tk.Tk()
    ZscoreDustApp(root)
    root.mainloop()
