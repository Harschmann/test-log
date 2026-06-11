import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

CANVAS_W = 600
CANVAS_H = 600

class ZscoreViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Z-Score Viewer")
        self.original = None
        self.centers = []
        self.result_img = None

        self.zoom = 1.0
        self.base_scale = 1.0
        self.view_x = 0.0
        self.view_y = 0.0
        self._dragging = False
        self._drag_start = (0, 0)
        self._last = (0, 0)

        self.input_photo = None
        self.result_photo = None

        self.radius_var = tk.StringVar(value="450")
        self.win_var    = tk.StringVar(value="31")
        self.zthr_var   = tk.StringVar(value="3.0")
        self.status     = tk.StringVar(value="Load an image.")

        self._build_ui()

    def _build_ui(self):
        bar = ttk.Frame(self.root, padding=6)
        bar.grid(row=0, column=0, columnspan=2)
        ttk.Button(bar, text="Load Image", command=self.load).pack(side=tk.LEFT, padx=3)
        ttk.Label(bar, text="Radius:").pack(side=tk.LEFT, padx=(10,2))
        ttk.Entry(bar, textvariable=self.radius_var, width=6).pack(side=tk.LEFT)
        ttk.Label(bar, text="Window:").pack(side=tk.LEFT, padx=(10,2))
        ttk.Entry(bar, textvariable=self.win_var, width=5).pack(side=tk.LEFT)
        ttk.Label(bar, text="Z thr:").pack(side=tk.LEFT, padx=(10,2))
        ttk.Entry(bar, textvariable=self.zthr_var, width=5).pack(side=tk.LEFT)
        ttk.Separator(bar, orient="vertical").pack(side=tk.LEFT, fill="y", padx=8)
        ttk.Button(bar, text="Undo", command=self.undo).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="Run", command=self.run).pack(side=tk.LEFT, padx=3)
        ttk.Separator(bar, orient="vertical").pack(side=tk.LEFT, fill="y", padx=8)
        ttk.Button(bar, text="Zoom In",  command=lambda: self._zoom_btn(1.25)).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Zoom Out", command=lambda: self._zoom_btn(0.8)).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Fit",      command=self.fit).pack(side=tk.LEFT, padx=2)
        ttk.Label(bar, textvariable=self.status).pack(side=tk.LEFT, padx=10)

        left = ttk.LabelFrame(self.root, text="INPUT (click=center, drag=pan, wheel=zoom)")
        left.grid(row=1, column=0, padx=6, pady=6)
        self.ic = tk.Canvas(left, width=CANVAS_W, height=CANVAS_H, bg="gray20")
        self.ic.pack()

        right = ttk.LabelFrame(self.root, text="Z-SCORE (white = abnormally bright = dust candidate)")
        right.grid(row=1, column=1, padx=6, pady=6)
        self.rc = tk.Canvas(right, width=CANVAS_W, height=CANVAS_H, bg="gray20")
        self.rc.pack()

        for cv_ in (self.ic, self.rc):
            cv_.bind("<MouseWheel>", self.on_wheel)
            cv_.bind("<Button-4>",   self.on_wheel)
            cv_.bind("<Button-5>",   self.on_wheel)
            cv_.bind("<ButtonPress-1>",   self.on_press)
            cv_.bind("<B1-Motion>",       self.on_drag)
        self.ic.bind("<ButtonRelease-1>", self.on_release_input)
        self.rc.bind("<ButtonRelease-1>", self.on_release_other)

    # ---------- load ----------
    def load(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"), ("All", "*.*")])
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            self.status.set("Could not load image.")
            return
        self.original = img
        self.centers  = []
        self.result_img = None
        self.rc.delete("all")
        self.fit()
        self.status.set("Loaded. Click camera centers then Run.")

    # ---------- zoom / pan ----------
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

    def _zoom_btn(self, factor):
        self._apply_zoom(factor, CANVAS_W / 2, CANVAS_H / 2)

    def on_wheel(self, event):
        if self.original is None:
            return
        if getattr(event, "delta", 0) > 0 or getattr(event, "num", None) == 4:
            self._apply_zoom(1.2, event.x, event.y)
        elif getattr(event, "delta", 0) < 0 or getattr(event, "num", None) == 5:
            self._apply_zoom(1/1.2, event.x, event.y)

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
            self.view_x += event.x - self._last[0]
            self.view_y += event.y - self._last[1]
            self._last = (event.x, event.y)
            self._refresh()

    def on_release_input(self, event):
        if not self._dragging:
            self._add_center(event)
        self._dragging = False

    def on_release_other(self, event):
        self._dragging = False

    # ---------- render ----------
    def _render(self, canvas, bgr, attr):
        canvas.delete("all")
        if bgr is None:
            return
        H, W = bgr.shape[:2]
        s = self.base_scale * self.zoom
        vx, vy = self.view_x, self.view_y
        l = max(0, int(-vx / s))
        t = max(0, int(-vy / s))
        r = min(W, int((CANVAS_W - vx) / s) + 1)
        b = min(H, int((CANVAS_H - vy) / s) + 1)
        if r <= l or b <= t:
            return
        crop = bgr[t:b, l:r]
        cw = max(1, int((r - l) * s))
        ch = max(1, int((b - t) * s))
        interp = cv2.INTER_NEAREST if self.zoom > 1.5 else cv2.INTER_AREA
        resized = cv2.resize(crop, (cw, ch), interpolation=interp)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        canvas.create_image(vx + l * s, vy + t * s, anchor="nw", image=photo)
        setattr(self, attr, photo)

    def _refresh(self):
        if self.original is None:
            return
        self._render(self.ic, self._build_input_disp(), "input_photo")
        self._render(self.rc, self.result_img,          "result_photo")

    def _build_input_disp(self):
        disp = self.original.copy()
        try:
            r_cam = int(self.radius_var.get())
        except ValueError:
            r_cam = 450
        for i, (cx, cy, r) in enumerate(self.centers):
            cv2.circle(disp, (cx, cy), r_cam, (0, 255, 0), 3)
            cv2.circle(disp, (cx, cy), 8, (0, 0, 255), -1)
            cv2.putText(disp, str(i+1), (cx+12, cy-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        return disp

    # ---------- centers ----------
    def _add_center(self, event):
        if self.original is None:
            return
        s = self.base_scale * self.zoom
        ix = int((event.x - self.view_x) / s)
        iy = int((event.y - self.view_y) / s)
        h, w = self.original.shape[:2]
        if 0 <= ix < w and 0 <= iy < h:
            try:
                r = int(self.radius_var.get())
            except ValueError:
                r = 450
            self.centers.append((ix, iy, r))
            self._refresh()
            self.status.set(f"{len(self.centers)} center(s) added.")

    def undo(self):
        if self.centers:
            self.centers.pop()
            self._refresh()
            self.status.set(f"Undo. {len(self.centers)} center(s) remain.")

    # ---------- detection ----------
    def run(self):
        if self.original is None or not self.centers:
            self.status.set("Load image and add at least one center first.")
            return
        try:
            win = int(self.win_var.get())
            if win % 2 == 0:
                win += 1
            z_thr = float(self.zthr_var.get())
        except ValueError:
            win = 31
            z_thr = 3.0

        gray   = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32)

        local_mean    = cv2.boxFilter(gray_f, -1, (win, win))
        local_mean_sq = cv2.boxFilter(gray_f * gray_f, -1, (win, win))
        local_std     = np.sqrt(np.maximum(local_mean_sq - local_mean * local_mean, 0))
        zscore        = np.where(local_std > 1e-5,
                                 (gray_f - local_mean) / local_std, 0.0)

        mask = np.zeros(gray.shape, dtype=np.uint8)
        try:
            r_cam = int(self.radius_var.get())
        except ValueError:
            r_cam = 450
        for (cx, cy, r) in self.centers:
            cv2.circle(mask, (cx, cy), r_cam, 255, -1)

        binary = np.where((zscore >= z_thr) & (mask == 255), 255, 0).astype(np.uint8)

        cv2.imwrite("zscore_binary.png", binary)
        max_z = zscore[mask == 255].max()
        print(f"max z={max_z:.2f}  mean z={zscore[mask==255].mean():.2f}")

        self.result_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self._refresh()
        self.status.set(f"Done. Z thr={z_thr}, max z={max_z:.2f}. Saved zscore_binary.png")


root = tk.Tk()
ZscoreViewer(root)
root.mainloop()
