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
        self.scale = 1.0
        self.off_x = 0
        self.off_y = 0
        self.input_photo = None
        self.result_photo = None

        bar = ttk.Frame(root, padding=6)
        bar.grid(row=0, column=0, columnspan=2)
        ttk.Button(bar, text="Load Image", command=self.load).pack(side=tk.LEFT, padx=3)
        ttk.Label(bar, text="Radius:").pack(side=tk.LEFT, padx=(10,2))
        self.radius_var = tk.StringVar(value="450")
        ttk.Entry(bar, textvariable=self.radius_var, width=6).pack(side=tk.LEFT)
        ttk.Label(bar, text="Window:").pack(side=tk.LEFT, padx=(10,2))
        self.win_var = tk.StringVar(value="31")
        ttk.Entry(bar, textvariable=self.win_var, width=5).pack(side=tk.LEFT)
        ttk.Label(bar, text="Z threshold:").pack(side=tk.LEFT, padx=(10,2))
        self.zthr_var = tk.StringVar(value="3.0")
        ttk.Entry(bar, textvariable=self.zthr_var, width=5).pack(side=tk.LEFT)
        ttk.Button(bar, text="Undo", command=self.undo).pack(side=tk.LEFT, padx=6)
        ttk.Button(bar, text="Run", command=self.run).pack(side=tk.LEFT, padx=3)
        self.status = tk.StringVar(value="Load an image.")
        ttk.Label(bar, textvariable=self.status).pack(side=tk.LEFT, padx=10)

        left = ttk.LabelFrame(root, text="INPUT")
        left.grid(row=1, column=0, padx=6, pady=6)
        self.ic = tk.Canvas(left, width=CANVAS_W, height=CANVAS_H, bg="gray20")
        self.ic.pack()
        self.ic.bind("<Button-1>", self.on_click)

        right = ttk.LabelFrame(root, text="Z-SCORE OUTPUT (white = abnormally bright = dust)")
        right.grid(row=1, column=1, padx=6, pady=6)
        self.rc = tk.Canvas(right, width=CANVAS_W, height=CANVAS_H, bg="gray20")
        self.rc.pack()

    def load(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"), ("All", "*.*")])
        if not path:
            return
        self.original = cv2.imread(path, cv2.IMREAD_COLOR)
        if self.original is None:
            self.status.set("Could not load image.")
            return
        self.centers = []
        self.rc.delete("all")
        self._fit_and_draw()
        self.status.set("Loaded. Click camera centers then Run.")

    def _fit_and_draw(self):
        if self.original is None:
            return
        h, w = self.original.shape[:2]
        self.scale = min(CANVAS_W / w, CANVAS_H / h)
        self.off_x = int((CANVAS_W - w * self.scale) / 2)
        self.off_y = int((CANVAS_H - h * self.scale) / 2)
        self._draw_input()

    def _draw_input(self):
        if self.original is None:
            return
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
        self._show(self.ic, disp, "input_photo")

    def _show(self, canvas, bgr, attr):
        h, w = bgr.shape[:2]
        sw = max(1, int(w * self.scale))
        sh = max(1, int(h * self.scale))
        rgb = cv2.cvtColor(cv2.resize(bgr, (sw, sh)), cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        canvas.delete("all")
        canvas.create_image(self.off_x, self.off_y, anchor="nw", image=photo)
        setattr(self, attr, photo)

    def on_click(self, event):
        if self.original is None:
            return
        ix = int((event.x - self.off_x) / self.scale)
        iy = int((event.y - self.off_y) / self.scale)
        h, w = self.original.shape[:2]
        if 0 <= ix < w and 0 <= iy < h:
            try:
                r = int(self.radius_var.get())
            except ValueError:
                r = 450
            self.centers.append((ix, iy, r))
            self._draw_input()
            self.status.set(f"{len(self.centers)} center(s) added.")

    def undo(self):
        if self.centers:
            self.centers.pop()
            self._draw_input()
            self.status.set(f"Undo. {len(self.centers)} center(s) remain.")

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

        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32)

        # local mean and std using box filter
        local_mean = cv2.boxFilter(gray_f, -1, (win, win))
        local_mean_sq = cv2.boxFilter(gray_f * gray_f, -1, (win, win))
        local_var = local_mean_sq - local_mean * local_mean
        local_var = np.maximum(local_var, 0)
        local_std = np.sqrt(local_var)

        # z-score: how many std devs above local mean
        zscore = np.where(local_std > 1e-5,
                          (gray_f - local_mean) / local_std,
                          0.0)

        # mask
        mask = np.zeros(gray.shape, dtype=np.uint8)
        try:
            r_cam = int(self.radius_var.get())
        except ValueError:
            r_cam = 450
        for (cx, cy, r) in self.centers:
            cv2.circle(mask, (cx, cy), r_cam, 255, -1)

        # threshold on z-score
        binary = np.where((zscore >= z_thr) & (mask == 255), 255, 0).astype(np.uint8)

        # save
        cv2.imwrite("zscore_binary.png", binary)
        print("max z-score inside mask:", zscore[mask == 255].max().round(2))
        print("mean z-score inside mask:", zscore[mask == 255].mean().round(2))

        self._show(self.rc, cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), "result_photo")
        self.status.set(f"Done. Z thr={z_thr}. Max z={zscore[mask==255].max():.2f}. "
                        f"Saved zscore_binary.png")


root = tk.Tk()
ZscoreViewer(root)
root.mainloop()
