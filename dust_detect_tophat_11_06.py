import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

CANVAS_W = 600
CANVAS_H = 600

class TophatViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Tophat Viewer")
        self.original = None
        self.centers = []
        self.scale = 1.0
        self.off_x = 0
        self.off_y = 0
        self.input_photo = None
        self.result_photo = None

        # controls
        bar = ttk.Frame(root, padding=6)
        bar.grid(row=0, column=0, columnspan=2)
        ttk.Button(bar, text="Load Image", command=self.load).pack(side=tk.LEFT, padx=3)
        ttk.Label(bar, text="Radius:").pack(side=tk.LEFT, padx=(10,2))
        self.radius_var = tk.StringVar(value="450")
        ttk.Entry(bar, textvariable=self.radius_var, width=6).pack(side=tk.LEFT)
        ttk.Label(bar, text="Kernel:").pack(side=tk.LEFT, padx=(10,2))
        self.kernel_var = tk.StringVar(value="41")
        ttk.Entry(bar, textvariable=self.kernel_var, width=5).pack(side=tk.LEFT)
        ttk.Button(bar, text="Undo", command=self.undo).pack(side=tk.LEFT, padx=10)
        ttk.Button(bar, text="Run Tophat", command=self.run).pack(side=tk.LEFT, padx=3)
        self.status = tk.StringVar(value="Load an image.")
        ttk.Label(bar, textvariable=self.status).pack(side=tk.LEFT, padx=10)

        # canvases
        left = ttk.LabelFrame(root, text="INPUT (click = add center)")
        left.grid(row=1, column=0, padx=6, pady=6)
        self.ic = tk.Canvas(left, width=CANVAS_W, height=CANVAS_H, bg="gray20")
        self.ic.pack()
        self.ic.bind("<Button-1>", self.on_click)

        right = ttk.LabelFrame(root, text="TOPHAT OUTPUT (boosted x5)")
        right.grid(row=1, column=1, padx=6, pady=6)
        self.rc = tk.Canvas(right, width=CANVAS_W, height=CANVAS_H, bg="gray20")
        self.rc.pack()

    def load(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")])
        if not path:
            return
        self.original = cv2.imread(path)
        self.centers = []
        self.rc.delete("all")
        self._fit_and_draw()
        self.status.set("Loaded. Click camera centers then Run Tophat.")

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
        for i, (cx, cy, r) in enumerate(self.centers):
            cv2.circle(disp, (cx, cy), r, (0, 255, 0), 4)
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
            k = int(self.kernel_var.get())
            if k % 2 == 0:
                k += 1
        except ValueError:
            k = 41

        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)

        mask = np.zeros(gray.shape, dtype=np.uint8)
        for (cx, cy, r) in self.centers:
            cv2.circle(mask, (cx, cy), r, 255, -1)
        tophat = cv2.bitwise_and(tophat, mask)

        boosted = cv2.convertScaleAbs(tophat, alpha=5.0, beta=0)
        boosted_bgr = cv2.cvtColor(boosted, cv2.COLOR_GRAY2BGR)

        cv2.imwrite("tophat_raw.png", tophat)
        cv2.imwrite("tophat_boosted.png", boosted)

        self._show(self.rc, boosted_bgr, "result_photo")
        self.status.set(f"Done. Max tophat value: {tophat.max()}. "
                        f"Files saved: tophat_raw.png, tophat_boosted.png")


root = tk.Tk()
TophatViewer(root)
root.mainloop()
