"""
Phone-camera dust inspection GUI.

Two-stage pipeline:

    Stage 1 — Lens detection
        grayscale -> blur -> threshold (Otsu or manual) -> morph close
        -> find contours -> filter by area + circularity

    Stage 2 — Dust detection (inside each detected lens)
        shrink outer circle by `inner_pct` to get the glass ROI
        (drops the outer plastic ring)
        convert to HSV inside that ROI
        drop high-saturation pixels   <-- this kills AR-coating reflections
                                          (dust is achromatic, AR is colored)
        keep dark pixels (V < max_value)
        connected components -> filter by tiny pixel-area range
        => each surviving component is a dust spot.

The "Show crops" button opens a popup that zooms in on each lens with
dust spots circled in red, so the tiny dots are actually visible.

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


DISPLAY_MAX = (420, 460)
RESAMPLE = (Image.Resampling.LANCZOS
            if hasattr(Image, "Resampling") else Image.LANCZOS)


# ============================================================================
# Pure CV functions — no GUI dependencies
# ============================================================================

def preprocess(img_bgr, *, blur, threshold, use_otsu, invert, morph_close):
    """Grayscale -> blur -> threshold -> morph close. Returns binary mask."""
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


def find_lens_blobs(binary, *, min_area_pct, max_area_pct, min_circularity):
    """Return [(cx, cy, outer_r, circularity), ...] sorted top→bottom, L→R."""
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    H, W = binary.shape
    img_area = float(H * W)
    a_min = img_area * float(min_area_pct) / 100.0
    a_max = img_area * float(max_area_pct) / 100.0
    c_min = float(min_circularity)

    out = []
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
        out.append((int(x), int(y), int(r), float(circ)))
    out.sort(key=lambda v: (v[1] // 30, v[0]))
    return out


def detect_dust_in_lens(img_bgr, cx, cy, outer_r, *,
                        inner_pct, max_saturation, max_value,
                        min_dust_area, max_dust_area):
    """Find dust spots inside one lens.

    Returns (dust_spots, inner_r, dust_mask).
      dust_spots = [(x, y, area), ...]  centers in full-image coords.
    """
    H, W = img_bgr.shape[:2]
    inner_r = max(1, int(outer_r * float(inner_pct)))

    # circular ROI: only consider pixels inside the inner lens
    roi = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(roi, (cx, cy), inner_r, 255, -1)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1]  # saturation: high = colorful (AR coating)
    v = hsv[..., 2]  # value:      low  = dark (dust candidate)

    # achromatic AND dark
    achromatic = s < int(max_saturation)
    is_dark    = v < int(max_value)
    candidate  = (achromatic & is_dark).astype(np.uint8) * 255
    dust_mask  = cv2.bitwise_and(candidate, roi)

    num, _labels, stats, centroids = \
        cv2.connectedComponentsWithStats(dust_mask, connectivity=8)

    spots = []
    a_min, a_max = int(min_dust_area), int(max_dust_area)
    for i in range(1, num):  # 0 = background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if a_min <= area <= a_max:
            x, y = centroids[i]
            spots.append((int(round(x)), int(round(y)), area))
    return spots, inner_r, dust_mask


def annotate_full(img_bgr, lenses, inner_radii, all_dust):
    """Draw lens outer ring (green) + inner ROI ring (cyan) + dust (red)."""
    out = img_bgr.copy()
    H, W = img_bgr.shape[:2]
    # Make dust markers visible after thumbnail downscale
    m_r = max(10, int(min(H, W) * 0.008))
    m_t = max(2,  int(min(H, W) * 0.002))

    for i, ((x, y, r, _c), ir) in enumerate(zip(lenses, inner_radii), 1):
        cv2.circle(out, (x, y), r,  (0, 255, 0), 3)
        cv2.circle(out, (x, y), ir, (255, 200, 0), 2)
        cv2.putText(out, str(i), (x - 12, y - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    for spots in all_dust:
        for dx, dy, _a in spots:
            cv2.circle(out, (dx, dy), m_r, (0, 0, 255), m_t)
    return out


def crop_lens_with_dust(img_bgr, cx, cy, outer_r, inner_r, dust_spots,
                        padding_pct=0.10):
    """Return a cropped lens view with the inner ROI ring + dust markers."""
    H, W = img_bgr.shape[:2]
    pad = int(outer_r * padding_pct)
    x1 = max(0, cx - outer_r - pad)
    y1 = max(0, cy - outer_r - pad)
    x2 = min(W, cx + outer_r + pad)
    y2 = min(H, cy + outer_r + pad)
    crop = img_bgr[y1:y2, x1:x2].copy()

    # Local coordinates of the lens centre within the crop
    lcx, lcy = cx - x1, cy - y1
    cv2.circle(crop, (lcx, lcy), inner_r, (255, 200, 0), 1)

    for dx, dy, area in dust_spots:
        local = (dx - x1, dy - y1)
        # draw a ring whose radius reflects blob area, never tinier than 4
        r_draw = max(4, int(np.sqrt(area / np.pi)) + 4)
        cv2.circle(crop, local, r_draw, (0, 0, 255), 2)
    return crop


# ============================================================================
# GUI
# ============================================================================

class DetectorGUI:
    LENS_SLIDERS = [
        ('blur',            'blur kernel (odd; 1 = none)',          1,    21,    5,    True),
        ('threshold',       'threshold (used if Otsu off)',         0,    255,   80,   True),
        ('morph_close',     'morphology close (0 = off)',           0,    15,    2,    True),
        ('min_area_pct',    'min lens area  (% of image)',          0.001, 5.0,  0.01, False),
        ('max_area_pct',    'max lens area  (% of image)',          0.05,  20.0, 5.0,  False),
        ('min_circularity', 'min circularity (1.0 = perfect)',      0.30,  1.00, 0.70, False),
    ]
    DUST_SLIDERS = [
        ('inner_pct',      'inner ROI radius  (× outer r)',         0.30,  1.00, 0.80, False),
        ('max_saturation', 'max saturation   (drop AR coating)',    0,     255,  60,   True),
        ('max_value',      'dust max brightness  (dark = lower)',   0,     255,  110,  True),
        ('min_dust_area',  'min dust area  (px)',                   1,     50,   2,    True),
        ('max_dust_area',  'max dust area  (px)',                   5,     500,  60,   True),
    ]

    def __init__(self, root, image_path=None):
        self.root = root
        self.root.title("Phone Camera Dust Inspector")
        self.root.geometry("1500x1020")

        self.original = None
        self.last_lenses = []         # [(x, y, outer_r, circ)]
        self.last_inner_r = []        # [inner_r per lens]
        self.last_dust = []           # [[(x, y, area), ...] per lens]
        self.params = {}
        self.value_labels = {}
        self.crops_window = None

        self._build_ui()
        if image_path and os.path.exists(image_path):
            self._load(image_path)

    # ---- UI ----------------------------------------------------------------
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=(10, 10, 10, 4))
        top.pack(fill=tk.X)
        ttk.Button(top, text="Load image…", command=self._browse).pack(side=tk.LEFT)
        self.file_label = ttk.Label(top, text="(no image)", foreground="#555")
        self.file_label.pack(side=tk.LEFT, padx=10)

        # Two side-by-side parameter columns to save vertical space
        params_row = ttk.Frame(self.root, padding=(10, 0))
        params_row.pack(fill=tk.X)

        lens_box = ttk.LabelFrame(params_row, text="Stage 1 — Lens detection",
                                  padding=10)
        lens_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        for cfg in self.LENS_SLIDERS:
            self._add_slider(lens_box, *cfg)

        cb = ttk.Frame(lens_box)
        cb.pack(fill=tk.X, pady=(6, 0))
        self.use_otsu = tk.BooleanVar(value=True)
        self.invert = tk.BooleanVar(value=True)
        ttk.Checkbutton(cb, text="Use Otsu auto-threshold",
                        variable=self.use_otsu).pack(side=tk.LEFT)
        ttk.Checkbutton(cb, text="Invert (target dark regions)",
                        variable=self.invert).pack(side=tk.LEFT, padx=14)

        dust_box = ttk.LabelFrame(params_row,
                                  text="Stage 2 — Dust detection inside each lens",
                                  padding=10)
        dust_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        for cfg in self.DUST_SLIDERS:
            self._add_slider(dust_box, *cfg)

        # Action row
        actions = ttk.Frame(self.root, padding=(10, 8))
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

        # Image panels
        panels = ttk.Frame(self.root, padding=10)
        panels.pack(fill=tk.BOTH, expand=True)
        self.orig_lbl = self._make_panel(panels, "Original")
        self.bin_lbl  = self._make_panel(panels, "Binary (lens stage)")
        self.out_lbl  = self._make_panel(panels,
                                         "Detected  (green=lens, cyan=ROI, red=dust)")

    def _make_panel(self, parent, title):
        box = ttk.LabelFrame(parent, text=title, padding=4)
        box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        lbl = tk.Label(box, bg="#1f2937")
        lbl.pack(fill=tk.BOTH, expand=True)
        return lbl

    def _add_slider(self, parent, key, label, mn, mx, default, is_int):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)

        ttk.Label(row, text=label, width=34).pack(side=tk.LEFT)

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
        return str(int(value)) if is_int else f"{float(value):.4g}"

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
        self.last_lenses, self.last_inner_r, self.last_dust = [], [], []

    def _gather(self):
        g = lambda k: self.params[k].get()
        return dict(
            blur=int(g('blur')),
            threshold=int(g('threshold')),
            use_otsu=bool(self.use_otsu.get()),
            invert=bool(self.invert.get()),
            morph_close=int(g('morph_close')),
            min_area_pct=float(g('min_area_pct')),
            max_area_pct=float(g('max_area_pct')),
            min_circularity=float(g('min_circularity')),
            inner_pct=float(g('inner_pct')),
            max_saturation=int(g('max_saturation')),
            max_value=int(g('max_value')),
            min_dust_area=int(g('min_dust_area')),
            max_dust_area=int(g('max_dust_area')),
        )

    def run(self):
        if self.original is None:
            self.status.config(text="Load an image first.", foreground="red")
            return

        p = self._gather()

        # Stage 1
        binary = preprocess(self.original,
                            blur=p['blur'], threshold=p['threshold'],
                            use_otsu=p['use_otsu'], invert=p['invert'],
                            morph_close=p['morph_close'])
        lenses = find_lens_blobs(binary,
                                 min_area_pct=p['min_area_pct'],
                                 max_area_pct=p['max_area_pct'],
                                 min_circularity=p['min_circularity'])

        # Stage 2
        inner_radii, all_dust = [], []
        for (x, y, r, _c) in lenses:
            spots, ir, _dm = detect_dust_in_lens(
                self.original, x, y, r,
                inner_pct=p['inner_pct'],
                max_saturation=p['max_saturation'],
                max_value=p['max_value'],
                min_dust_area=p['min_dust_area'],
                max_dust_area=p['max_dust_area'],
            )
            inner_radii.append(ir)
            all_dust.append(spots)

        self.last_lenses = lenses
        self.last_inner_r = inner_radii
        self.last_dust = all_dust

        annotated = annotate_full(self.original, lenses, inner_radii, all_dust)
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        self._show(self.original, self.orig_lbl)
        self._show(binary_bgr, self.bin_lbl)
        self._show(annotated, self.out_lbl)

        n_lens = len(lenses)
        n_dust = sum(len(s) for s in all_dust)
        color = "green" if n_lens else "red"
        self.status.config(
            text=f"{n_lens} lens(es), {n_dust} dust spot(s) total. "
                 f"Open 'Show crops' to inspect each lens.",
            foreground=color)

    def _reset(self):
        for cfg in self.LENS_SLIDERS + self.DUST_SLIDERS:
            key, _l, _mn, _mx, default, is_int = cfg
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
        if self.last_lenses:
            print(f"\nLenses ({len(self.last_lenses)}):")
            for i, ((x, y, r, c), ir, spots) in enumerate(
                    zip(self.last_lenses, self.last_inner_r, self.last_dust), 1):
                print(f"  Lens #{i}: center=({x},{y}) outer_r={r} inner_r={ir} "
                      f"circ={c:.3f}  dust={len(spots)}")
                for j, (dx, dy, area) in enumerate(spots, 1):
                    print(f"    dust #{j}: ({dx},{dy}) area={area}px")

    def _show_crops(self):
        if not self.last_lenses:
            self.status.config(text="Run detection first.", foreground="red")
            return

        if self.crops_window is not None and self.crops_window.winfo_exists():
            for w in self.crops_window.winfo_children():
                w.destroy()
        else:
            self.crops_window = tk.Toplevel(self.root)
            self.crops_window.title("Per-lens inspection — red = dust")

        cols = min(max(len(self.last_lenses), 1), 4)
        for i, ((x, y, r, _c), ir, spots) in enumerate(
                zip(self.last_lenses, self.last_inner_r, self.last_dust)):
            row, col = divmod(i, cols)
            frame = ttk.LabelFrame(
                self.crops_window,
                text=f"Lens #{i+1}  —  {len(spots)} dust spot(s)",
                padding=4)
            frame.grid(row=row, column=col, padx=6, pady=6)

            crop = crop_lens_with_dust(self.original, x, y, r, ir, spots)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            pil.thumbnail((320, 320), RESAMPLE)
            tk_img = ImageTk.PhotoImage(pil)

            lbl = tk.Label(frame, image=tk_img)
            lbl.image = tk_img
            lbl.pack()

    def _show(self, bgr, widget):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail(DISPLAY_MAX, RESAMPLE)
        tk_img = ImageTk.PhotoImage(pil)
        widget.config(image=tk_img)
        widget.image = tk_img


if __name__ == "__main__":
    initial = sys.argv[1] if len(sys.argv) > 1 else "img29.png"
    root = tk.Tk()
    DetectorGUI(root, initial if os.path.exists(initial) else None)
    root.mainloop()
