"""
Detect and crop camera circles from Samsung phone back images.
Handles: S26 (black), S25 Edge (silver, black camera module), S26 Ultra (black).
"""

import cv2
import numpy as np
import os
from pathlib import Path


def detect_and_crop_cameras(
    image_path,
    output_dir=None,
    model_type='auto',
    debug=False,
    # tunable params
    min_radius_pct=0.015,    # smallest circle as % of min(h, w)
    max_radius_pct=0.13,     # largest circle as % of min(h, w)
    min_dist_pct=0.04,       # min spacing between circle centers
    param1=60,               # Canny upper threshold
    param2=28,               # accumulator threshold (lower => more circles)
    crop_padding_pct=0.15,   # extra padding around the crop
):
    """
    Detect camera lens circles in a phone back image and crop them out.

    Returns: (list of cropped BGR images, ndarray of (x, y, r) circles)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    original = img.copy()
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Auto-detect phone color if not specified
    if model_type == 'auto':
        mean_brightness = float(np.mean(gray))
        model_type = 's25_edge' if mean_brightness > 130 else 's26'

    # Different preprocessing for bright vs dark phones
    if model_type == 's25_edge':
        # Silver body, black camera module → camera ring is the darkest blob.
        # Slight blur is enough; lenses pop naturally.
        processed = cv2.GaussianBlur(gray, (9, 9), 2)

        # Optional: make the dark module even more distinct
        # by clipping the bright phone body
        _, dark_mask = cv2.threshold(processed, 90, 255, cv2.THRESH_BINARY_INV)
        # Use the mask to emphasize dark regions (subtle help to Hough)
        processed = cv2.bitwise_and(processed, processed, mask=dark_mask) \
                    if np.any(dark_mask) else processed
    else:
        # Black phone, black camera ring — low contrast.
        # CLAHE boosts the subtle highlights inside the lens (glass reflections).
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        processed = cv2.GaussianBlur(enhanced, (9, 9), 2)

    min_radius = max(8, int(min(h, w) * min_radius_pct))
    max_radius = int(min(h, w) * max_radius_pct)
    min_dist = max(20, int(min(h, w) * min_dist_pct))

    circles = cv2.HoughCircles(
        processed,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    cropped_images = []
    if circles is None:
        if debug:
            print(f"[{Path(image_path).name}] No circles detected. "
                  f"Try lowering param2 or adjusting radius bounds.")
        return cropped_images, None

    circles = np.round(circles[0, :]).astype("int")

    # Optional: filter circles whose interior is too bright
    # (e.g. logos, flash highlights). Lens centers tend to be dark.
    filtered = []
    for x, y, r in circles:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), max(2, int(r * 0.6)), 255, -1)
        inner_mean = cv2.mean(gray, mask=mask)[0]
        # Lenses interior should be relatively dark
        if inner_mean < 160:
            filtered.append((x, y, r, inner_mean))
    if filtered:
        circles = np.array([[x, y, r] for x, y, r, _ in filtered])

    # Sort top-to-bottom, then left-to-right
    circles = sorted(circles.tolist(), key=lambda c: (c[1] // 30, c[0]))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    base = Path(image_path).stem

    vis = original.copy() if debug else None

    for i, (x, y, r) in enumerate(circles):
        pad = int(r * crop_padding_pct)
        x1, y1 = max(0, x - r - pad), max(0, y - r - pad)
        x2, y2 = min(w, x + r + pad), min(h, y + r + pad)
        crop = original[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        cropped_images.append(crop)

        if output_dir:
            cv2.imwrite(os.path.join(output_dir, f"{base}_cam_{i+1}.png"), crop)

        if debug:
            cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vis, f"{i+1}", (x - 10, y + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if debug and vis is not None:
        out_vis = os.path.join(output_dir or '.', f"{base}_debug.png")
        cv2.imwrite(out_vis, vis)
        print(f"[{base}] {len(cropped_images)} cameras → {out_vis}")

    return cropped_images, np.array(circles)


# ---- Per-model presets ---------------------------------------------------
MODEL_PRESETS = {
    's26':       dict(model_type='s26',       param2=26, max_radius_pct=0.11),
    's25_edge':  dict(model_type='s25_edge',  param2=30, max_radius_pct=0.10),
    's26_ultra': dict(model_type='s26_ultra', param2=24, max_radius_pct=0.14),
}


def process_phone(image_path, model_key, out_root='output', debug=True):
    """Run detection with the right preset for a given model."""
    preset = MODEL_PRESETS.get(model_key, {})
    out_dir = os.path.join(out_root, model_key)
    crops, circles = detect_and_crop_cameras(
        image_path, output_dir=out_dir, debug=debug, **preset
    )
    print(f"{model_key}: {len(crops)} camera(s) cropped → {out_dir}")
    return crops, circles


if __name__ == "__main__":
    # Edit these paths for your images
    jobs = [
        ('s26_back.jpg',       's26'),
        ('s25_edge_back.jpg',  's25_edge'),
        ('s26_ultra_back.jpg', 's26_ultra'),
    ]
    for path, key in jobs:
        if os.path.exists(path):
            process_phone(path, key)
        else:
            print(f"skip (missing): {path}")
