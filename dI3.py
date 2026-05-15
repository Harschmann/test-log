"""
Detect camera circles in a phone-back image and SHOW original vs result
side-by-side. Nothing is saved to disk.

Usage:
    python detect_phone_cameras.py                       # img29.png + s26
    python detect_phone_cameras.py img29.png s25_edge
    python detect_phone_cameras.py img29.png s26_ultra
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


MODEL_PRESETS = {
    's26':       dict(model_type='s26',       param2=26, max_radius_pct=0.11),
    's25_edge':  dict(model_type='s25_edge',  param2=30, max_radius_pct=0.10),
    's26_ultra': dict(model_type='s26_ultra', param2=24, max_radius_pct=0.14),
}


def detect_cameras(
    img,
    model_type='s26',
    min_radius_pct=0.015,
    max_radius_pct=0.13,
    min_dist_pct=0.04,
    param1=60,
    param2=28,
):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if model_type == 's25_edge':
        # Silver body, black camera module -> isolate dark regions
        processed = cv2.GaussianBlur(gray, (9, 9), 2)
        _, dark_mask = cv2.threshold(processed, 90, 255, cv2.THRESH_BINARY_INV)
        if np.any(dark_mask):
            processed = cv2.bitwise_and(processed, processed, mask=dark_mask)
    else:
        # Black phone -> boost subtle lens reflections with CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        processed = cv2.GaussianBlur(clahe.apply(gray), (9, 9), 2)

    circles = cv2.HoughCircles(
        processed,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, int(min(h, w) * min_dist_pct)),
        param1=param1,
        param2=param2,
        minRadius=max(8, int(min(h, w) * min_radius_pct)),
        maxRadius=int(min(h, w) * max_radius_pct),
    )
    if circles is None:
        return np.empty((0, 3), dtype=int)

    circles = np.round(circles[0, :]).astype(int)

    # Drop circles whose interior is too bright (logos, flash reflections)
    keep = []
    for x, y, r in circles:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), max(2, int(r * 0.6)), 255, -1)
        if cv2.mean(gray, mask=mask)[0] < 160:
            keep.append((x, y, r))
    if not keep:
        return np.empty((0, 3), dtype=int)

    # Sort top-to-bottom, then left-to-right
    return np.array(sorted(keep, key=lambda c: (c[1] // 30, c[0])))


def show_side_by_side(img_bgr, circles, title=''):
    """Display original + annotated + each crop in one matplotlib window."""
    annotated = img_bgr.copy()
    h, w = img_bgr.shape[:2]

    crops = []
    for i, (x, y, r) in enumerate(circles):
        pad = int(r * 0.15)
        x1, y1 = max(0, x - r - pad), max(0, y - r - pad)
        x2, y2 = min(w, x + r + pad), min(h, y + r + pad)
        crops.append(img_bgr[y1:y2, x1:x2])

        cv2.circle(annotated, (x, y), r, (0, 255, 0), 3)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(annotated, f"{i+1}", (x - 12, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)

    bgr2rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    n_crops = len(crops)
    cols = max(2, n_crops)
    fig = plt.figure(figsize=(min(18, 4 * cols), 8))

    ax1 = plt.subplot2grid((2, cols), (0, 0), colspan=cols // 2)
    ax1.imshow(bgr2rgb(img_bgr)); ax1.set_title('Original'); ax1.axis('off')

    ax2 = plt.subplot2grid((2, cols), (0, cols // 2),
                           colspan=cols - cols // 2)
    ax2.imshow(bgr2rgb(annotated))
    ax2.set_title(f'Detected: {n_crops} camera(s)'); ax2.axis('off')

    for i, crop in enumerate(crops):
        ax = plt.subplot2grid((2, cols), (1, i))
        ax.imshow(bgr2rgb(crop)); ax.set_title(f'Camera {i+1}'); ax.axis('off')

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "img29.png"
    model_key  = sys.argv[2] if len(sys.argv) > 2 else "s26"

    if not os.path.exists(image_path):
        raise SystemExit(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit(f"Cannot read: {image_path}")

    preset = MODEL_PRESETS.get(model_key, {})
    circles = detect_cameras(img, **preset)
    print(f"{model_key}: detected {len(circles)} camera(s)")
    show_side_by_side(img, circles, title=f"{image_path} ({model_key})")
