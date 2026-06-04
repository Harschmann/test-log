import cv2
import numpy as np
import math

# ===================== CONFIG =====================
IMAGE_PATH = "test2.png"

CENTERS = [(821, 1630), (2207, 1630), (3613, 1630)]
RADII   = [450, 450, 450]

THRESHOLD_BRIGHT = 200    # bright/bada dust ke liye
THRESHOLD_FAINT  = 15     # faint/chure dust ke liye (tophat ke baad)
TOPHAT_KERNEL    = 41     # dust se thoda bada rakho
MIN_AREA         = 3
MAX_AREA         = 500
CIRCULARITY_MIN  = 0.4    # isse kam = ring/arc, skip
# ==================================================

img = cv2.imread(IMAGE_PATH)

mask = np.zeros(img.shape[:2], dtype=np.uint8)
for c, r in zip(CENTERS, RADII):
    cv2.circle(mask, c, r, 255, -1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

result = img.copy()
dust_count = 0

# ===================== BADA / BRIGHT DUST =====================
big_detected = np.zeros(img.shape[:2], dtype=np.uint8)
_, binary_big = cv2.threshold(blurred, THRESHOLD_BRIGHT, 255, cv2.THRESH_BINARY)
binary_big = cv2.bitwise_and(binary_big, mask)
cont_big, _ = cv2.findContours(binary_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in cont_big:
    area = cv2.contourArea(cnt)
    if area < MIN_AREA:
        continue
    (x, y), r = cv2.minEnclosingCircle(cnt)
    cv2.circle(result, (int(x), int(y)), max(10, int(r) + 4), (0, 0, 255), 3)
    cv2.circle(big_detected, (int(x), int(y)), int(r) + 20, 255, -1)
    dust_count += 1

# ===================== FAINT / CHURE DUST (tophat) =====================
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_KERNEL, TOPHAT_KERNEL))
tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
tophat_masked = cv2.bitwise_and(tophat, mask)

_, binary = cv2.threshold(tophat_masked, THRESHOLD_FAINT, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < MIN_AREA or area > MAX_AREA:
        continue
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue
    circularity = (4 * math.pi * area) / (perimeter ** 2)
    if circularity < CIRCULARITY_MIN:
        continue
    (x, y), r = cv2.minEnclosingCircle(cnt)
    if big_detected[int(y), int(x)] == 255:   # already counted in bright pass
        continue
    cv2.circle(result, (int(x), int(y)), max(10, int(r) + 4), (0, 0, 255), 3)
    dust_count += 1

print("Dust found:", dust_count)

cv2.imwrite("result.png", result)
print("saved result.png")

cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
