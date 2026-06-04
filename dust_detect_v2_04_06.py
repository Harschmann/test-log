import cv2
import numpy as np
import math

img = cv2.imread("test2.png")

centers = [(821, 1630), (2207, 1630), (3613, 1630)]
radii   = [300, 300, 300]

mask = np.zeros(img.shape[:2], dtype=np.uint8)
for c, r in zip(centers, radii):
    cv2.circle(mask, c, r, 255, -1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

result = img.copy()
dust_count = 0
detected = np.zeros(img.shape[:2], dtype=np.uint8)   # double-count rokne ke liye

# ===== PASS A: bright dust (tera original method, koi shape filter nahi) =====
_, binary_bright = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
binary_bright = cv2.bitwise_and(binary_bright, mask)

cont_b, _ = cv2.findContours(binary_bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in cont_b:
    area = cv2.contourArea(cnt)
    if area < 3:
        continue
    (x, y), r = cv2.minEnclosingCircle(cnt)
    cv2.circle(result, (int(x), int(y)), max(10, int(r) + 4), (0, 0, 255), 3)
    cv2.circle(detected, (int(x), int(y)), int(r) + 20, 255, -1)
    dust_count += 1

# ===== PASS B: faint dust via tophat (shape filter rings ke liye) =====
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
tophat_masked = cv2.bitwise_and(tophat, mask)
_, binary_faint = cv2.threshold(tophat_masked, 15, 255, cv2.THRESH_BINARY)

cont_f, _ = cv2.findContours(binary_faint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in cont_f:
    area = cv2.contourArea(cnt)
    if area < 3 or area > 500:
        continue
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue
    circularity = (4 * math.pi * area) / (perimeter ** 2)
    if circularity < 0.4:
        continue
    (x, y), r = cv2.minEnclosingCircle(cnt)
    if detected[int(y), int(x)] == 255:   # bright pass mein already mila
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
