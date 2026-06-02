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

binary = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    51, -5
)

binary_masked = cv2.bitwise_and(binary, mask)

contours, _ = cv2.findContours(binary_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 2:
        continue
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue
    circularity = (4 * math.pi * area) / (perimeter ** 2)
    print(f"area={area:.0f}  circ={circularity:.2f}")
