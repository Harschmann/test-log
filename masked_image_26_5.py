import cv2
import numpy as np

img = cv2.imread("test.jpg")

centers = [(2409, 1046), (1010, 1012), (3807, 1063)]
radii   = [325, 300, 250]

mask = np.zeros(img.shape[:2], dtype=np.uint8)
for c, r in zip(centers, radii):
    cv2.circle(mask, c, r, 255, -1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_masked = cv2.bitwise_and(gray, mask)

pixels = gray_masked[mask == 255]
print("Min  :", pixels.min())
print("Max  :", pixels.max())
print("Avg  :", pixels.mean().round(1))

cv2.imwrite("gray_masked.png", gray_masked)
