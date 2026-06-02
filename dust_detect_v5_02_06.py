import cv2
import numpy as np

img = cv2.imread("test2.png")

centers = [(821, 1630), (2207, 1630), (3613, 1630)]
radii   = [300, 300, 300]

mask = np.zeros(img.shape[:2], dtype=np.uint8)
for c, r in zip(centers, radii):
    cv2.circle(mask, c, r, 255, -1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# camera area ke andar ki brightness stats
pixels = gray[mask == 255]
print("min :", pixels.min())
print("max :", pixels.max())
print("mean:", round(float(pixels.mean()), 1))
print("90th percentile:", round(float(np.percentile(pixels, 90)), 1))
print("95th percentile:", round(float(np.percentile(pixels, 95)), 1))
print("99th percentile:", round(float(np.percentile(pixels, 99)), 1))
