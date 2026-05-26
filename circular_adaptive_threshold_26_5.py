import cv2
import numpy as np
import math

img = cv2.imread("img3.png")

centers = [(2409, 1046), (1010, 1012), (3807, 1063)]
radii   = [325, 300, 250]

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

result = img.copy()
dust_count = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 5 or area > 500:
        continue

    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue

    circularity = (4 * math.pi * area) / (perimeter ** 2)

    # debug - dekho kya aa raha hai
    print(f"area={area:.0f}  circularity={circularity:.2f}")

    if circularity < 0.5:   # arc/ring shape hai, skip
        continue

    (x, y), r = cv2.minEnclosingCircle(cnt)
    cv2.circle(result, (int(x), int(y)), max(10, int(r)+4), (0, 0, 255), 3)
    dust_count += 1

print("Dust found:", dust_count)

cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
