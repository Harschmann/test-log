import cv2
import numpy as np

img = cv2.imread("test2.png")

centers = [(821, 1630), (2207, 1630), (3613, 1630)]
radii   = [300, 300, 300]

mask = np.zeros(img.shape[:2], dtype=np.uint8)
for c, r in zip(centers, radii):
    cv2.circle(mask, c, r, 255, -1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# brightness boost
brightened = cv2.convertScaleAbs(blurred, alpha=2.0, beta=0)

_, binary = cv2.threshold(brightened, 200, 255, cv2.THRESH_BINARY)

# dilation - chote particles bade ho jayenge
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(binary, kernel, iterations=2)

binary_masked = cv2.bitwise_and(dilated, mask)

contours, _ = cv2.findContours(binary_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

result = img.copy()
dust_count = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    if 5 <= area <= 500:
        (x, y), r = cv2.minEnclosingCircle(cnt)
        cv2.circle(result, (int(x), int(y)), max(10, int(r) + 4), (0, 0, 255), 3)
        dust_count += 1

print("Dust found:", dust_count)

cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
