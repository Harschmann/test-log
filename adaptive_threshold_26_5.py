import cv2
import numpy as np

img = cv2.imread("test.jpg")

centers = [(2409, 1046), (1010, 1012), (3807, 1063)]
radii   = [325, 300, 250]

mask = np.zeros(img.shape[:2], dtype=np.uint8)
for c, r in zip(centers, radii):
    cv2.circle(mask, c, r, 255, -1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

binary = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    51,    # block size - neighborhood kitna bada ho
    -5     # C - neighbor average se kitna differ karna chahiye
)

binary_masked = cv2.bitwise_and(binary, mask)
cv2.imwrite("adaptive_binary.png", binary_masked)

cv2.namedWindow("Adaptive", cv2.WINDOW_NORMAL)
cv2.imshow("Adaptive", binary_masked)
cv2.waitKey(0)
cv2.destroyAllWindows()
