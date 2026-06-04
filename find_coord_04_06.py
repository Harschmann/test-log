import cv2

IMAGE_PATH = "test2.png"

clicks = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        print(f"Center {len(clicks)}: ({x}, {y})")
        cv2.circle(display, (x, y), 8, (0, 0, 255), -1)
        cv2.imshow("Click to get coords", display)

img = cv2.imread(IMAGE_PATH)
display = img.copy()

cv2.namedWindow("Click to get coords", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Click to get coords", click_event)
cv2.imshow("Click to get coords", display)

print("Click on camera centers. Press any key when done.")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nAll centers:")
print(clicks)
