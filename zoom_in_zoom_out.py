import cv2

def add_zoom_support(window_name, image):
    # Zoom level store karne ke liye ek dictionary
    state = {'zoom_factor': 1.0}
    
    # Pehli baar original image show karein
    cv2.imshow(window_name, image)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            # Trackpad Pinch Out / Scroll Up -> Zoom In
            if flags > 0:
                state['zoom_factor'] += 0.1
            # Trackpad Pinch In / Scroll Down -> Zoom Out
            else:
                state['zoom_factor'] -= 0.1
            
            # Zoom limits set karna (1.0 = normal, 5.0 = 5x zoom)
            state['zoom_factor'] = max(1.0, min(state['zoom_factor'], 5.0))
            
            # Original dimensions
            h, w = image.shape[:2]
            
            # Nayi cropped dimensions calculate karein
            new_w = int(w / state['zoom_factor'])
            new_h = int(h / state['zoom_factor'])
            
            # Center coordinates nikaalein taaki center par zoom ho
            start_x = (w - new_w) // 2
            start_y = (h - new_h) // 2
            
            # Image ko crop karein (Zoom effect)
            cropped_image = image[start_y:start_y + new_h, start_x:start_x + new_w]
            
            # Nayi zoomed image show karein
            cv2.imshow(window_name, cropped_image)

    # Window par mouse event attach karein
    cv2.setMouseCallback(window_name, mouse_callback)
