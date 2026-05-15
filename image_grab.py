import customtkinter as ctk
import cv2
import numpy as np
from pypylon import pylon
from PIL import Image, ImageTk

class VisionGui(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VisionPro 20MP Live"); self.geometry("900x600")
        self.camera = None; self.is_live = False
        
        # UI Setup
        self.sidebar = ctk.CTkFrame(self, width=200); self.sidebar.pack(side="left", fill="y", padx=10, pady=10)
        self.cam_type = ctk.CTkOptionMenu(self.sidebar, values=["Basler", "IMI Tech"]); self.cam_type.pack(pady=10)
        ctk.CTkButton(self.sidebar, text="START LIVE", command=self.toggle_live).pack(pady=10)
        ctk.CTkButton(self.sidebar, text="CAPTURE RAW", command=self.save_raw).pack(pady=10)
        
        self.view = ctk.CTkLabel(self, text="Camera Feed Off"); self.view.pack(expand=True, fill="both", padx=10, pady=10)

    def toggle_live(self):
        if not self.is_live:
            if self.cam_type.get() == "Basler":
                self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
                self.camera.Open()
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            else:
                self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                self.camera.set(3, 1280); self.camera.set(4, 720) # Low res for fast preview
            self.is_live = True; self.update_view()
        else:
            self.is_live = False
            if hasattr(self.camera, 'Close'): self.camera.Close()
            else: self.camera.release()

    def update_view(self):
        if self.is_live:
            frame = None
            if self.cam_type.get() == "Basler":
                res = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if res.GrabSucceeded():
                    # Convert Bayer to BGR just for the GUI preview
                    frame = cv2.cvtColor(res.Array, cv2.COLOR_BayerRG2BGR)
                res.Release()
            else:
                ret, frame = self.camera.read()
            
            if frame is not None:
                # Resize for GUI and convert to RGB for Tkinter
                preview = cv2.resize(frame, (800, 450))
                img = Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
                self.view.configure(image=ImageTk.PhotoImage(img), text="")
                self.view.image = ImageTk.PhotoImage(img)
            self.after(10, self.update_view)

    def save_raw(self):
        # Stop live briefly to grab high-res raw
        path = "raw_20mp_capture.tiff"
        if self.cam_type.get() == "Basler":
            # For Basler, we use the existing open camera
            self.camera.StopGrabbing()
            self.camera.PixelFormat, self.camera.Gain, self.camera.Gamma = "BayerRG12", 0, 1.0
            res = self.camera.GrabOne(5000)
            if res.GrabSucceeded(): 
                cv2.imwrite(path, res.Array, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        else:
            # For IMI Tech, grab current frame at full settings
            self.camera.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            ret, frame = self.camera.read()
            if ret: cv2.imwrite(path, frame, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            self.camera.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        print("Zero-Loss RAW Saved.")

if __name__ == "__main__":
    VisionGui().mainloop()
