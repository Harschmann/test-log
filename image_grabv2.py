import customtkinter as ctk
import cv2
from pypylon import pylon
from PIL import Image, ImageTk

class VisionGui(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VisionPro 20MP Live"); self.geometry("900x600")
        self.camera = None
        self.is_live = False
        self.active_cam = None  # locked at start-live time
        self._photo = None      # keep ref to avoid GC

        self.sidebar = ctk.CTkFrame(self, width=200)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)
        self.cam_type = ctk.CTkOptionMenu(self.sidebar, values=["Basler", "IMI Tech"])
        self.cam_type.pack(pady=10)
        ctk.CTkButton(self.sidebar, text="START LIVE", command=self.toggle_live).pack(pady=10)
        ctk.CTkButton(self.sidebar, text="CAPTURE RAW", command=self.save_raw).pack(pady=10)

        self.view = ctk.CTkLabel(self, text="Camera Feed Off")
        self.view.pack(expand=True, fill="both", padx=10, pady=10)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def toggle_live(self):
        if not self.is_live:
            self.active_cam = self.cam_type.get()
            try:
                if self.active_cam == "Basler":
                    self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
                    self.camera.Open()
                    self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                else:
                    self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    if not self.camera.isOpened():
                        raise RuntimeError("Could not open IMI Tech camera")
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.is_live = True
                self.update_view()
            except Exception as e:
                print(f"Start failed: {e}")
                self._release_camera()
        else:
            self.is_live = False
            self._release_camera()
            self.view.configure(image="", text="Camera Feed Off")
            self._photo = None

    def _release_camera(self):
        if self.camera is None:
            return
        try:
            if self.active_cam == "Basler":
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                if self.camera.IsOpen():
                    self.camera.Close()
            else:
                self.camera.release()
        except Exception as e:
            print(f"Release error: {e}")
        finally:
            self.camera = None

    def update_view(self):
        if not self.is_live or self.camera is None:
            return
        frame = None
        try:
            if self.active_cam == "Basler":
                res = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
                if res is not None:
                    try:
                        if res.GrabSucceeded():
                            frame = cv2.cvtColor(res.Array, cv2.COLOR_BayerRG2BGR)
                    finally:
                        res.Release()
            else:
                ret, frame = self.camera.read()
                if not ret:
                    frame = None
        except Exception as e:
            print(f"Grab error: {e}")

        if frame is not None:
            preview = cv2.resize(frame, (800, 450))
            img = Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
            self._photo = ImageTk.PhotoImage(img)   # single instance, kept alive
            self.view.configure(image=self._photo, text="")

        self.after(10, self.update_view)

    def save_raw(self):
        if self.camera is None:
            print("Camera not started.")
            return
        path = "raw_20mp_capture.tiff"
        try:
            if self.active_cam == "Basler":
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                # Correct pypylon API: use .SetValue on the nodes
                self.camera.PixelFormat.SetValue("BayerRG12")
                self.camera.Gain.SetValue(0)
                self.camera.Gamma.SetValue(1.0)
                res = self.camera.GrabOne(5000)
                try:
                    if res.GrabSucceeded():
                        cv2.imwrite(path, res.Array, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
                finally:
                    res.Release()
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            else:
                ret, frame = self.camera.read()
                if ret:
                    cv2.imwrite(path, frame, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            print("RAW saved:", path)
        except Exception as e:
            print(f"Capture failed: {e}")

    def _on_close(self):
        self.is_live = False
        self._release_camera()
        self.destroy()

if __name__ == "__main__":
    VisionGui().mainloop()
