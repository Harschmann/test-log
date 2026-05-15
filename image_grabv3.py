import customtkinter as ctk
import cv2
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from pypylon import pylon
from PIL import Image, ImageTk


class VisionGui(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VisionPro 20MP Live")
        self.geometry("900x600")
        self.camera = None
        self.is_live = False
        self.active_cam = None
        self._photo = None

        # Save directory: Desktop/raw_images (created if missing)
        self.save_dir = Path.home() / "Desktop" / "raw_images"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # UI
        self.sidebar = ctk.CTkFrame(self, width=200)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        self.cam_type = ctk.CTkOptionMenu(self.sidebar, values=["Basler", "IMI Tech"])
        self.cam_type.pack(pady=10)

        ctk.CTkButton(self.sidebar, text="START LIVE", command=self.toggle_live).pack(pady=10)
        ctk.CTkButton(self.sidebar, text="CAPTURE RAW", command=self.save_raw).pack(pady=10)

        self.status = ctk.CTkLabel(self.sidebar, text="Idle", wraplength=180)
        self.status.pack(pady=10)

        self.view = ctk.CTkLabel(self, text="Camera Feed Off")
        self.view.pack(expand=True, fill="both", padx=10, pady=10)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _set_status(self, msg):
        print(msg)
        try:
            self.status.configure(text=msg)
        except Exception:
            pass

    def toggle_live(self):
        if not self.is_live:
            self.active_cam = self.cam_type.get()
            try:
                if self.active_cam == "Basler":
                    self.camera = pylon.InstantCamera(
                        pylon.TlFactory.GetInstance().CreateFirstDevice()
                    )
                    self.camera.Open()
                    # Always run at 12-bit for lossless capture
                    try:
                        self.camera.PixelFormat.SetValue("BayerRG12")
                    except Exception as e:
                        print(f"PixelFormat warning: {e}")
                    try:
                        self.camera.Gain.SetValue(0)
                    except Exception as e:
                        print(f"Gain warning: {e}")
                    try:
                        self.camera.Gamma.SetValue(1.0)
                    except Exception as e:
                        print(f"Gamma warning: {e}")
                    self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                else:
                    self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    if not self.camera.isOpened():
                        raise RuntimeError("Could not open IMI Tech camera")
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                self.is_live = True
                self._set_status(f"Live: {self.active_cam}")
                self.update_view()
            except Exception as e:
                self._set_status(f"Start failed: {e}")
                self._release_camera()
        else:
            self.is_live = False
            self._release_camera()
            self.view.configure(image="", text="Camera Feed Off")
            self._photo = None
            self._set_status("Stopped")

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
                            arr = res.Array
                            # Downconvert to 8-bit only for the preview
                            if arr.dtype == np.uint16:
                                preview_8 = (arr >> 4).astype(np.uint8)
                            else:
                                preview_8 = arr
                            frame = cv2.cvtColor(preview_8, cv2.COLOR_BayerRG2BGR)
                    finally:
                        res.Release()
            else:
                ret, f = self.camera.read()
                if ret:
                    frame = f
        except Exception as e:
            print(f"Grab error: {e}")

        if frame is not None:
            try:
                preview = cv2.resize(frame, (800, 450))
                img = Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
                self._photo = ImageTk.PhotoImage(img)
                self.view.configure(image=self._photo, text="")
            except Exception as e:
                print(f"Display error: {e}")

        self.after(10, self.update_view)

    def save_raw(self):
        if self.camera is None or not self.is_live:
            self._set_status("Start live first.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"raw_{timestamp}.tiff"
        path = str(self.save_dir / filename)

        try:
            if self.active_cam == "Basler":
                res = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_Return)
                if res is None:
                    self._set_status("Grab timeout.")
                    return
                try:
                    if not res.GrabSucceeded():
                        self._set_status("Grab failed.")
                        return
                    arr = res.Array  # uint16, 12 bits of real data — lossless
                    ok = cv2.imwrite(path, arr, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
                finally:
                    res.Release()
            else:
                ret, frame = self.camera.read()
                if not ret:
                    self._set_status("Frame read failed.")
                    return
                ok = cv2.imwrite(path, frame, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

            if ok:
                self._set_status(f"Saved: {filename}")
            else:
                self._set_status("imwrite returned False")
        except Exception as e:
            self._set_status(f"Capture failed: {e}")

    def _on_close(self):
        self.is_live = False
        self._release_camera()
        self.destroy()


if __name__ == "__main__":
    VisionGui().mainloop()
