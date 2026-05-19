"""Main application window."""
import time
from pathlib import Path
from typing import Optional

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from config import (
    APP_NAME, APP_VERSION,
    WINDOW_WIDTH, WINDOW_HEIGHT, PREVIEW_WIDTH, PREVIEW_HEIGHT,
    APPEARANCE_MODE, COLOR_THEME,
    DEFAULT_MODE, BURST_COUNT, BURST_INTERVAL_MS, LIVE_INTERVAL_MS,
)
from cameras import CameraBase, CameraFactory
from core import CaptureManager
from ui.toast import Toast
from ui.viewer_window import ViewerWindow


class MainWindow(ctk.CTk):
    """Main VisionPro window."""

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode(APPEARANCE_MODE)
        ctk.set_default_color_theme(COLOR_THEME)

        self.title(f"{APP_NAME} v{APP_VERSION}")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.minsize(1200, 800)

        # ---------- State ----------
        self.camera: Optional[CameraBase] = None
        self.is_live: bool = False
        self.mode: str = DEFAULT_MODE                  # "raw" or "normal"
        self.show_crosshair: bool = False
        self.show_histogram: bool = False
        self._photo = None                              # ImageTk ref (don't GC)
        self._last_frame_bgr: Optional[np.ndarray] = None
        self._fps_samples = []
        self._fps_last_time = time.time()
        self._after_id: Optional[str] = None
        self._viewer_window: Optional[ViewerWindow] = None
        self._burst_remaining = 0

        self.capture = CaptureManager()

        # ---------- UI ----------
        self._build_ui()
        self._bind_shortcuts()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Auto-detect on startup (after first paint so UI is responsive)
        self.after(400, self._auto_connect)

    # ============================================================
    # UI construction
    # ============================================================

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_left_sidebar()
        self._build_preview_area()
        self._build_right_sidebar()
        self._build_status_bar()

        self._update_count_label()

    def _build_header(self):
        header = ctk.CTkFrame(self, height=60, corner_radius=0)
        header.grid(row=0, column=0, columnspan=3, sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(
            header, text=f"  {APP_NAME}",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title.grid(row=0, column=0, sticky="w", padx=15, pady=12)

        self.camera_lbl = ctk.CTkLabel(
            header, text="No camera connected",
            font=ctk.CTkFont(size=13), text_color="gray70",
        )
        self.camera_lbl.grid(row=0, column=1, sticky="w", padx=10)

        self.connection_dot = ctk.CTkLabel(
            header, text="●", text_color="#c0392b",
            font=ctk.CTkFont(size=22),
        )
        self.connection_dot.grid(row=0, column=2, sticky="e", padx=(0, 4))

        self.connection_state_lbl = ctk.CTkLabel(
            header, text="DISCONNECTED",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.connection_state_lbl.grid(row=0, column=3, sticky="e", padx=(0, 18))

    def _build_left_sidebar(self):
        self.left = ctk.CTkFrame(self, width=230, corner_radius=0)
        self.left.grid(row=1, column=0, sticky="nsew")
        self.left.grid_propagate(False)

        pad = {"padx": 14, "pady": 6}

        self._section_label(self.left, "CAMERA")
        self.btn_connect = ctk.CTkButton(
            self.left, text="🔍  Detect & Connect", height=40,
            command=self._auto_connect,
        )
        self.btn_connect.pack(fill="x", **pad)

        self.btn_disconnect = ctk.CTkButton(
            self.left, text="✕  Disconnect", height=36,
            fg_color="#6e6e6e", hover_color="#525252",
            command=self._disconnect,
        )
        self.btn_disconnect.pack(fill="x", **pad)

        self._section_label(self.left, "LIVE")
        self.btn_live = ctk.CTkButton(
            self.left, text="▶  START LIVE", height=46,
            fg_color="#27ae60", hover_color="#1e8449",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._toggle_live,
        )
        self.btn_live.pack(fill="x", **pad)

        self._section_label(self.left, "CAPTURE")
        self.btn_capture = ctk.CTkButton(
            self.left, text="📸  Take Screenshot", height=46,
            fg_color="#2980b9", hover_color="#21618c",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._capture_one,
        )
        self.btn_capture.pack(fill="x", **pad)

        self.btn_burst = ctk.CTkButton(
            self.left, text=f"🔁  Burst Capture ({BURST_COUNT}x)", height=36,
            command=self._capture_burst,
        )
        self.btn_burst.pack(fill="x", **pad)

        self._section_label(self.left, "VIEW")
        self.btn_viewer = ctk.CTkButton(
            self.left, text="🖼  Image Viewer", height=40,
            fg_color="#8e44ad", hover_color="#6c3483",
            command=self._open_viewer,
        )
        self.btn_viewer.pack(fill="x", **pad)

        ctk.CTkLabel(self.left, text="").pack(expand=True)  # spacer

        self.btn_exit = ctk.CTkButton(
            self.left, text="Exit", height=34,
            fg_color="#7b1f1f", hover_color="#5e1717",
            command=self._on_close,
        )
        self.btn_exit.pack(fill="x", **pad)

    def _build_preview_area(self):
        self.center = ctk.CTkFrame(self, fg_color=("gray85", "gray10"))
        self.center.grid(row=1, column=1, sticky="nsew", padx=2, pady=2)
        self.center.grid_rowconfigure(0, weight=1)
        self.center.grid_columnconfigure(0, weight=1)

        self.preview = ctk.CTkLabel(
            self.center,
            text="📷   Live preview will appear here\n\nClick  ▶  START LIVE  to begin",
            font=ctk.CTkFont(size=18), text_color="gray60",
        )
        self.preview.grid(row=0, column=0, sticky="nsew")

    def _build_right_sidebar(self):
        self.right = ctk.CTkFrame(self, width=270, corner_radius=0)
        self.right.grid(row=1, column=2, sticky="nsew")
        self.right.grid_propagate(False)

        pad = {"padx": 14, "pady": 6}

        self._section_label(self.right, "MODE")
        self.mode_switch = ctk.CTkSwitch(
            self.right, text="Raw Mode (12-bit TIFF)",
            command=self._toggle_mode,
        )
        if self.mode == "raw":
            self.mode_switch.select()
        self.mode_switch.pack(fill="x", **pad)

        ctk.CTkLabel(
            self.right,
            text="• Normal:  8-bit JPEG, debayered\n• Raw:  12-bit TIFF, sensor data",
            font=ctk.CTkFont(size=10), text_color="gray60",
            justify="left", anchor="w",
        ).pack(fill="x", padx=14, pady=(0, 6))

        self._section_label(self.right, "EXPOSURE (µs)")
        self.exposure_value_lbl = ctk.CTkLabel(
            self.right, text="—", font=ctk.CTkFont(size=12),
        )
        self.exposure_value_lbl.pack(anchor="w", padx=14)

        self.exposure_slider = ctk.CTkSlider(
            self.right, from_=1000, to=100000, number_of_steps=99,
            command=self._on_exposure,
        )
        self.exposure_slider.set(20000)
        self.exposure_slider.pack(fill="x", **pad)
        self.exposure_slider.configure(state="disabled")

        self._section_label(self.right, "OVERLAYS")
        self.crosshair_switch = ctk.CTkSwitch(
            self.right, text="Crosshair", command=self._toggle_crosshair,
        )
        self.crosshair_switch.pack(fill="x", **pad)

        self.hist_switch = ctk.CTkSwitch(
            self.right, text="Histogram", command=self._toggle_histogram,
        )
        self.hist_switch.pack(fill="x", **pad)

        self._section_label(self.right, "CAPTURES")
        self.capture_counters = ctk.CTkLabel(
            self.right, text="Normal: 0\nRaw: 0",
            font=ctk.CTkFont(size=13), justify="left", anchor="w",
        )
        self.capture_counters.pack(fill="x", padx=14, pady=(0, 6))

        ctk.CTkLabel(self.right, text="").pack(expand=True)  # spacer
        ctk.CTkLabel(
            self.right,
            text=("KEYBOARD SHORTCUTS\n"
                  "Space — Capture\n"
                  "L — Toggle Live\n"
                  "M — Toggle Mode\n"
                  "V — Image Viewer\n"
                  "C — Crosshair\n"
                  "Esc — Exit"),
            font=ctk.CTkFont(size=10), text_color="gray60",
            justify="left", anchor="w",
        ).pack(fill="x", padx=14, pady=(0, 14))

    def _build_status_bar(self):
        bar = ctk.CTkFrame(self, height=32, corner_radius=0)
        bar.grid(row=2, column=0, columnspan=3, sticky="ew")
        bar.grid_columnconfigure(4, weight=1)

        self.status_lbl = ctk.CTkLabel(bar, text="Ready", font=ctk.CTkFont(size=11))
        self.status_lbl.grid(row=0, column=0, sticky="w", padx=10, pady=4)

        self.fps_lbl = ctk.CTkLabel(bar, text="FPS: --",
                                    font=ctk.CTkFont(size=11), width=80)
        self.fps_lbl.grid(row=0, column=1, padx=10)

        self.res_lbl = ctk.CTkLabel(bar, text="--",
                                    font=ctk.CTkFont(size=11), width=120)
        self.res_lbl.grid(row=0, column=2, padx=10)

        self.mode_lbl = ctk.CTkLabel(
            bar, text=f"Mode: {self.mode.upper()}",
            font=ctk.CTkFont(size=11, weight="bold"), width=130,
        )
        self.mode_lbl.grid(row=0, column=3, padx=10)

        self.count_lbl = ctk.CTkLabel(bar, text="", font=ctk.CTkFont(size=11))
        self.count_lbl.grid(row=0, column=4, sticky="e", padx=10)

    def _section_label(self, parent, text: str):
        ctk.CTkLabel(
            parent, text=text,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="gray60",
        ).pack(anchor="w", padx=14, pady=(14, 4))

    def _bind_shortcuts(self):
        self.bind("<space>", lambda e: self._capture_one())
        self.bind("<l>", lambda e: self._toggle_live())
        self.bind("<L>", lambda e: self._toggle_live())
        self.bind("<m>", lambda e: self._toggle_mode_via_keyboard())
        self.bind("<M>", lambda e: self._toggle_mode_via_keyboard())
        self.bind("<v>", lambda e: self._open_viewer())
        self.bind("<V>", lambda e: self._open_viewer())
        self.bind("<c>", lambda e: self._toggle_crosshair_via_keyboard())
        self.bind("<C>", lambda e: self._toggle_crosshair_via_keyboard())
        self.bind("<Escape>", lambda e: self._on_close())

    # ============================================================
    # Camera lifecycle
    # ============================================================

    def _auto_connect(self):
        if self.camera is not None and self.camera.is_connected:
            Toast.show(self, f"Already connected:\n{self.camera.name}", "info")
            return

        self._set_status("Detecting cameras...")
        self.update_idletasks()

        available = CameraFactory.list_available()
        if not available:
            self._set_status("No cameras detected.")
            Toast.show(self, "No cameras detected", "error")
            return

        self._set_status(f"Found: {', '.join(available)} — connecting...")
        self.update_idletasks()

        cam = CameraFactory.auto_detect_and_create()
        if cam is None:
            self._set_status("Failed to connect to any camera.")
            Toast.show(self, "Connection failed", "error")
            return

        self.camera = cam
        self._on_connected(cam.get_info())
        Toast.show(self, f"Connected ✓\n{cam.name}", "success")

    def _on_connected(self, info: dict):
        self.camera_lbl.configure(text=info.get("name", "Unknown"))
        self.connection_dot.configure(text_color="#27ae60")
        self.connection_state_lbl.configure(text="CONNECTED")

        w = info.get("width")
        h = info.get("height")
        if w and h:
            self.res_lbl.configure(text=f"{w} x {h}")
        else:
            self.res_lbl.configure(text="--")

        # Configure exposure slider if supported
        if self.camera is not None and self.camera.supports_exposure():
            try:
                mn, mx = self.camera.get_exposure_range()
                mx = min(mx, 200_000.0)
                steps = max(10, int((mx - mn) / 1000))
                self.exposure_slider.configure(
                    from_=mn, to=mx, state="normal", number_of_steps=steps,
                )
                initial = max(mn, min(mx, 20_000.0))
                self.exposure_slider.set(initial)
                self.camera.set_exposure(initial)
                self.exposure_value_lbl.configure(text=f"{int(initial):,} µs")
            except Exception as e:
                print(f"Exposure init: {e}")
                self.exposure_slider.configure(state="disabled")
                self.exposure_value_lbl.configure(text="(not supported)")
        else:
            self.exposure_slider.configure(state="disabled")
            self.exposure_value_lbl.configure(text="(not supported)")

        self._set_status(f"Connected to {info.get('name', 'camera')}.")

    def _disconnect(self):
        if self.is_live:
            self._stop_live()
        if self.camera is not None:
            try:
                self.camera.disconnect()
            except Exception as e:
                print(f"Disconnect error: {e}")
            self.camera = None

        self.camera_lbl.configure(text="No camera connected")
        self.connection_dot.configure(text_color="#c0392b")
        self.connection_state_lbl.configure(text="DISCONNECTED")
        self.res_lbl.configure(text="--")

        self._photo = None
        try:
            self.preview.configure(
                image="",
                text="📷   Live preview will appear here\n\nClick  ▶  START LIVE  to begin",
            )
        except Exception:
            pass

        self.exposure_slider.configure(state="disabled")
        self.exposure_value_lbl.configure(text="—")

        self._set_status("Disconnected.")
        Toast.show(self, "Disconnected", "info")

    # ============================================================
    # Live preview
    # ============================================================

    def _toggle_live(self):
        if self.camera is None:
            Toast.show(self, "Connect a camera first", "warning")
            return
        if self.is_live:
            self._stop_live()
        else:
            self._start_live()

    def _start_live(self):
        self.is_live = True
        self.btn_live.configure(
            text="■  STOP LIVE",
            fg_color="#c0392b", hover_color="#922b21",
        )
        self._set_status("Live streaming...")
        self._fps_samples.clear()
        self._fps_last_time = time.time()
        self._update_preview()

    def _stop_live(self):
        self.is_live = False
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        self.btn_live.configure(
            text="▶  START LIVE",
            fg_color="#27ae60", hover_color="#1e8449",
        )
        self._set_status("Live stopped.")
        self.fps_lbl.configure(text="FPS: --")

    def _update_preview(self):
        if not self.is_live or self.camera is None:
            return

        frame = self.camera.grab_frame()
        if frame is not None:
            # Keep a clean copy (no overlays) for capture in normal mode
            self._last_frame_bgr = frame.copy()

            # FPS calculation
            now = time.time()
            dt = now - self._fps_last_time
            self._fps_last_time = now
            if dt > 0:
                self._fps_samples.append(1.0 / dt)
                if len(self._fps_samples) > 30:
                    self._fps_samples.pop(0)
                fps = sum(self._fps_samples) / len(self._fps_samples)
                self.fps_lbl.configure(text=f"FPS: {fps:.1f}")

            # Overlays + resize for display
            display = self._apply_overlays(frame.copy())
            display = self._fit_to_preview(display)
            try:
                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                self._photo = ImageTk.PhotoImage(img)
                self.preview.configure(image=self._photo, text="")
            except Exception as e:
                print(f"Display error: {e}")

        self._after_id = self.after(LIVE_INTERVAL_MS, self._update_preview)

    def _apply_overlays(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self.show_crosshair:
            cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
            cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 0), 1)
            r = min(h, w) // 8
            cv2.circle(frame, (w // 2, h // 2), r, (0, 255, 0), 1)
        if self.show_histogram:
            self._draw_histogram(frame)
        return frame

    def _draw_histogram(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        hist_w, hist_h = 256, 110
        x0, y0 = w - hist_w - 18, 18

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0 - 4, y0 - 4),
                      (x0 + hist_w + 4, y0 + hist_h + 4),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, dst=frame)

        for i, color in enumerate(((255, 80, 80), (80, 255, 80), (80, 80, 255))):
            hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, hist_h, cv2.NORM_MINMAX)
            for x in range(1, 256):
                cv2.line(
                    frame,
                    (x0 + x - 1, y0 + hist_h - int(hist[x - 1])),
                    (x0 + x,     y0 + hist_h - int(hist[x])),
                    color, 1,
                )

    def _fit_to_preview(self, frame: np.ndarray) -> np.ndarray:
        try:
            w = max(self.preview.winfo_width() - 8, 400)
            h = max(self.preview.winfo_height() - 8, 300)
        except Exception:
            w, h = PREVIEW_WIDTH, PREVIEW_HEIGHT
        fh, fw = frame.shape[:2]
        if fw <= 0 or fh <= 0:
            return frame
        scale = min(w / fw, h / fh)
        new_w, new_h = max(1, int(fw * scale)), max(1, int(fh * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        return cv2.resize(frame, (new_w, new_h), interpolation=interp)

    # ============================================================
    # Capture
    # ============================================================

    def _capture_one(self):
        if self.camera is None:
            Toast.show(self, "Connect a camera first", "warning")
            return
        if not self.is_live:
            Toast.show(self, "Start live preview first", "warning")
            return

        cam_name = self.camera.name

        if self.mode == "raw":
            arr = self.camera.grab_raw()
            if arr is None:
                Toast.show(self, "Raw grab failed", "error")
                return
            ok, info = self.capture.save_raw(arr, cam_name)
        else:
            frame = self._last_frame_bgr
            if frame is None:
                frame = self.camera.grab_frame()
            if frame is None:
                Toast.show(self, "Frame grab failed", "error")
                return
            ok, info = self.capture.save_normal(frame, cam_name)

        if ok:
            filename = Path(info).name
            kind_label = "RAW" if self.mode == "raw" else "Screenshot"
            Toast.show(self, f"✓  {kind_label} saved\n{filename}", "success")
            self._set_status(f"Saved: {info}")
        else:
            Toast.show(self, f"Save failed:\n{info}", "error")
            self._set_status(f"Save failed: {info}")

        self._update_count_label()

    def _capture_burst(self):
        if self.camera is None:
            Toast.show(self, "Connect a camera first", "warning")
            return
        if not self.is_live:
            Toast.show(self, "Start live preview first", "warning")
            return

        self._burst_remaining = BURST_COUNT
        Toast.show(self, f"Burst started: {BURST_COUNT} shots", "info")
        self._burst_step()

    def _burst_step(self):
        if self._burst_remaining <= 0:
            Toast.show(self, "✓  Burst complete", "success")
            self._update_count_label()
            return
        try:
            if self.mode == "raw":
                arr = self.camera.grab_raw() if self.camera else None
                if arr is not None:
                    self.capture.save_raw(arr, self.camera.name)
            else:
                frame = self.camera.grab_frame() if self.camera else None
                if frame is None:
                    frame = self._last_frame_bgr
                if frame is not None:
                    self.capture.save_normal(frame, self.camera.name)
        except Exception as e:
            print(f"Burst step error: {e}")

        self._burst_remaining -= 1
        self._update_count_label()
        self.after(BURST_INTERVAL_MS, self._burst_step)

    def _update_count_label(self):
        n = self.capture.normal_count
        r = self.capture.raw_count
        try:
            self.count_lbl.configure(text=f"Captures — Normal: {n}   Raw: {r}")
        except Exception:
            pass
        try:
            self.capture_counters.configure(text=f"Normal: {n}\nRaw: {r}")
        except Exception:
            pass

    # ============================================================
    # Settings handlers
    # ============================================================

    def _toggle_mode(self):
        self.mode = "raw" if self.mode_switch.get() else "normal"
        try:
            self.mode_lbl.configure(text=f"Mode: {self.mode.upper()}")
        except Exception:
            pass
        Toast.show(self, f"Mode: {self.mode.upper()}", "info", duration_ms=1200)

    def _toggle_mode_via_keyboard(self):
        if self.mode_switch.get():
            self.mode_switch.deselect()
        else:
            self.mode_switch.select()
        self._toggle_mode()

    def _toggle_crosshair(self):
        self.show_crosshair = bool(self.crosshair_switch.get())

    def _toggle_crosshair_via_keyboard(self):
        if self.crosshair_switch.get():
            self.crosshair_switch.deselect()
        else:
            self.crosshair_switch.select()
        self._toggle_crosshair()

    def _toggle_histogram(self):
        self.show_histogram = bool(self.hist_switch.get())

    def _on_exposure(self, value):
        if self.camera is not None and self.camera.supports_exposure():
            self.camera.set_exposure(float(value))
            try:
                self.exposure_value_lbl.configure(text=f"{int(value):,} µs")
            except Exception:
                pass

    def _open_viewer(self):
        if self._viewer_window is not None:
            try:
                if self._viewer_window.winfo_exists():
                    self._viewer_window.focus()
                    self._viewer_window.lift()
                    self._viewer_window._refresh_all()
                    return
            except Exception:
                pass
        self._viewer_window = ViewerWindow(self)

    # ============================================================
    # Misc
    # ============================================================

    def _set_status(self, msg: str):
        try:
            self.status_lbl.configure(text=msg)
        except Exception:
            pass
        print(f"[Status] {msg}")

    def _on_close(self):
        self.is_live = False
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
        if self.camera is not None:
            try:
                self.camera.disconnect()
            except Exception:
                pass
        try:
            self.destroy()
        except Exception:
            pass
