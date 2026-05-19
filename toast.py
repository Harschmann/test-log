"""Toast notification widget - non-blocking, auto-dismissing pop-up."""
import customtkinter as ctk


_COLOR_MAP = {
    "info":    ("#1f6aa5", "#ffffff"),
    "success": ("#2fa84f", "#ffffff"),
    "error":   ("#c0392b", "#ffffff"),
    "warning": ("#e67e22", "#ffffff"),
}


class Toast(ctk.CTkToplevel):
    """Non-blocking toast notification. Use Toast.show(master, message, kind)."""

    _active = []  # class-level list of currently-visible toasts

    @classmethod
    def show(cls, master, message: str, kind: str = "info",
             duration_ms: int = 2200) -> "Toast":
        """Show a toast. kind: 'info' | 'success' | 'error' | 'warning'."""
        return cls(master, message, kind, duration_ms)

    def __init__(self, master, message: str, kind: str, duration_ms: int):
        super().__init__(master)
        self.overrideredirect(True)              # remove window decorations
        self.attributes("-topmost", True)
        try:
            self.attributes("-alpha", 0.0)        # start invisible for fade-in
        except Exception:
            pass

        bg, fg = _COLOR_MAP.get(kind, _COLOR_MAP["info"])

        frame = ctk.CTkFrame(self, fg_color=bg, corner_radius=10)
        frame.pack(padx=4, pady=4)
        label = ctk.CTkLabel(
            frame, text=message, text_color=fg,
            font=ctk.CTkFont(size=14, weight="bold"),
            padx=22, pady=14, justify="left",
        )
        label.pack()

        # Position bottom-right of master, stacking with existing toasts
        self.update_idletasks()
        try:
            mx = master.winfo_rootx()
            my = master.winfo_rooty()
            mw = master.winfo_width()
            mh = master.winfo_height()
            w = self.winfo_width()
            h = self.winfo_height()
            offset = 0
            for t in Toast._active:
                try:
                    if t.winfo_exists():
                        offset += t.winfo_height() + 10
                except Exception:
                    pass
            x = mx + mw - w - 30
            y = my + mh - h - 50 - offset
            self.geometry(f"+{x}+{y}")
        except Exception:
            pass

        Toast._active.append(self)
        self._fade_in(0.0)
        self.after(duration_ms, self._fade_out)

    def _fade_in(self, alpha: float):
        if not self.winfo_exists():
            return
        if alpha >= 1.0:
            try:
                self.attributes("-alpha", 1.0)
            except Exception:
                pass
            return
        try:
            self.attributes("-alpha", alpha)
        except Exception:
            return
        self.after(20, lambda: self._fade_in(alpha + 0.1))

    def _fade_out(self, alpha: float = 1.0):
        if not self.winfo_exists():
            return
        if alpha <= 0:
            try:
                Toast._active.remove(self)
            except ValueError:
                pass
            try:
                self.destroy()
            except Exception:
                pass
            return
        try:
            self.attributes("-alpha", alpha)
        except Exception:
            try:
                self.destroy()
            except Exception:
                pass
            return
        self.after(30, lambda: self._fade_out(alpha - 0.1))
