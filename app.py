# ── At the top of app.py, add these imports ──
from config import REMOTE_SAVE_DIR, LOCAL_FALLBACK_DIR
from network_storage import save_image_to_network

# ── Inside _on_capture(), replace the save block ──

# Save image locally first (temp), then push to network
import tempfile, shutil

ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
safe     = "".join(c if c.isalnum() or c in "-_" else "_" for c in qr_value[:30])
filename = f"{ts}_{safe}.jpg"

# Write to a temp local file first
tmp_path = os.path.join(tempfile.gettempdir(), filename)
cv2.imwrite(tmp_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

# Push to network share (with local fallback)
net_result = save_image_to_network(
    local_path=tmp_path,
    remote_dir=REMOTE_SAVE_DIR,
    filename=filename,
    fallback_dir=LOCAL_FALLBACK_DIR,
)

# Show result in status bar
if net_result.success:
    status_msg = f"✓ SAVED TO PRODUCTION  |  QR: {qr_value}  |  {filename}"
    self.statusBar().showMessage(status_msg)
elif net_result.fallback_used:
    status_msg = f"⚠ NETWORK FAIL — SAVED LOCALLY  |  QR: {qr_value}"
    self.statusBar().showMessage(status_msg)
    self._show_network_error(net_result.message)
else:
    self.statusBar().showMessage(f"✗ SAVE FAILED  |  QR: {qr_value}")
    self._show_network_error(net_result.message)

# Save to SQLite — use actual saved path
final_path = net_result.saved_path or tmp_path
blob = open(tmp_path, "rb").read()
save_scan(qr_value, final_path, blob)

# Clean up temp file
try:
    os.remove(tmp_path)
except Exception:
    pass
