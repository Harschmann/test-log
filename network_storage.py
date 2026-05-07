# network_storage.py
import os
import shutil
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SaveResult:
    success: bool
    message: str
    saved_path: Optional[str] = None
    error_type: Optional[str] = None
    fallback_used: bool = False


def ensure_remote_dir(remote_dir: str) -> SaveResult:
    """
    Try to create the remote directory if it doesn't exist.
    Returns SaveResult — check .success before uploading.
    """
    try:
        os.makedirs(remote_dir, exist_ok=True)
        return SaveResult(True, f"Directory ready: {remote_dir}", saved_path=remote_dir)

    except PermissionError:
        return SaveResult(
            False,
            f"PERMISSION DENIED\n"
            f"Cannot write to: {remote_dir}\n\n"
            f"Fix on production PC:\n"
            f"  • Right-click shared folder → Properties → Sharing\n"
            f"  • Set Everyone permission to: Read/Write",
            error_type="PermissionError"
        )
    except FileNotFoundError:
        return SaveResult(
            False,
            f"NETWORK PATH NOT FOUND\n"
            f"Path: {remote_dir}\n\n"
            f"Check:\n"
            f"  • Is the production PC powered on?\n"
            f"  • Is the IP correct?\n"
            f"  • Is the folder shared with this exact share name?\n"
            f"  • Try in Explorer: open  \\\\<production-ip>\\<share-name>",
            error_type="FileNotFoundError"
        )
    except OSError as exc:
        code = exc.winerror if hasattr(exc, 'winerror') else exc.errno

        # Common Windows network error codes
        error_map = {
            53:   (
                "NETWORK PATH NOT FOUND (Error 53)\n"
                "Production PC is unreachable.\n\n"
                "  • Ping it:  ping " + remote_dir.split("\\")[2] + "\n"
                "  • Is it on the same network?\n"
                "  • Is Windows Firewall blocking File Sharing?"
            ),
            67:   (
                "NETWORK NAME NOT FOUND (Error 67)\n"
                "The share name doesn't exist on the production PC.\n\n"
                "  • Check the share name in: Network & Sharing Center\n"
                "  • Share names are case-sensitive on some systems"
            ),
            5:    (
                "ACCESS DENIED (Error 5)\n"
                "You don't have permission to write to this share.\n\n"
                "On the production PC:\n"
                "  • Open the shared folder properties → Sharing\n"
                "  • Add 'Everyone' with Read/Write access"
            ),
            1326: (
                "LOGON FAILURE (Error 1326)\n"
                "The production PC rejected the login.\n\n"
                "  • Both PCs must be on the same workgroup\n"
                "  • Or map the drive manually with credentials:\n"
                "    net use Z: \\\\<ip>\\<share> /user:<username> <password>"
            ),
            64:   (
                "NETWORK NAME NO LONGER AVAILABLE (Error 64)\n"
                "Connection dropped mid-transfer.\n\n"
                "  • Check network cable / WiFi stability\n"
                "  • Check if the production PC went to sleep"
            ),
            121:  (
                "NETWORK TIMEOUT (Error 121)\n"
                "Production PC took too long to respond.\n\n"
                "  • Is the production PC under heavy load?\n"
                "  • Try again in a moment"
            ),
        }

        if code in error_map:
            return SaveResult(False, error_map[code], error_type=f"OSError_{code}")

        return SaveResult(
            False,
            f"NETWORK ERROR (Code {code})\n{exc}\n\n"
            f"Path attempted: {remote_dir}",
            error_type=f"OSError_{code}"
        )


def save_image_to_network(
    local_path: str,
    remote_dir: str,
    filename: str,
    fallback_dir: Optional[str] = None,
) -> SaveResult:
    """
    Copy a locally-saved image to the network share.

    Args:
        local_path:   full path to the image on THIS machine
        remote_dir:   UNC path e.g. \\\\192.168.1.100\\QRCaptures\\scans
        filename:     just the filename, e.g. 20260507_143201_ABC123.jpg
        fallback_dir: if set, saves locally here when network fails

    Returns:
        SaveResult with .success, .message, .saved_path, .fallback_used
    """

    # ── Pre-flight ─────────────────────────────────────────────────────────────
    if not os.path.exists(local_path):
        return SaveResult(
            False,
            f"LOCAL FILE MISSING\nExpected at: {local_path}",
            error_type="LocalFileMissing"
        )

    local_size = os.path.getsize(local_path)
    if local_size == 0:
        return SaveResult(
            False,
            f"LOCAL FILE IS EMPTY (0 bytes)\nPath: {local_path}",
            error_type="EmptyFile"
        )

    # ── Ensure remote dir exists ───────────────────────────────────────────────
    dir_result = ensure_remote_dir(remote_dir)
    if not dir_result.success:
        return _handle_fallback(dir_result, local_path, filename, fallback_dir)

    # ── Copy file ──────────────────────────────────────────────────────────────
    remote_path = os.path.join(remote_dir, filename)

    try:
        shutil.copy2(local_path, remote_path)
    except PermissionError:
        result = SaveResult(
            False,
            f"PERMISSION DENIED writing file\n"
            f"Destination: {remote_path}\n\n"
            f"  • Check share permissions on production PC\n"
            f"  • Make sure the folder isn't read-only",
            error_type="PermissionError"
        )
        return _handle_fallback(result, local_path, filename, fallback_dir)
    except OSError as exc:
        code = exc.winerror if hasattr(exc, 'winerror') else exc.errno
        if code == 39 or (hasattr(exc, 'winerror') and exc.winerror == 112):
            result = SaveResult(
                False,
                f"DISK FULL ON PRODUCTION PC\n"
                f"No space left at: {remote_dir}\n\n"
                f"  • Free up space on the production machine",
                error_type="DiskFull"
            )
        else:
            result = SaveResult(
                False,
                f"COPY FAILED (Code {code})\n{exc}\n\n"
                f"Destination: {remote_path}",
                error_type=f"OSError_{code}"
            )
        return _handle_fallback(result, local_path, filename, fallback_dir)
    except Exception as exc:
        result = SaveResult(
            False,
            f"UNEXPECTED ERROR DURING COPY\n{type(exc).__name__}: {exc}",
            error_type="Unknown"
        )
        return _handle_fallback(result, local_path, filename, fallback_dir)

    # ── Verify file size on remote ─────────────────────────────────────────────
    try:
        remote_size = os.path.getsize(remote_path)
        if remote_size != local_size:
            os.remove(remote_path)  # remove corrupted copy
            result = SaveResult(
                False,
                f"INTEGRITY CHECK FAILED\n"
                f"Local: {local_size} bytes  |  Remote: {remote_size} bytes\n"
                f"Corrupted copy removed from: {remote_path}\n\n"
                f"  • Check network stability and retry",
                error_type="IntegrityError"
            )
            return _handle_fallback(result, local_path, filename, fallback_dir)
    except Exception as exc:
        logger.warning(f"[Network] Could not verify remote file size: {exc}")

    logger.info(f"[Network] Saved → {remote_path}")
    return SaveResult(
        True,
        f"Saved to production server:\n{remote_path}",
        saved_path=remote_path,
    )


def _handle_fallback(
    original_result: SaveResult,
    local_path: str,
    filename: str,
    fallback_dir: Optional[str],
) -> SaveResult:
    """If network save failed and fallback_dir is set, copy there instead."""
    if not fallback_dir:
        return original_result

    try:
        os.makedirs(fallback_dir, exist_ok=True)
        fallback_path = os.path.join(fallback_dir, filename)
        shutil.copy2(local_path, fallback_path)
        return SaveResult(
            False,
            f"{original_result.message}\n\n"
            f"⚠ FALLBACK: Image saved locally instead:\n{fallback_path}",
            saved_path=fallback_path,
            error_type=original_result.error_type,
            fallback_used=True,
        )
    except Exception as exc:
        return SaveResult(
            False,
            f"{original_result.message}\n\n"
            f"⚠ FALLBACK ALSO FAILED: {exc}",
            error_type=original_result.error_type,
        )
