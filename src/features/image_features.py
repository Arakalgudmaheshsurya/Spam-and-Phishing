"""
src/features/image_features.py
------------------------------
Image analysis utilities for phishing detection:
- OCR text extraction
- QR / barcode URL decoding
- Perceptual hashes (pHash, dHash, aHash)
- Basic EXIF/IPTC/XMP metadata (select keys)
- MIME sanity via magic bytes
- Heuristics: entropy, appended data after EOF, oversized ICC profiles

Requires: pillow, pytesseract, pyzbar, imagehash, exifread, python-magic
"""

from typing import Dict, Any, List, Tuple, Optional
import io
import binascii
import math
import shutil
import pytesseract


from PIL import Image, ImageOps
import pytesseract
from pyzbar.pyzbar import decode as qr_decode
import imagehash
import exifread
import magic

# ---------- helpers ----------
# Attempt to auto-configure the Tesseract binary for pytesseract
_TESS_PATH_CANDIDATES = [
    "/opt/homebrew/bin/tesseract",   # common on Apple Silicon
    "/usr/local/bin/tesseract",      # common on Intel Macs
    shutil.which("tesseract"),       # whatever is on PATH
]

for _p in _TESS_PATH_CANDIDATES:
    if _p:
        try:
            pytesseract.pytesseract.tesseract_cmd = _p
            break
        except Exception:
            continue


def _bytes_to_image(b: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(b))
    if img.mode not in ("RGB", "L", "RGBA"):
        img = img.convert("RGB")
    return img

def _shannon_entropy(b: bytes, sample: int = 65536) -> float:
    """
    Estimate byte-level entropy on a prefix sample for speed.
    """
    data = b[:sample] if len(b) > sample else b
    if not data:
        return 0.0
    freq = [0]*256
    for x in data:
        freq[x] += 1
    ent = 0.0
    n = len(data)
    for c in freq:
        if c == 0: 
            continue
        p = c / n
        ent -= p * math.log2(p)
    return round(ent, 3)

def _icc_profile_size(img: Image.Image) -> int:
    try:
        icc = img.info.get("icc_profile")
        return len(icc) if icc else 0
    except Exception:
        return 0

def _detect_appended_data(b: bytes, mime_magic: str) -> bool:
    """
    Lightweight check for bytes after expected EOF markers.
    """
    try:
        if "jpeg" in mime_magic:
            # JPEG ends with 0xFFD9
            eoi = b"\xff\xd9"
            idx = b.rfind(eoi)
            return idx != -1 and idx < len(b) - 2
        if "png" in mime_magic:
            # PNG IEND chunk ends with 'IEND' and CRC; look for IEND
            iend = b"IEND"
            idx = b.rfind(iend)
            return idx != -1 and idx < len(b) - 8
        return False
    except Exception:
        return False

def _safe_exif(b: bytes) -> Dict[str, str]:
    """
    Extract a small, safe subset of EXIF tags.
    """
    try:
        tags = exifread.process_file(io.BytesIO(b), details=False, strict=True)
        keep_keys = ("Image Software", "Image Model", "Image Make", "EXIF DateTimeOriginal",
                     "GPS GPSLatitude", "GPS GPSLongitude")
        out = {}
        for k in keep_keys:
            if k in tags:
                v = str(tags[k])
                out[k] = v[:200]
        return out
    except Exception:
        return {}

# ---------- main API ----------

def process_image_bytes(
    b: bytes,
    do_ocr: bool = True,
    do_qr: bool = True,
    do_hash: bool = True,
) -> Dict[str, Any]:
    """
    Process a single image byte blob and return extracted features.
    """
    result: Dict[str, Any] = {
        "width": None,
        "height": None,
        "mime_magic": "",
        "ocr_text": "",
        "qr_links": [],
        "phash": "",
        "dhash": "",
        "ahash": "",
        "exif": {},
        "entropy": 0.0,
        "icc_profile_len": 0,
        "suspicious_flags": [],   # strings describing oddities
        "errors": [],
    }

    # MIME / magic bytes
    try:
        result["mime_magic"] = magic.from_buffer(b, mime=True) or ""
    except Exception as e:
        result["errors"].append(f"magic_error:{type(e).__name__}")

    # Entropy early
    try:
        result["entropy"] = _shannon_entropy(b)
    except Exception as e:
        result["errors"].append(f"entropy_error:{type(e).__name__}")

    # EXIF subset
    try:
        result["exif"] = _safe_exif(b)
    except Exception as e:
        result["errors"].append(f"exif_error:{type(e).__name__}")

    # Pillow decode + derived features
    img: Optional[Image.Image] = None
    try:
        img = _bytes_to_image(b)
        w, h = img.size
        result["width"] = int(w)
        result["height"] = int(h)
        result["icc_profile_len"] = _icc_profile_size(img)

        if do_hash:
            try:
                result["phash"] = str(imagehash.phash(img))
                result["dhash"] = str(imagehash.dhash(img))
                result["ahash"] = str(imagehash.average_hash(img))
            except Exception as e:
                result["errors"].append(f"hash_error:{type(e).__name__}")

        if do_ocr:
            try:
                # Upscale small images to help OCR a bit
                if min(w, h) < 240:
                    scale = max(2, int(240 / max(1, min(w, h))))
                    img_for_ocr = img.resize((w * scale, h * scale))
                else:
                    img_for_ocr = img
                # Convert to grayscale for OCR robustness
                gray = ImageOps.grayscale(img_for_ocr)
                result["ocr_text"] = pytesseract.image_to_string(gray)
            except Exception as e:
                result["errors"].append(f"ocr_error:{type(e).__name__}")

        if do_qr:
            try:
                codes = qr_decode(img)
                links = []
                for obj in codes:
                    try:
                        links.append(obj.data.decode("utf-8", "ignore"))
                    except Exception:
                        # raw bytes fallback
                        links.append(str(obj.data))
                result["qr_links"] = list(sorted(set(links)))
            except Exception as e:
                result["errors"].append(f"qr_error:{type(e).__name__}")

    except Exception as e:
        result["errors"].append(f"pillow_error:{type(e).__name__}")

    # Heuristics / flags
    try:
        if result["icc_profile_len"] and result["icc_profile_len"] > 512*1024:
            result["suspicious_flags"].append("oversized_icc_profile")
    except Exception:
        pass

    try:
        if _detect_appended_data(b, result["mime_magic"] or ""):
            result["suspicious_flags"].append("appended_data_after_eof")
    except Exception:
        pass

    try:
        if result["entropy"] >= 7.5:
            result["suspicious_flags"].append("high_entropy_image")
    except Exception:
        pass

    return result


def process_many_images(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convenience: takes a list of dicts with at least {'payload': bytes, ...}
    (e.g., from email parser) and returns processed results with original metadata.
    """
    out: List[Dict[str, Any]] = []
    for img in images:
        payload = img.get("payload", b"")
        res = process_image_bytes(payload)
        # Attach source metadata (filename, content_id, etc.)
        merged = {**{k: v for k, v in img.items() if k != "payload"}, **res}
        out.append(merged)
    return out
