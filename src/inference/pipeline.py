"""
src/inference/pipeline.py
-------------------------
End-to-end feature extraction + scoring pipeline.

Steps:
- Parse MIME email
- Extract headers, plain text, HTML
- Extract inline/attached images & run OCR/QR/EXIF/etc.
- Collect URLs from text, HTML, OCR, QR codes
- Compute lightweight URL features
- Run:
    * text model (TF-IDF + LogisticRegression)
    * URL model (LightGBM / RandomForest)
- Fuse scores into a single phish_score in [0, 1]

Return structure (main keys):
    headers, text, html_len,
    urls, url_feats,
    images, ocr_texts, qr_links, image_flags,
    text_score, url_scores, url_max_score, phish_score,
    errors
"""

from typing import Dict, Any, List
from itertools import chain
from src.utils.config import CONFIG

from src.email_parser.parser import (
    parse_email,
    extract_basic_info,
    extract_text_and_html,
    extract_images,
)
from src.email_parser.extract_urls import extract_all_urls
from src.features.image_features import process_many_images

# New model wrappers
from src.models.text_model import score_text
from src.models.url_model import score_urls


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _dedupe(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _basic_url_features(urls: List[str]) -> List[Dict[str, Any]]:
    """
    Lightweight URL feature sketch (fast + dependency-light).

    These are the same fields expected by src/models/url_model.py:
        length, dots, has_ip, contains_login, has_unicode
    """
    feats: List[Dict[str, Any]] = []
    for u in urls:
        hostname = ""
        try:
            if u.startswith(("http://", "https://")):
                hostname = u.split("/")[2]
        except Exception:
            hostname = ""

        has_ip = False
        try:
            if hostname and hostname.replace(".", "").isdigit():
                has_ip = True
        except Exception:
            has_ip = False

        f = {
            "url": u,
            "length": len(u),
            "dots": u.count("."),
            "has_ip": int(has_ip),
            "contains_login": int(
                any(k in u.lower() for k in ["login", "signin", "verify", "update", "password"])
            ),
            "has_unicode": int(any(ord(c) > 127 for c in u)),
        }
        feats.append(f)
    return feats


def _combine_scores(text_score: float, url_max_score: float, image_flags: List[str]) -> float:
    """
    Simple heuristic fusion:
      - 0.6 weight to text model
      - 0.4 weight to strongest URL score
      - +0.05 per suspicious image flag (up to 3 flags)
    """
    ts = float(text_score or 0.0)
    us = float(url_max_score or 0.0)

    base = 0.6 * ts + 0.4 * us

    if image_flags:
        base += 0.05 * min(len(image_flags), 3)

    # clamp to [0, 1]
    if base < 0.0:
        base = 0.0
    if base > 1.0:
        base = 1.0
    return base


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_all(raw_bytes: bytes) -> Dict[str, Any]:
    """
    Master function used by the API.

    Returns a dict with:
      headers, text, html_len,
      urls, url_feats,
      images (processed),
      ocr_texts, qr_links, image_flags,
      text_score, url_scores, url_max_score, phish_score,
      errors
    """
    errors: List[str] = []

    # 1) Parse email & parts
    try:
        msg, parts = parse_email(raw_bytes)
    except Exception as e:
        # Catastrophic parse failure; surface minimal info
        return {"errors": [f"parse_email_error:{type(e).__name__}"]}

    # 2) Headers
    try:
        headers = extract_basic_info(msg)
    except Exception as e:
        headers = {}
        errors.append(f"extract_basic_info_error:{type(e).__name__}")

    # 3) Text & HTML
    text, html = "", ""
    try:
        th = extract_text_and_html(parts)
        text = th.get("text", "") or ""
        html = th.get("html", "") or ""
    except Exception as e:
        errors.append(f"text_html_error:{type(e).__name__}")

    # 4) Images (attachments + inline)
    img_meta: List[Dict[str, Any]] = []
    try:
        img_meta = extract_images(parts)
    except Exception as e:
        errors.append(f"extract_images_error:{type(e).__name__}")

    # 5) Process images (OCR/QR/EXIF/hashes/heuristics)
    processed_images: List[Dict[str, Any]] = []
    try:
        processed_images = process_many_images(img_meta) if img_meta else []
    except Exception as e:
        errors.append(f"process_images_error:{type(e).__name__}")

    # 6) OCR text & QR links from images
    ocr_texts = [im.get("ocr_text", "") for im in processed_images if im.get("ocr_text")]
    qr_links = list(
        _dedupe(list(chain.from_iterable(im.get("qr_links", []) for im in processed_images)))
    )

    # 7) URLs from text + HTML + OCR + QR
    urls_core = extract_all_urls(text, html)
    urls_from_ocr: List[str] = []
    for blob in ocr_texts:
        try:
            urls_from_ocr.extend(extract_all_urls(blob, ""))  # text-only
        except Exception:
            pass

    all_urls = _dedupe(urls_core + urls_from_ocr + qr_links)

    # 8) Quick URL features
    url_feats = _basic_url_features(all_urls)

    # 9) Aggregate suspicious flags from images
    image_flags = sorted(
        _dedupe(list(chain.from_iterable(im.get("suspicious_flags", []) for im in processed_images)))
    )

    # 10) TEXT MODEL SCORE
    text_blob = (headers.get("subject", "") or "") + "\n" + text + "\n" + "\n".join(ocr_texts)
    text_score = None
    try:
        text_score = score_text(text_blob)
    except Exception as e:
        errors.append(f"text_model_error:{type(e).__name__}")

    # 11) URL MODEL SCORES
    url_scores = None
    url_max_score = None
    try:
        if url_feats:
            url_scores = score_urls(url_feats)
            if url_scores is not None and len(url_scores) > 0:
                url_max_score = max(url_scores)
    except Exception as e:
        errors.append(f"url_model_error:{type(e).__name__}")

    # 12) Combined phishing score
    phish_score = _combine_scores(
        text_score if text_score is not None else 0.0,
        url_max_score if url_max_score is not None else 0.0,
        image_flags,
    )
    # 13) Discrete label from score
    if phish_score >= CONFIG.PHISH_HIGH_THRESHOLD:
        label = "phishing"
    elif phish_score <= CONFIG.PHISH_LOW_THRESHOLD:
        label = "legit"
    else:
        label = "suspicious"
    # 14) Explanations (why we think so)
    explanations = {
        "text_score": text_score,
        "url_max_score": url_max_score,
        "top_urls": [],
        "image_flags": image_flags,
    }

    # Attach URL + score pairs if we have them
    if url_scores is not None and url_feats:
        # pair up and sort by score desc
        url_pairs = sorted(
            zip(url_feats, url_scores),
            key=lambda p: p[1],
            reverse=True
        )
        # keep top 3
        explanations["top_urls"] = [
            {
                "url": uf.get("url"),
                "score": float(score),
                "contains_login": uf.get("contains_login"),
                "has_ip": uf.get("has_ip"),
            }
            for uf, score in url_pairs[:3]
        ]

    return {
        "headers": headers,
        "text": text,
        "html_len": len(html),                 # keep HTML length only (avoid bloat)
        "urls": all_urls,
        "url_feats": url_feats,
        "images": processed_images,            # includes OCR/QR/hash/exif per image
        "ocr_texts": [t[:1000] for t in ocr_texts],  # truncated for safety
        "qr_links": qr_links,
        "image_flags": image_flags,
        "text_score": text_score,
        "url_scores": url_scores,
        "url_max_score": url_max_score,
        "phish_score": phish_score,
        "label": label,                    # <── NEW
        "explanations": explanations,      # <── NEW
        "errors": errors,
    }
