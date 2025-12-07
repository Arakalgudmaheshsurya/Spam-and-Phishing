"""
src/email_parser/parser.py
--------------------------
Core MIME email parser that extracts headers, text, HTML, and attachments
from raw email (.eml) bytes.
"""

import base64
import re
import io
from email import policy
from email.parser import BytesParser
from typing import Dict, Any, List, Tuple

def parse_email(raw_bytes: bytes) -> Tuple[Any, List[Tuple[str, bytes, Any]]]:
    """
    Parses a raw email into its MIME parts.

    Returns:
        msg: EmailMessage object
        parts: list of tuples (content_type, payload_bytes, part)
    """
    msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    parts = []

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            payload = part.get_payload(decode=True) or b""
            parts.append((ctype, payload, part))
    else:
        parts.append((msg.get_content_type(), msg.get_payload(decode=True) or b"", msg))

    return msg, parts


def extract_basic_info(msg) -> Dict[str, Any]:
    """
    Extracts common email headers and metadata.
    """
    return {
        "from": msg.get("From", ""),
        "to": msg.get("To", ""),
        "subject": msg.get("Subject", ""),
        "date": msg.get("Date", ""),
        "reply_to": msg.get("Reply-To", ""),
        "spf": msg.get("Received-SPF", ""),
        "dkim": bool(msg.get("DKIM-Signature")),
        "dmarc": msg.get("Authentication-Results", ""),
    }


def extract_text_and_html(parts: List[Tuple[str, bytes, Any]]) -> Dict[str, str]:
    """
    Separates plain text and HTML content from MIME parts.
    """
    text_content, html_content = "", ""
    for ctype, payload, part in parts:
        charset = part.get_content_charset() or "utf-8"
        try:
            decoded = payload.decode(charset, "ignore")
        except Exception:
            decoded = payload.decode("utf-8", "ignore")

        if ctype == "text/plain":
            text_content += decoded + "\n"
        elif ctype == "text/html":
            html_content += decoded + "\n"

    return {"text": text_content.strip(), "html": html_content.strip()}


def extract_images(parts: List[Tuple[str, bytes, Any]]) -> List[Dict[str, Any]]:
    """
    Extracts attached or inline images.
    """
    images = []
    for ctype, payload, part in parts:
        if ctype.startswith("image/"):
            images.append({
                "filename": part.get_filename() or "unnamed",
                "content_type": ctype,
                "payload": payload,
                "content_id": part.get("Content-ID", ""),
                "content_location": part.get("Content-Location", ""),
            })

    return images
