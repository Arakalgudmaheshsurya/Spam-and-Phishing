"""
src/email_parser/extract_urls.py
--------------------------------
Utilities to extract and clean URLs from both HTML and plaintext email content.
"""

import re
from bs4 import BeautifulSoup
from typing import List

# Regex to match HTTP/HTTPS URLs
URL_REGEX = re.compile(
    r'https?://[^\s"\'<>()]+',
    re.IGNORECASE
)

def extract_urls_from_html(html: str) -> List[str]:
    """
    Extracts URLs from HTML content using BeautifulSoup and regex fallback.

    Args:
        html (str): Raw HTML email body.

    Returns:
        List[str]: Clean list of URLs.
    """
    urls = set()

    try:
        soup = BeautifulSoup(html, "lxml")
        # <a href=""> links
        for a in soup.find_all("a", href=True):
            urls.add(a["href"].strip())

        # Inline scripts or onclick URLs
        for tag in soup.find_all(onclick=True):
            match = URL_REGEX.search(tag["onclick"])
            if match:
                urls.add(match.group(0))

        # Fallback: raw URLs in HTML text
        urls.update(URL_REGEX.findall(soup.get_text(" ")))

    except Exception:
        # fallback regex if HTML parsing fails
        urls.update(URL_REGEX.findall(html))

    # Cleanup
    cleaned = []
    for url in urls:
        # Strip punctuation and fragments
        u = url.strip().rstrip(").,;\"'")
        # Remove anchors (#something)
        u = u.split("#")[0]
        if u:
            cleaned.append(u)

    return sorted(set(cleaned))


def extract_urls_from_text(text: str) -> List[str]:
    """
    Extracts URLs from plaintext email body.

    Args:
        text (str): Plaintext email content.

    Returns:
        List[str]: List of URLs.
    """
    urls = URL_REGEX.findall(text)
    cleaned = []

    for url in urls:
        u = url.strip().rstrip(").,;\"'")
        u = u.split("#")[0]
        if u:
            cleaned.append(u)

    return sorted(set(cleaned))


def extract_all_urls(text: str, html: str) -> List[str]:
    """
    Combines URL extraction from both HTML and text.

    Args:
        text (str): Plaintext email body.
        html (str): HTML email body.

    Returns:
        List[str]: Unique URLs found in the email.
    """
    urls = set()
    urls.update(extract_urls_from_text(text))
    urls.update(extract_urls_from_html(html))
    return sorted(urls)
