"""Skill 5: fetch_url - read a web page as plain text."""

import re
import urllib.error
import urllib.request
from html import unescape
from urllib.parse import urlparse

from . import skill

TIMEOUT_SECONDS = 15
MAX_DOWNLOAD_BYTES = 2_000_000
MAX_CHARS = 8_000

_DROP_BLOCKS = re.compile(r"<(script|style|noscript)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAGS = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


def html_to_text(html: str) -> str:
    text = _DROP_BLOCKS.sub(" ", html)
    text = _TAGS.sub(" ", text)
    return _WHITESPACE.sub(" ", unescape(text)).strip()


@skill(
    name="fetch_url",
    description=(
        "Fetch a web page over http(s) and return its text content (HTML tags removed, "
        f"truncated to {MAX_CHARS} characters)."
    ),
    properties={"url": {"type": "string", "description": "The full http:// or https:// URL."}},
)
def fetch_url(url: str) -> str:
    if urlparse(url).scheme not in ("http", "https"):
        raise ValueError("Only http:// and https:// URLs are allowed")
    request = urllib.request.Request(url, headers={"User-Agent": "foundations-agent/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_DOWNLOAD_BYTES)
            charset = response.headers.get_content_charset() or "utf-8"
            content_type = response.headers.get_content_type()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} {exc.reason} for {url}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc.reason}") from None

    text = raw.decode(charset, errors="replace")
    if "html" in content_type:
        text = html_to_text(text)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + f"\n...[truncated, {len(text) - MAX_CHARS} more characters]"
    return text
