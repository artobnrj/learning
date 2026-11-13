"""Level 1: each skill on its own - no API calls."""

import io
import re
import urllib.error

import pytest

from skills import run_skill
from skills import web


def ok(name, **kwargs):
    result, is_error = run_skill(name, kwargs)
    assert not is_error, result
    return result


def err(name, **kwargs):
    result, is_error = run_skill(name, kwargs)
    assert is_error, f"expected an error, got {result!r}"
    return result


# --- calculator -------------------------------------------------------------

@pytest.mark.parametrize(
    "expression, expected",
    [
        ("2+3*4", "14"),
        ("(1+2)**3", "27"),
        ("17/100*2340", "397.8"),
        ("-5 + 2", "-3"),
        ("7 // 2", "3"),
        ("7 % 3", "1"),
        ("2,340 * 2", "4680"),
        ("0.1 + 0.2", "0.3"),
    ],
)
def test_calculator_math(expression, expected):
    assert ok("calculator", expression=expression) == expected


def test_calculator_divide_by_zero():
    assert "ZeroDivisionError" in err("calculator", expression="1/0")


@pytest.mark.parametrize(
    "expression",
    ["__import__('os')", "x + 1", "abs(-1)", "(1).real", "True + 1", "'a' * 3", "[1, 2]"],
)
def test_calculator_rejects_non_arithmetic(expression):
    err("calculator", expression=expression)


def test_calculator_rejects_huge_exponent():
    assert "Exponent too large" in err("calculator", expression="9**9**9")


def test_calculator_rejects_long_input():
    err("calculator", expression="1+" * 200 + "1")


# --- get_current_time -------------------------------------------------------

def test_time_defaults_to_utc():
    result = ok("get_current_time", timezone=None)
    assert re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", result)
    assert "UTC" in result


def test_time_in_timezone():
    result = ok("get_current_time", timezone="Asia/Tokyo")
    assert "+09:00" in result and "Asia/Tokyo" in result


def test_time_bad_timezone():
    assert "Unknown timezone" in err("get_current_time", timezone="Mars/Base")


# --- read_file / write_file -------------------------------------------------

def test_write_then_read(workspace):
    assert "Wrote 5 bytes" in ok("write_file", path="notes.txt", content="hello")
    assert (workspace / "notes.txt").read_text() == "hello"
    assert ok("read_file", path="notes.txt") == "hello"


def test_write_creates_subfolders(workspace):
    ok("write_file", path="a/b/c.txt", content="deep")
    assert (workspace / "a" / "b" / "c.txt").read_text() == "deep"


def test_read_missing_file(workspace):
    assert "FileNotFoundError" in err("read_file", path="nope.txt")


@pytest.mark.parametrize("path", ["../secret.txt", "../../etc/passwd", "/etc/passwd", "a/../../x.txt"])
def test_read_outside_workspace_rejected(workspace, path):
    (workspace.parent / "secret.txt").write_text("top secret")
    assert "outside the workspace" in err("read_file", path=path)


@pytest.mark.parametrize("path", ["../x.txt", "/tmp/x.txt"])
def test_write_outside_workspace_rejected(workspace, path):
    assert "outside the workspace" in err("write_file", path=path, content="x")
    assert not (workspace.parent / "x.txt").exists()


def test_symlink_escape_rejected(workspace):
    (workspace.parent / "secret.txt").write_text("top secret")
    (workspace / "link.txt").symlink_to(workspace.parent / "secret.txt")
    assert "outside the workspace" in err("read_file", path="link.txt")


def test_size_caps(workspace, monkeypatch):
    from skills import files

    monkeypatch.setattr(files, "MAX_BYTES", 10)
    assert "larger than" in err("write_file", path="big.txt", content="x" * 11)
    (workspace / "big.txt").write_text("x" * 11)
    assert "larger than" in err("read_file", path="big.txt")


# --- fetch_url --------------------------------------------------------------

class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, content_type: str = "text/html; charset=utf-8"):
        super().__init__(body)
        from email.message import Message

        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "example.com"])
def test_fetch_rejects_bad_schemes(url):
    assert "Only http" in err("fetch_url", url=url)


def test_fetch_strips_html(monkeypatch):
    html = b"<html><head><style>p{}</style><script>alert(1)</script></head>" \
           b"<body><h1>Title</h1><p>Hello &amp; welcome</p></body></html>"
    monkeypatch.setattr(web.urllib.request, "urlopen", lambda req, timeout: FakeResponse(html))
    assert ok("fetch_url", url="https://example.com") == "Title Hello & welcome"


def test_fetch_plain_text_kept(monkeypatch):
    monkeypatch.setattr(
        web.urllib.request, "urlopen",
        lambda req, timeout: FakeResponse(b"a <b> c", "text/plain"),
    )
    assert ok("fetch_url", url="https://example.com/x.txt") == "a <b> c"


def test_fetch_truncates(monkeypatch):
    monkeypatch.setattr(web, "MAX_CHARS", 10)
    monkeypatch.setattr(
        web.urllib.request, "urlopen",
        lambda req, timeout: FakeResponse(b"x" * 25, "text/plain"),
    )
    result = ok("fetch_url", url="https://example.com")
    assert result.startswith("x" * 10) and "truncated, 15 more characters" in result


def test_fetch_http_error(monkeypatch):
    def boom(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", None, None)

    monkeypatch.setattr(web.urllib.request, "urlopen", boom)
    assert "HTTP 404" in err("fetch_url", url="https://example.com/missing")


@pytest.mark.network
def test_fetch_real_page():
    assert "Example Domain" in ok("fetch_url", url="https://example.com")
