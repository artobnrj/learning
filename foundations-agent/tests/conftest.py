import sys
import urllib.request
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# Make `import agent` / `import skills` work no matter where pytest is started from.
sys.path.insert(0, str(ROOT))
# Live tests read LLM_BASE_URL / LLM_MODEL from .env too, just like `python agent.py`.
load_dotenv(ROOT / ".env")

from agent import settings  # noqa: E402
from skills import files  # noqa: E402


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point the file skills at a temporary workspace so tests never touch the real one."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setattr(files, "WORKSPACE", ws)
    return ws


def _has_network() -> bool:
    # Same path as the fetch_url skill (honours HTTPS_PROXY), so the check is accurate.
    try:
        urllib.request.urlopen("https://example.com", timeout=5).close()
        return True
    except OSError:
        return False


def _model_server_up() -> bool:
    """True if something answers the OpenAI-compatible /models endpoint (e.g. Ollama)."""
    cfg = settings()
    request = urllib.request.Request(
        cfg["base_url"].rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {cfg['api_key']}"},
    )
    try:
        urllib.request.urlopen(request, timeout=3).close()
        return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    server = None
    network = None
    for item in items:
        if "live" in item.keywords:
            if server is None:
                server = _model_server_up()
            if not server:
                item.add_marker(pytest.mark.skip(reason=f"no model server at {settings()['base_url']}"))
        if "network" in item.keywords:
            if network is None:
                network = _has_network()
            if not network:
                item.add_marker(pytest.mark.skip(reason="no network access"))
