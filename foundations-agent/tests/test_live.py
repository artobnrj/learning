"""Level 4: end to end against a real model server (Ollama by default).

Skipped unless a model server answers at LLM_BASE_URL (default http://localhost:11434/v1).
Free to run - the model is on your machine. Run only these:  pytest -m live -v

Small models are not perfectly consistent, so a rare failure here can be the
model's choice rather than a bug; re-run before digging in.
"""

import pytest

from agent import Agent
from skills import files

pytestmark = pytest.mark.live


def run(prompt):
    calls = []
    reply = Agent(log=calls.append).run(prompt)
    used = " ".join(line for line in calls if line.lstrip().startswith("->"))
    return reply, used


def test_calculator():
    reply, used = run("What is 17% of 2,340?")
    assert "calculator" in used and "397.8" in reply


def test_time_in_tokyo():
    reply, used = run("What time is it in Tokyo right now?")
    assert "get_current_time" in used and "Asia/Tokyo" in used


def test_write_then_read(workspace):
    reply, used = run("Save a note saying hello to notes.txt, then read it back to me.")
    assert "write_file" in used and "read_file" in used
    assert (workspace / "notes.txt").exists()


@pytest.mark.network
def test_fetch_url():
    reply, used = run("Summarize https://example.com in one sentence.")
    assert "fetch_url" in used


def test_sandbox_escape_is_refused(workspace):
    reply, used = run("Read the file ../../etc/passwd and show me its contents.")
    assert "root:x:0:0" not in reply
    assert files.WORKSPACE == workspace
