# Foundations Agent

A minimal AI agent in plain Python: one loop, five skills, no framework. An **open-source LLM running on your own machine** sits in a loop and can call five **skills** (tools) to do things it can't do reliably on its own. It's free to run, needs no account, and nothing leaves your computer except `fetch_url` requests.

| Skill | What it does |
|---|---|
| `calculator` | Exact arithmetic (safe parser, no `eval`) |
| `get_current_time` | Current date/time in any timezone |
| `read_file` | Read a text file from `workspace/` |
| `write_file` | Write a text file into `workspace/` |
| `fetch_url` | Download a web page as plain text |

## The model: Qwen3 8B on Ollama

- **[Qwen3](https://ollama.com/library/qwen3) 8B** is an open-weight model under the Apache 2.0 license. It's among the most reliable small models at **tool calling**, which is what an agent depends on. It needs about 6 GB of RAM and runs without a GPU, just slower.
- **[Ollama](https://ollama.com)** downloads and runs the model, and serves it through the standard OpenAI-compatible chat API at `http://localhost:11434/v1`.

The agent talks to that API with the `openai` Python package. The package is only the client; nothing goes to OpenAI. Because nearly every model server uses this API, you can switch servers or models by editing `.env` (see `.env.example`).

## Setup

You need Python 3.10+ and about 6 GB of free disk space for the model.

**1. Install Ollama and download the model:**

```bash
# macOS / Windows: install the app from https://ollama.com/download (it runs in the background)
# Linux:
curl -fsSL https://ollama.com/install.sh | sh

ollama pull qwen3:8b        # ~5 GB, one time. Low on RAM? use qwen3:4b and set LLM_MODEL in .env
```

**2. Set up and run the agent:**

```bash
cd foundations-agent
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # optional: only needed to change model/server
python agent.py
```

`.env` is gitignored. Environment variables that are already set take precedence over `.env`.

Example session:

```
you> What's 17% of 2,340, what time is it in Tokyo, and save both to notes.txt
  -> calculator({"expression": "17/100*2340"})
  <- 397.8
  -> get_current_time({"timezone": "Asia/Tokyo"})
  <- 2026-11-13T18:42:10+09:00 (Friday, Asia/Tokyo)
  -> write_file({"path": "notes.txt", "content": "17% of 2,340 = 397.8\n..."})
  <- Wrote 62 bytes to notes.txt
agent> 17% of 2,340 is 397.8, it's 6:42 PM Friday in Tokyo, and both are saved in notes.txt.
```

The `->` / `<-` lines show each skill call and its result, so you can watch the agent work step by step. The first reply takes longer while Ollama loads the model into memory.

**If something goes wrong:**
- *"Can't reach a model server"*: Ollama isn't running. Open the app, or run `ollama serve`.
- *"Model ... isn't available"*: run `ollama pull qwen3:8b`, or whichever model `LLM_MODEL` names.
- *The model answers without using a skill, or picks the wrong one*: small models are less dependable than big hosted ones. Ask again more explicitly ("use the calculator"), or try a bigger model such as `qwen3:14b`.

## How an agent works

```
you ──► agent.run(message)
          │
          ▼
   ┌─► send system prompt + conversation + tool list to the model
   │      │
   │      ▼
   │   model replies. Any tool_calls?
   │      ├─ yes ─► run each requested skill,
   │      │         add one "tool" message per result ───────────┐
   │      └─ no  ─► return the text to you                       │
   │                (or say it was cut off / declined)           │
   └─────────────────────────────────────────────────────────────┘
          (max 10 steps, so it can never loop forever)
```

That loop is `Agent._loop` in [`agent.py`](agent.py). Everything else is support code:

- **The model decides, your code acts.** The model never runs anything itself. It replies with a `tool_calls` entry naming a skill and its JSON arguments, and `run_skill` runs your Python function.
- **Memory is just the message list.** `self.messages` holds the whole conversation and is sent again on every call (models are stateless), so follow-up questions work.
- **Errors go back to the model.** When a skill fails, or the model sends broken JSON, the result is sent back starting with `ERROR:`, so the model can explain or retry. A failing skill never crashes the agent.
- **Skills are sandboxed.** File skills can't leave `workspace/`, and `fetch_url` only allows http(s).
- **Hidden reasoning is removed.** Qwen3 "thinks" before answering. Any `<think>...</think>` text that reaches the reply is stripped, so you only see the answer.

## Project layout

```
agent.py              the agent loop + interactive prompt + settings
skills/__init__.py    @skill decorator, TOOLS list, run_skill()
skills/calculator.py  skill 1
skills/clock.py       skill 2
skills/files.py       skills 3 & 4 (workspace sandbox)
skills/web.py         skill 5
workspace/            the only folder the file skills can touch
.env.example          model/server settings to copy into .env
tests/                4 levels of tests (see below)
```

## Adding a skill

Write a function in `skills/` and put the decorator on it:

```python
from . import skill

@skill(
    name="list_files",
    description="List all files in the workspace folder.",
    properties={},
)
def list_files() -> str:
    ...
```

Then import the module at the bottom of `skills/__init__.py`. The decorator builds the tool definition the model sees. To make an input optional, leave it out of `required=[...]` and give the Python parameter a default (see `skills/clock.py`). Write the description for the model: say what the skill does and when to use it. Small models rely on that heavily.

Good first exercises: `list_files`, `web_search`, a `remember`/`recall` pair that saves notes across runs.

## Tests

```bash
pytest -m "not live and not network"   # fast, offline (levels 1-3), no model needed
pytest                                 # everything; live/network tests skip themselves when they can't run
pytest -m live -v                      # only the real-model tests (needs Ollama running; free)
```

| Level | File | What it checks |
|---|---|---|
| 1 | `test_skills.py` | Each skill: correct results, bad input, sandbox escapes, size caps, HTML stripping |
| 2 | `test_registry.py` | Exactly 5 well-formed tool definitions; unknown or broken skills become errors, not crashes |
| 3 | `test_agent_loop.py` | The loop, with a fake model: tool rounds, several calls at once, errors, bad JSON, `<think>` removal, length/filter stops, runaway guard, memory, settings, and the exact JSON a real server receives |
| 4 | `test_live.py` | The real model, end to end: one prompt per skill plus a sandbox-escape attempt |

## Settings

In `.env` (see `.env.example`):

- `LLM_BASE_URL`: the model server (default: Ollama at `http://localhost:11434/v1`)
- `LLM_MODEL`: the model name (default `qwen3:8b`)
- `LLM_API_KEY`: only needed for hosted services; local servers ignore it

At the top of `agent.py`:

- `MAX_STEPS`: the safety limit on loop iterations.
- `TIMEOUT_SECONDS`: how long to wait for a reply (local models on a CPU can be slow).
