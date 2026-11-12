# Foundations Agent

A minimal AI agent in plain Python: one loop, five skills, no framework. Claude runs in a loop and can call five **skills** (tools) to do things it can't do reliably on its own.

| Skill | What it does |
|---|---|
| `calculator` | Exact arithmetic (safe parser, no `eval`) |
| `get_current_time` | Current date/time in any timezone |
| `read_file` | Read a text file from `workspace/` |
| `write_file` | Write a text file into `workspace/` |
| `fetch_url` | Download a web page as plain text |

## How an agent works

```
you ──► agent.run(message)
          │
          ▼
   ┌─► send conversation + tool list to Claude
   │      │
   │      ▼
   │   Claude replies. stop_reason?
   │      ├─ "tool_use" ─► run each requested skill
   │      │                send ALL results back in one message ──┐
   │      ├─ "end_turn" ─► return the text to you                 │
   │      └─ "refusal" / "max_tokens" ─► tell you and stop        │
   └──────────────────────────────────────────────────────────────┘
          (max 10 steps, so it can never loop forever)
```

That loop is `Agent._loop` in [`agent.py`](agent.py). Everything else is support code:

- **The model decides, your code acts.** Claude never runs anything itself. It asks for a tool by name with JSON arguments, and `run_skill` runs your Python function.
- **Memory is just the message list.** `self.messages` holds the whole conversation, so follow-up questions work.
- **Errors go back to the model.** When a skill fails, the error text is returned with `is_error: true` so Claude can explain it or try again. A failing skill never crashes the agent.
- **Skills are sandboxed.** File skills can't leave `workspace/`, and `fetch_url` only allows http(s).

## Setup

Needs Python 3.10+.

```bash
cd foundations-agent
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...                   # from https://console.anthropic.com
python agent.py
```

Example session:

```
you> What's 17% of 2,340, what time is it in Tokyo, and save both to notes.txt
  -> calculator({"expression": "17/100*2340"})
  <- 397.8
  -> get_current_time({"timezone": "Asia/Tokyo"})
  <- 2026-11-13T01:42:10+09:00 (Friday, Asia/Tokyo)
  -> write_file({"path": "notes.txt", "content": "17% of 2,340 = 397.8\n..."})
  <- Wrote 62 bytes to notes.txt
agent> 17% of 2,340 is 397.8, it's 1:42 AM Friday in Tokyo, and both are saved in notes.txt.
```

The `->` / `<-` lines show each skill call and its result, so you can watch the agent work step by step.

## Project layout

```
agent.py              the agent loop + interactive prompt
skills/__init__.py    @skill decorator, TOOLS list, run_skill()
skills/calculator.py  skill 1
skills/clock.py       skill 2
skills/files.py       skills 3 & 4 (workspace sandbox)
skills/web.py         skill 5
workspace/            the only folder the file skills can touch
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

Then import the module at the bottom of `skills/__init__.py`. The decorator builds the tool definition Claude sees (with `strict: true`, so Claude's arguments always match your schema). Write the description for the model: say what the skill does and when to use it.

Good first exercises: `list_files`, `web_search`, a `remember`/`recall` pair that saves notes across runs.

## Tests

```bash
pytest -m "not live and not network"   # fast, offline, free (levels 1-3)
pytest                                 # everything; live/network tests skip themselves when they can't run
pytest -m live -v                      # only the real-API tests (needs ANTHROPIC_API_KEY, costs a few cents)
```

| Level | File | What it checks |
|---|---|---|
| 1 | `test_skills.py` | Each skill: correct results, bad input, sandbox escapes, size caps, HTML stripping |
| 2 | `test_registry.py` | Exactly 5 well-formed tool definitions; unknown or broken skills become errors, not crashes |
| 3 | `test_agent_loop.py` | The loop, with a fake Claude: tool rounds, parallel calls, errors, pause/refusal/max_tokens, runaway guard, memory, request shape |
| 4 | `test_live.py` | Real API, end to end: one prompt per skill plus a sandbox-escape attempt |

## Settings

These are at the top of `agent.py`:

- `MODEL`: `claude-opus-5-5`
- `EFFORT`: how hard Claude thinks (`low` → `max`). `medium` keeps it quick and cheap; raise it for harder tasks.
- `MAX_STEPS`: the safety limit on loop iterations.
- Refusal fallback is on (`fallbacks="default"`). If Claude's safety system declines a request, the API retries it on a fallback model automatically.
