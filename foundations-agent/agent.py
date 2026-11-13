"""Foundations Agent: an LLM in a loop that can call skills (tools).

The whole idea of an "agent" fits in one function, `Agent._loop`:

    send the conversation + tool list to the model
    while the model asks to use tools:
        run each tool, send the results back
    return the model's final answer

The model is an open-source LLM (default: Qwen3 8B) served locally by Ollama,
spoken to through the OpenAI-compatible chat API that almost every model server
supports. Change LLM_BASE_URL / LLM_MODEL in .env to use another server or model.

Run it:  python agent.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

import openai
from dotenv import load_dotenv

from skills import TOOLS, run_skill

DEFAULT_BASE_URL = "http://localhost:11434/v1"  # Ollama's OpenAI-compatible endpoint
DEFAULT_MODEL = "qwen3:8b"
MAX_STEPS = 10  # safety net: stop if the model keeps calling tools forever
TIMEOUT_SECONDS = 300  # local models on a CPU can be slow

SYSTEM_PROMPT = """You are Foundations Agent, a small helpful assistant with five skills:
calculator, get_current_time, read_file, write_file and fetch_url.
Use a skill whenever it gives a more reliable answer than guessing - always use the
calculator for arithmetic and get_current_time for anything about "now".
File paths are relative to your workspace folder; you cannot access files outside it.
If a skill returns an error, explain it plainly or try a corrected call.
Keep answers short and direct."""

# Reasoning models (Qwen3, DeepSeek-R1, ...) may include their private reasoning
# in <think>...</think> tags; the user only needs the answer.
_THINK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)


def settings() -> dict[str, str]:
    """Connection settings, read at runtime so values from .env are picked up."""
    return {
        "base_url": os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
        "model": os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        # Local servers ignore the key, but the client library requires one.
        "api_key": os.environ.get("LLM_API_KEY", "not-needed"),
    }


def _preview(value: Any, limit: int = 120) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _parse_arguments(raw: Any) -> Any:
    """Tool arguments normally arrive as a JSON string; a few servers send an object."""
    if isinstance(raw, (dict, list)):
        return raw
    if raw in (None, ""):
        return {}
    return json.loads(raw)


class Agent:
    def __init__(
        self,
        client: Any | None = None,
        model: str | None = None,
        log: Callable[[str], None] = print,
        max_steps: int = MAX_STEPS,
    ):
        cfg = settings()
        self.client = client or openai.OpenAI(
            base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=TIMEOUT_SECONDS, max_retries=1
        )
        self.model = model or cfg["model"]
        self.log = log
        self.max_steps = max_steps
        self.messages: list[dict[str, Any]] = []  # the whole conversation = the agent's memory

    def _call_model(self):
        return self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *self.messages],
            tools=TOOLS,
        )

    def run(self, user_input: str) -> str:
        """Handle one user message, running tools as needed. Returns the final reply."""
        start = len(self.messages)
        self.messages.append({"role": "user", "content": user_input})
        try:
            return self._loop()
        except Exception:
            del self.messages[start:]  # drop the half-finished turn so the chat stays valid
            raise

    def _loop(self) -> str:
        for _ in range(self.max_steps):
            choice = self._call_model().choices[0]
            message = choice.message
            tool_calls = message.tool_calls or []

            # Record the model's turn, including any tool calls it made.
            turn: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            if tool_calls:
                turn["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.function.name, "arguments": call.function.arguments},
                    }
                    for call in tool_calls
                ]
            self.messages.append(turn)

            if tool_calls:
                for call in tool_calls:
                    self.messages.append(self._run_tool_call(call))
                continue  # let the model read the results and decide what's next

            text = _THINK.sub("", message.content or "").strip()
            if choice.finish_reason == "length":
                return (text + "\n" if text else "") + "[Reply cut off: hit the model's length limit.]"
            if choice.finish_reason == "content_filter":
                return "[The model declined this request.]"
            return text

        return f"[Stopped after {self.max_steps} steps without a final answer.]"

    def _run_tool_call(self, call) -> dict[str, Any]:
        name = call.function.name
        try:
            args = _parse_arguments(call.function.arguments)
        except json.JSONDecodeError:
            result, is_error = f"Arguments were not valid JSON: {call.function.arguments!r}", True
            args = call.function.arguments
        else:
            result, is_error = run_skill(name, args)
        self.log(f"  -> {name}({_preview(args)})")
        self.log(f"  <- {'ERROR ' if is_error else ''}{_preview(result)}")
        # Each result goes back as a "tool" message tied to the call's id.
        return {
            "role": "tool",
            "tool_call_id": call.id,
            "content": f"ERROR: {result}" if is_error else result,
        }


def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"))  # variables already in the environment win
    cfg = settings()
    agent = Agent()
    print(f"Foundations Agent - model {agent.model} at {cfg['base_url']}")
    print("Skills: " + ", ".join(t["function"]["name"] for t in TOOLS))
    print("Type a request, or 'quit' to exit.\n")
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue
        try:
            reply = agent.run(user_input)
        except openai.APIConnectionError:
            sys.exit(
                f"Can't reach a model server at {cfg['base_url']}.\n"
                "Is Ollama running? Start the Ollama app, or run `ollama serve` in another terminal."
            )
        except openai.NotFoundError:
            sys.exit(f"Model {agent.model!r} isn't available. Download it with:  ollama pull {agent.model}")
        except openai.AuthenticationError:
            sys.exit("The server rejected the API key: check LLM_API_KEY in .env.")
        except openai.APIStatusError as exc:
            print(f"agent> [Server error {exc.status_code}: {exc.message}]\n")
            continue
        print(f"agent> {reply}\n")


if __name__ == "__main__":
    main()
