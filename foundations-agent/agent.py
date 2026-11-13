"""Foundations Agent: an LLM in a loop that can call skills (tools).

The whole idea of an "agent" fits in one function, `Agent.run`:

    send the conversation + tool list to Claude
    while Claude asks to use tools:
        run each tool, send all the results back
    return Claude's final answer

Run it:  python agent.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import anthropic
from dotenv import load_dotenv

from skills import TOOLS, run_skill

MODEL = "claude-opus-5-5"
EFFORT = "medium"  # low | medium | high | xhigh | max - trades depth for speed/cost
MAX_TOKENS = 16000
MAX_STEPS = 10  # safety net: stop if Claude keeps calling tools forever

SYSTEM_PROMPT = """You are Foundations Agent, a small helpful assistant with five skills:
calculator, get_current_time, read_file, write_file and fetch_url.
Use a skill whenever it gives a more reliable answer than guessing - always use the
calculator for arithmetic and get_current_time for anything about "now".
File paths are relative to your workspace folder; you cannot access files outside it.
If a skill returns an error, explain it plainly or try a corrected call.
Keep answers short and direct."""


def _preview(value: Any, limit: int = 120) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


class Agent:
    def __init__(
        self,
        client: Any | None = None,
        log: Callable[[str], None] = print,
        max_steps: int = MAX_STEPS,
    ):
        # A bare Anthropic() reads ANTHROPIC_API_KEY (or an `ant auth login` profile).
        self.client = client if client is not None else anthropic.Anthropic()
        self.log = log
        self.max_steps = max_steps
        self.messages: list[dict[str, Any]] = []  # the whole conversation = the agent's memory

    def _call_claude(self):
        return self.client.beta.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=self.messages,
            output_config={"effort": EFFORT},
            # If a safety classifier declines, the API retries on a fallback model in the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
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
            response = self._call_claude()
            # Keep Claude's turn exactly as returned (text, tool calls, thinking blocks).
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    self.log(f"  -> {block.name}({_preview(block.input)})")
                    result, is_error = run_skill(block.name, block.input)
                    self.log(f"  <- {'ERROR ' if is_error else ''}{_preview(result)}")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                            "is_error": is_error,
                        }
                    )
                # All results go back together in ONE user message.
                self.messages.append({"role": "user", "content": tool_results})
                continue

            if response.stop_reason == "pause_turn":
                continue  # the server paused a long turn; resend to let it continue

            text = "".join(b.text for b in response.content if b.type == "text").strip()
            if response.stop_reason == "refusal":
                return "[Claude declined this request.]" + (f" {text}" if text else "")
            if response.stop_reason == "max_tokens":
                return (text + "\n" if text else "") + "[Reply cut off: hit the max_tokens limit.]"
            return text  # end_turn (or stop_sequence)

        return f"[Stopped after {self.max_steps} steps without a final answer.]"


def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"))  # variables already in the environment win
    agent = Agent()
    print("Foundations Agent - skills: " + ", ".join(t["name"] for t in TOOLS))
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
        except anthropic.AuthenticationError:
            sys.exit("Authentication failed: set ANTHROPIC_API_KEY (or run `ant auth login`).")
        except anthropic.RateLimitError:
            print("agent> [Rate limited - wait a moment and try again.]\n")
            continue
        except anthropic.APIStatusError as exc:
            print(f"agent> [API error {exc.status_code}: {exc.message}]\n")
            continue
        except anthropic.APIConnectionError:
            print("agent> [Could not reach the Anthropic API - check your network.]\n")
            continue
        print(f"agent> {reply}\n")


if __name__ == "__main__":
    main()
