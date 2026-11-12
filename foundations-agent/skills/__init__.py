"""Skill registry.

A "skill" is a plain Python function plus a JSON schema describing its inputs.
The @skill decorator records both, so the agent can:
  1. send every skill's definition to Claude (TOOLS), and
  2. run whichever skill Claude asks for (run_skill).

Adding a new skill = write one function, put @skill on it, import its module below.
"""

from typing import Any, Callable

_REGISTRY: dict[str, Callable[..., str]] = {}
TOOLS: list[dict[str, Any]] = []


def skill(name: str, description: str, properties: dict[str, Any]):
    """Register a function as a skill Claude can call.

    `properties` is the JSON-schema "properties" object for the inputs.
    Every property is required and no extra keys are allowed, and `strict`
    makes the API guarantee Claude's input matches this schema exactly.
    Optional inputs are expressed as nullable types (e.g. ["string", "null"]).
    """

    def decorator(fn: Callable[..., str]) -> Callable[..., str]:
        if name in _REGISTRY:
            raise ValueError(f"Skill {name!r} is already registered")
        _REGISTRY[name] = fn
        TOOLS.append(
            {
                "name": name,
                "description": description,
                "strict": True,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            }
        )
        return fn

    return decorator


def run_skill(name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
    """Run a skill by name. Returns (result_text, is_error) and never raises,
    so a buggy or misused skill can't crash the agent loop."""
    fn = _REGISTRY.get(name)
    if fn is None:
        return f"Unknown skill: {name!r}", True
    try:
        return str(fn(**tool_input)), False
    except Exception as exc:  # report every failure back to Claude
        return f"{type(exc).__name__}: {exc}", True


def skill_names() -> list[str]:
    return list(_REGISTRY)


# Importing the modules registers their skills (order = order shown to Claude).
from . import calculator, clock, files, web  # noqa: E402,F401
