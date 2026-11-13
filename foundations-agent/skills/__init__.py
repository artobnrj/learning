"""Skill registry.

A "skill" is a plain Python function plus a JSON schema describing its inputs.
The @skill decorator records both, so the agent can:
  1. send every skill's definition to the model (TOOLS), and
  2. run whichever skill the model asks for (run_skill).

Adding a new skill = write one function, put @skill on it, import its module below.
"""

from typing import Any, Callable

_REGISTRY: dict[str, Callable[..., str]] = {}
TOOLS: list[dict[str, Any]] = []


def skill(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
):
    """Register a function as a skill the model can call.

    `properties` is the JSON-schema "properties" object for the inputs.
    `required` lists the inputs the model must always send (default: all of them);
    leave an input out of it to make it optional, and give the Python
    parameter a default value.

    TOOLS uses the OpenAI-style "function" format, the standard understood by
    Ollama, LM Studio, llama.cpp, vLLM and most hosted open-model APIs.
    """
    required = list(properties) if required is None else required
    unknown = set(required) - set(properties)
    if unknown:
        raise ValueError(f"required names unknown properties: {sorted(unknown)}")

    def decorator(fn: Callable[..., str]) -> Callable[..., str]:
        if name in _REGISTRY:
            raise ValueError(f"Skill {name!r} is already registered")
        _REGISTRY[name] = fn
        TOOLS.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
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
    if not isinstance(tool_input, dict):
        return f"Arguments must be a JSON object, got {type(tool_input).__name__}", True
    try:
        return str(fn(**tool_input)), False
    except Exception as exc:  # report every failure back to the model
        return f"{type(exc).__name__}: {exc}", True


def skill_names() -> list[str]:
    return list(_REGISTRY)


# Importing the modules registers their skills (order = order shown to the model).
from . import calculator, clock, files, web  # noqa: E402,F401
