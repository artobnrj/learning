"""Level 3: the agent loop, driven by a fake Claude client - no API calls."""

from types import SimpleNamespace

import anthropic
import httpx2
import pytest

import agent as agent_module
from agent import Agent
from skills import TOOLS


# --- tiny helpers that imitate SDK response objects ---------------------------

def text(t):
    return SimpleNamespace(type="text", text=t)


def tool_use(id, name, input):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def response(stop_reason, *content):
    return SimpleNamespace(stop_reason=stop_reason, content=list(content))


class FakeClient:
    """Returns scripted responses in order and records every request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        # Snapshot the message list: the agent keeps appending to it after the call.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item


def make_agent(*responses, **kwargs):
    client = FakeClient(*responses)
    logs = []
    return Agent(client=client, log=logs.append, **kwargs), client, logs


# --- tests --------------------------------------------------------------------

def test_plain_answer_needs_one_call():
    agent, client, _ = make_agent(response("end_turn", text("Hello!")))
    assert agent.run("hi") == "Hello!"
    assert len(client.requests) == 1


def test_one_tool_round():
    agent, client, logs = make_agent(
        response("tool_use", text("Let me calculate."), tool_use("t1", "calculator", {"expression": "2+2"})),
        response("end_turn", text("It's 4.")),
    )
    assert agent.run("what is 2+2?") == "It's 4."
    assert len(client.requests) == 2

    second = client.requests[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert second[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "4", "is_error": False}
    ]
    assert any("calculator" in line for line in logs)


def test_parallel_tools_return_in_one_message_in_order():
    agent, client, _ = make_agent(
        response(
            "tool_use",
            tool_use("a", "calculator", {"expression": "1+1"}),
            tool_use("b", "calculator", {"expression": "3*3"}),
        ),
        response("end_turn", text("2 and 9")),
    )
    agent.run("two sums")
    results = client.requests[1]["messages"][-1]
    assert results["role"] == "user"
    assert [(r["tool_use_id"], r["content"]) for r in results["content"]] == [("a", "2"), ("b", "9")]


def test_tool_error_is_flagged_and_loop_continues():
    agent, client, _ = make_agent(
        response("tool_use", tool_use("t1", "calculator", {"expression": "1/0"})),
        response("end_turn", text("You can't divide by zero.")),
    )
    assert agent.run("1/0?") == "You can't divide by zero."
    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True and "ZeroDivisionError" in result["content"]


def test_pause_turn_resends_without_tool_results():
    agent, client, _ = make_agent(
        response("pause_turn", text("working...")),
        response("end_turn", text("done")),
    )
    assert agent.run("long task") == "done"
    second = client.requests[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant"]


def test_refusal_is_reported():
    agent, _, _ = make_agent(response("refusal"))
    assert "declined" in agent.run("something bad")


def test_max_tokens_is_reported():
    agent, _, _ = make_agent(response("max_tokens", text("partial answer")))
    reply = agent.run("write a novel")
    assert reply.startswith("partial answer") and "max_tokens" in reply


def test_runaway_loop_is_stopped():
    forever = response("tool_use", tool_use("t", "calculator", {"expression": "1"}))
    agent, client, _ = make_agent(forever, max_steps=3)
    assert "Stopped after 3 steps" in agent.run("loop")
    assert len(client.requests) == 3


def test_conversation_memory_across_turns():
    agent, client, _ = make_agent(
        response("end_turn", text("Nice to meet you, Sam.")),
        response("end_turn", text("Your name is Sam.")),
    )
    agent.run("My name is Sam.")
    agent.run("What's my name?")
    second = client.requests[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert second[0]["content"] == "My name is Sam."


def test_api_error_rolls_back_the_turn():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.APIConnectionError(request=request)
    agent, _, _ = make_agent(response("end_turn", text("first")), error)
    agent.run("first message")
    with pytest.raises(anthropic.APIConnectionError):
        agent.run("second message")
    assert [m["role"] for m in agent.messages] == ["user", "assistant"]


def test_request_shape():
    agent, client, _ = make_agent(response("end_turn", text("ok")))
    agent.run("hi")
    req = client.requests[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["tools"] is TOOLS
    assert req["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in req["betas"]
    assert req["output_config"] == {"effort": agent_module.EFFORT}
    assert req["system"] == agent_module.SYSTEM_PROMPT
