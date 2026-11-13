"""Level 3: the agent loop, driven by a fake model server - no real model needed."""

import json
from types import SimpleNamespace

import httpx2 as httpx  # the HTTP library the openai package uses
import openai
import pytest

import agent as agent_module
from agent import Agent
from skills import TOOLS


# --- helpers that imitate the openai client's response objects -----------------

def call(id, name, args):
    arguments = args if isinstance(args, str) else json.dumps(args)
    return SimpleNamespace(id=id, type="function", function=SimpleNamespace(name=name, arguments=arguments))


def reply(content=None, *tool_calls, finish_reason=None):
    finish_reason = finish_reason or ("tool_calls" if tool_calls else "stop")
    message = SimpleNamespace(content=content, tool_calls=list(tool_calls) or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


class FakeClient:
    """Returns scripted replies in order (repeating the last one) and records every request."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        # Snapshot the message list: the agent keeps appending to it after the call.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        item = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(item, Exception):
            raise item
        return item


def make_agent(*replies, **kwargs):
    client = FakeClient(*replies)
    logs = []
    return Agent(client=client, model="test-model", log=logs.append, **kwargs), client, logs


# --- loop behaviour -------------------------------------------------------------

def test_plain_answer_needs_one_call():
    agent, client, _ = make_agent(reply("Hello!"))
    assert agent.run("hi") == "Hello!"
    assert len(client.requests) == 1


def test_one_tool_round():
    agent, client, logs = make_agent(
        reply(None, call("c1", "calculator", {"expression": "2+2"})),
        reply("It's 4."),
    )
    assert agent.run("what is 2+2?") == "It's 4."
    assert len(client.requests) == 2

    sent = client.requests[1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool"]
    assert sent[2]["tool_calls"][0]["id"] == "c1"
    assert sent[3] == {"role": "tool", "tool_call_id": "c1", "content": "4"}
    assert any("calculator" in line for line in logs)


def test_several_tool_calls_in_one_reply():
    agent, client, _ = make_agent(
        reply(None, call("a", "calculator", {"expression": "1+1"}), call("b", "calculator", {"expression": "3*3"})),
        reply("2 and 9"),
    )
    agent.run("two sums")
    tool_msgs = [m for m in client.requests[1]["messages"] if m["role"] == "tool"]
    assert [(m["tool_call_id"], m["content"]) for m in tool_msgs] == [("a", "2"), ("b", "9")]


def test_tool_error_is_reported_and_loop_continues():
    agent, client, _ = make_agent(
        reply(None, call("c1", "calculator", {"expression": "1/0"})),
        reply("You can't divide by zero."),
    )
    assert agent.run("1/0?") == "You can't divide by zero."
    result = client.requests[1]["messages"][-1]
    assert result["content"].startswith("ERROR: ZeroDivisionError")


def test_invalid_json_arguments_become_an_error():
    agent, client, _ = make_agent(
        reply(None, call("c1", "calculator", "{not json")),
        reply("Sorry, let me try again."),
    )
    agent.run("broken")
    assert "not valid JSON" in client.requests[1]["messages"][-1]["content"]


def test_empty_arguments_mean_no_arguments():
    agent, client, _ = make_agent(reply(None, call("c1", "get_current_time", "")), reply("done"))
    agent.run("time?")
    assert "UTC" in client.requests[1]["messages"][-1]["content"]


def test_think_tags_are_removed():
    agent, _, _ = make_agent(reply("<think>let me reason...</think>\n\nThe answer is 4."))
    assert agent.run("2+2") == "The answer is 4."


def test_length_limit_is_reported():
    agent, _, _ = make_agent(reply("partial answer", finish_reason="length"))
    out = agent.run("write a novel")
    assert out.startswith("partial answer") and "length limit" in out


def test_content_filter_is_reported():
    agent, _, _ = make_agent(reply(None, finish_reason="content_filter"))
    assert "declined" in agent.run("something bad")


def test_runaway_loop_is_stopped():
    forever = reply(None, call("c", "calculator", {"expression": "1"}))
    agent, client, _ = make_agent(forever, max_steps=3)
    assert "Stopped after 3 steps" in agent.run("loop")
    assert len(client.requests) == 3


def test_conversation_memory_across_turns():
    agent, client, _ = make_agent(reply("Nice to meet you, Sam."), reply("Your name is Sam."))
    agent.run("My name is Sam.")
    agent.run("What's my name?")
    sent = client.requests[1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[1]["content"] == "My name is Sam."


def test_server_error_rolls_back_the_turn():
    request = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
    error = openai.APIConnectionError(request=request)
    agent, _, _ = make_agent(reply("first"), error)
    agent.run("first message")
    with pytest.raises(openai.APIConnectionError):
        agent.run("second message")
    assert [m["role"] for m in agent.messages] == ["user", "assistant"]


def test_request_shape():
    agent, client, _ = make_agent(reply("ok"))
    agent.run("hi")
    req = client.requests[0]
    assert req["model"] == "test-model"
    assert req["tools"] is TOOLS
    assert req["messages"][0] == {"role": "system", "content": agent_module.SYSTEM_PROMPT}


# --- settings ---------------------------------------------------------------------

def test_default_settings_point_at_ollama(monkeypatch):
    for var in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    agent = Agent()
    assert agent.model == "qwen3:8b"
    assert str(agent.client.base_url).rstrip("/") == "http://localhost:11434/v1"


def test_settings_come_from_environment(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("LLM_MODEL", "llama3.1:8b")
    agent = Agent()
    assert agent.model == "llama3.1:8b"
    assert str(agent.client.base_url).rstrip("/") == "http://localhost:1234/v1"


# --- the real openai client against a fake HTTP server -----------------------------

def test_wire_format_with_real_client():
    """Run a full tool round through the real `openai` library, with a fake
    server behind it, to check the exact JSON a real server would receive."""
    bodies = []

    def server(request):
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            message = {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "calculator", "arguments": '{"expression": "17/100*2340"}'},
            }]}
            finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": "17% of 2,340 is 397.8."}
            finish = "stop"
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        })

    client = openai.OpenAI(
        base_url="http://localhost:11434/v1", api_key="not-needed",
        http_client=httpx.Client(transport=httpx.MockTransport(server)),
    )
    agent = Agent(client=client, model="qwen3:8b", log=lambda _: None)
    assert agent.run("What is 17% of 2,340?") == "17% of 2,340 is 397.8."

    first, second = bodies
    assert first["model"] == "qwen3:8b"
    assert [t["function"]["name"] for t in first["tools"]] == [t["function"]["name"] for t in TOOLS]
    assert [m["role"] for m in second["messages"]] == ["system", "user", "assistant", "tool"]
    assert second["messages"][2]["tool_calls"][0]["function"]["name"] == "calculator"
    assert second["messages"][3] == {"role": "tool", "tool_call_id": "call_1", "content": "397.8"}
