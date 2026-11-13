"""Level 2: the skill registry and the tool definitions sent to the model."""

import pytest

import skills
from skills import TOOLS, run_skill, skill_names

EXPECTED = ["calculator", "get_current_time", "read_file", "write_file", "fetch_url"]


def test_exactly_five_skills():
    assert skill_names() == EXPECTED
    assert [t["function"]["name"] for t in TOOLS] == EXPECTED


@pytest.mark.parametrize("tool", TOOLS, ids=lambda t: t["function"]["name"])
def test_tool_definition_shape(tool):
    # OpenAI-style "function" tool - the format Ollama and most model servers accept.
    assert tool["type"] == "function"
    fn = tool["function"]
    assert set(fn) == {"name", "description", "parameters"}
    assert fn["description"].strip()
    schema = fn["parameters"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) <= set(schema["properties"])
    for prop in schema["properties"].values():
        # Plain string types only: small models handle them better than type unions.
        assert isinstance(prop["type"], str) and prop.get("description")


def test_optional_timezone():
    tool = next(t for t in TOOLS if t["function"]["name"] == "get_current_time")
    assert tool["function"]["parameters"]["required"] == []
    result, is_error = run_skill("get_current_time", {})
    assert not is_error and "UTC" in result


def test_unknown_skill_is_an_error_not_a_crash():
    result, is_error = run_skill("does_not_exist", {})
    assert is_error and "Unknown skill" in result


def test_bad_arguments_are_an_error_not_a_crash():
    result, is_error = run_skill("calculator", {"wrong_arg": "1"})
    assert is_error and "TypeError" in result


def test_non_object_arguments_are_an_error():
    result, is_error = run_skill("calculator", ["1+1"])
    assert is_error and "JSON object" in result


def test_exception_in_skill_is_reported(monkeypatch):
    def broken(expression):
        raise RuntimeError("kaboom")

    monkeypatch.setitem(skills._REGISTRY, "calculator", broken)
    assert run_skill("calculator", {"expression": "1"}) == ("RuntimeError: kaboom", True)


def test_duplicate_registration_rejected():
    with pytest.raises(ValueError):
        skills.skill("calculator", "dup", {})(lambda: "x")


def test_required_must_name_real_properties():
    with pytest.raises(ValueError):
        skills.skill("new_skill", "x", {"a": {"type": "string"}}, required=["b"])
