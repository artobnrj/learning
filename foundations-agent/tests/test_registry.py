"""Level 2: the skill registry and the tool definitions sent to Claude."""

import pytest

import skills
from skills import TOOLS, run_skill, skill_names

EXPECTED = ["calculator", "get_current_time", "read_file", "write_file", "fetch_url"]


def test_exactly_five_skills():
    assert skill_names() == EXPECTED
    assert [t["name"] for t in TOOLS] == EXPECTED


@pytest.mark.parametrize("tool", TOOLS, ids=lambda t: t["name"])
def test_tool_definition_shape(tool):
    assert set(tool) == {"name", "description", "strict", "input_schema"}
    assert tool["description"].strip()
    assert tool["strict"] is True
    schema = tool["input_schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert sorted(schema["required"]) == sorted(schema["properties"])
    for prop in schema["properties"].values():
        assert "type" in prop and prop.get("description")


def test_unknown_skill_is_an_error_not_a_crash():
    result, is_error = run_skill("does_not_exist", {})
    assert is_error and "Unknown skill" in result


def test_bad_arguments_are_an_error_not_a_crash():
    result, is_error = run_skill("calculator", {"wrong_arg": "1"})
    assert is_error and "TypeError" in result


def test_exception_in_skill_is_reported(monkeypatch):
    def broken(expression):
        raise RuntimeError("kaboom")

    monkeypatch.setitem(skills._REGISTRY, "calculator", broken)
    assert run_skill("calculator", {"expression": "1"}) == ("RuntimeError: kaboom", True)


def test_duplicate_registration_rejected():
    with pytest.raises(ValueError):
        skills.skill("calculator", "dup", {})(lambda: "x")
