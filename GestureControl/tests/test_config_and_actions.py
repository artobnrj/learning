import pytest

from gesturectl.actions import ActionRunner, HOTKEYS, known_actions
from gesturectl.actions.backend import DryRunBackend, parse_combo
from gesturectl.config import Bindings, ConfigError, deep_merge, load_config, parse_bindings


def test_default_config_is_valid():
    b = Bindings.from_config(load_config())
    assert {"desktop", "reels", "mouse"} <= set(b.modes)


def test_every_semantic_hotkey_exists_for_every_os():
    for name, table in HOTKEYS.items():
        assert "*" in table or {"windows", "darwin", "linux"} <= set(table), name


def test_parse_combo_aliases():
    assert parse_combo("Win+Shift+S") == ["cmd", "shift", "s"]
    assert parse_combo("ctrl + =") == ["ctrl", "="]
    assert parse_combo("volume_up") == ["media_volume_up"]


def test_deep_merge_overrides_scalars_and_replaces_lists():
    base = {"a": {"b": 1, "c": 2}, "l": [1, 2]}
    assert deep_merge(base, {"a": {"c": 3}, "l": [9]}) == {"a": {"b": 1, "c": 3}, "l": [9]}


@pytest.mark.parametrize("bad, msg", [
    ({"gesture": "fist", "action": "sleep"}, "exactly one trigger"),
    ({"gesture": "fist", "hold": 1, "swipe": "up", "action": "sleep"}, "exactly one trigger"),
    ({"gesture": "fisst", "hold": 1, "action": "sleep"}, "unknown gesture"),
    ({"gesture": "fist", "hold": 1, "action": "nap"}, "unknown action"),
    ({"gesture": "fist", "swipe": "sideways", "action": "sleep"}, "swipe must be"),
    ({"gesture": "pinch", "dial": True, "cw": "volume_up"}, "needs both"),
])
def test_bad_bindings_give_helpful_errors(bad, msg):
    with pytest.raises(ConfigError, match=msg):
        parse_bindings([bad], "test")


def test_run_actions_are_opt_in():
    errors = []
    runner = ActionRunner(DryRunBackend(), synchronous=True, on_error=errors.append)
    runner("run:echo hi")
    assert errors and "disabled" in errors[0]


def test_runner_dispatch():
    b = DryRunBackend(size=(1000, 800))
    r = ActionRunner(b, synchronous=True, os_name="windows")
    for a in ("show_desktop", "hotkey:ctrl+shift+t", "like", "scroll:3", "type:hi", "lock"):
        r(a)
    assert b.calls == [
        ("tap", "cmd+d", 1), ("tap", "ctrl+shift+t", 1), ("click", "left", 2, (500, 400)),
        ("scroll", 0, -3), ("type", "hi"), ("shell", "rundll32.exe user32.dll,LockWorkStation")]


def test_known_actions_cover_defaults():
    for a in ("sleep", "lock", "close_window", "reels_next", "like", "volume_up"):
        assert a in known_actions()
