"""End-to-end scenarios: synthetic hands -> engine -> recorded OS calls."""
import numpy as np

from gesturectl.custom import CustomGestureModel, embed
from gesturectl.synthetic import make_hand
from helpers import Sim, engine


def flick(sim, pose, direction, center=(0.9, 0.55), dist=0.3, seconds=0.2, **kw):
    dx, dy = {"up": (0, -dist), "down": (0, dist), "left": (-dist, 0), "right": (dist, 0)}[direction]
    sim.hold(pose, 0.3, center=center, **kw)
    sim.move(pose, seconds, center, (center[0] + dx, center[1] + dy), **kw)
    sim.none(0.6)


def test_reels_swipe_up_and_down():
    eng, backend = engine("reels")
    sim = Sim(eng)
    flick(sim, "open_palm", "up")
    flick(sim, "open_palm", "down")
    assert sim.actions == ["reels_next", "reels_prev"]
    assert ("tap", "down", 1) in backend.calls and ("tap", "up", 1) in backend.calls


def test_swipe_return_stroke_does_not_double_fire():
    eng, _ = engine("reels")
    sim = Sim(eng)
    sim.hold("open_palm", 0.3, center=(0.9, 0.7))
    sim.move("open_palm", 0.2, (0.9, 0.7), (0.9, 0.4))
    sim.move("open_palm", 0.25, (0.9, 0.4), (0.9, 0.7))   # natural return
    assert sim.actions == ["reels_next"]


def test_fist_hold_puts_computer_to_sleep():
    eng, backend = engine("desktop")
    Sim(eng).hold("fist", 2.7)
    assert ("shell", "systemctl suspend") in backend.calls


def test_fist_released_early_does_nothing_and_progress_is_reported():
    eng, backend = engine("desktop")
    sim = Sim(eng).hold("fist", 1.5)
    assert eng.progress and 0.5 < eng.progress[0][1] < 0.7
    sim.none(0.5).hold("fist", 1.5)
    assert backend.calls == []


def test_hold_does_not_count_while_hand_is_moving():
    eng, _ = engine("desktop")
    sim = Sim(eng)
    # three fingers waved around for 2 s: no close_window
    sim.frames(2.0, lambda p: [make_hand("three", center=(0.9 + 0.25 * np.sin(p * 12), 0.5))])
    assert "close_window" not in sim.actions
    sim.hold("three", 1.0, center=(0.9, 0.5))
    assert sim.actions.count("close_window") == 1


def test_close_window_resolves_per_os():
    for os_name, combo in [("linux", "alt+f4"), ("windows", "alt+f4"), ("darwin", "cmd+w")]:
        eng, backend = engine("desktop", os_name=os_name)
        Sim(eng).hold("three", 1.0)
        assert ("tap", combo, 1) in backend.calls, os_name


def test_pause_blocks_everything_but_unpause():
    eng, backend = engine("desktop")
    sim = Sim(eng).hold("shaka", 1.2)
    assert eng.paused
    sim.none(0.3).hold("fist", 3.0).hold("three", 1.0)
    assert backend.calls == []
    sim.none(0.3).hold("shaka", 1.2)
    assert not eng.paused


def test_L_cycles_modes_once_per_hold():
    eng, _ = engine("desktop")
    sim = Sim(eng).hold("L", 3.0)
    assert eng.mode == "reels"
    sim.none(0.3).hold("L", 1.2)
    assert eng.mode == "mouse"


def test_rock_opens_reels_and_switches_mode():
    eng, backend = engine("desktop")
    Sim(eng).hold("rock", 1.2)
    assert ("open", "https://www.instagram.com/reels/") in backend.calls
    assert eng.mode == "reels"


def test_peace_joystick_scrolls_proportionally():
    eng, backend = engine("desktop")
    sim = Sim(eng).hold("peace", 0.3, center=(0.9, 0.5))
    sim.hold("peace", 1.0, center=(0.9, 0.5 + 0.18 * 0.9))  # hand lowered ~0.9 palm lengths
    down = sum(c[2] for c in backend.calls if c[0] == "scroll")
    assert down < -5  # pynput: negative dy == scroll down
    backend.calls.clear()
    sim.hold("peace", 1.0, center=(0.9, 0.5 + 0.18 * 0.1))  # back inside the dead zone
    assert not any(c[0] == "scroll" for c in backend.calls)


def test_pinch_dial_changes_volume():
    eng, backend = engine("desktop")
    sim = Sim(eng)
    sim.frames(1.0, lambda p: [make_hand("pinch", roll=p * 60)])
    assert sim.actions.count("volume_up") >= 1
    assert all(a != "volume_down" for a in sim.actions)
    keys = [c[1] for c in backend.calls if c[0] == "tap"]
    assert set(keys) == {"media_volume_up"}


def test_two_open_palms_show_desktop_and_suppress_single_hand():
    eng, backend = engine("desktop")
    sim = Sim(eng)
    sim.frames(1.0, lambda p: [make_hand("open_palm", center=(0.5, 0.5), handedness="Left"),
                               make_hand("open_palm", center=(1.3, 0.5), handedness="Right")])
    assert sim.actions == ["show_desktop"]


def test_two_hand_pinch_zoom():
    eng, backend = engine("desktop")
    sim = Sim(eng)
    sim.frames(1.0, lambda p: [
        make_hand("pinch", center=(0.75 - 0.25 * p, 0.5), handedness="Left"),
        make_hand("pinch", center=(1.0 + 0.25 * p, 0.5), handedness="Right")])
    assert sim.actions.count("zoom_in") >= 2 and "volume_up" not in sim.actions


def test_background_hands_are_ignored():
    eng, backend = engine("desktop")
    Sim(eng).hold("fist", 3.0, size=0.04)
    assert backend.calls == []


def test_closest_hand_controls():
    eng, _ = engine("desktop")
    sim = Sim(eng)
    sim.frames(1.0, lambda p: [make_hand("three", size=0.25, center=(0.5, 0.5)),
                               make_hand("fist", size=0.1, center=(1.4, 0.5), handedness="Left")])
    assert sim.actions == ["close_window"]


def test_mouse_mode_moves_cursor_and_pinch_drags():
    eng, backend = engine("mouse")
    sim = Sim(eng)
    sim.move("open_palm", 1.0, (0.55, 0.3), (1.2, 0.6))
    moves = [c for c in backend.calls if c[0] == "move"]
    assert len(moves) > 20 and moves[-1][1] > moves[0][1] and moves[-1][2] > moves[0][2]
    sim.hold("pinch", 0.4, center=(1.2, 0.6))
    assert ("press", "left") in backend.calls
    sim.hold("open_palm", 0.3, center=(1.2, 0.6))
    assert backend.calls[-1][0] == "move" and ("release", "left") in backend.calls


def test_mouse_clutch_freezes_cursor():
    eng, backend = engine("mouse")
    sim = Sim(eng).hold("open_palm", 0.5)
    sim.hold("fist", 0.15)  # smoother needs ~3 frames to commit to the clutch pose
    n = len(backend.calls)
    sim.move("fist", 0.5, (0.89, 0.5), (1.3, 0.7))
    assert not any(c[0] == "move" for c in backend.calls[n:])


def test_mouse_clutch_does_not_trigger_sleep():
    eng, backend = engine("mouse")
    Sim(eng).hold("fist", 3.5)
    assert not any(c[0] == "shell" for c in backend.calls)


def test_custom_gesture_binding():
    rng = np.random.default_rng(3)
    model = CustomGestureModel()
    model.add("spidey", np.array([embed(make_hand("rock", noise=0.02, rng=rng)) for _ in range(30)]))
    eng, backend = engine("desktop", custom=model, overrides={
        "modes.desktop": [{"gesture": "custom:spidey", "hold": 0.5, "action": "screenshot"}]})
    sim = Sim(eng).hold("rock", 0.8)
    assert sim.actions == ["screenshot"]
    assert ("tap", "print_screen", 1) in backend.calls


def test_action_cooldown():
    eng, backend = engine("desktop", overrides={
        "modes.desktop": [{"gesture": "thumbs_up", "tap": 0.4, "action": "play_pause"}]})
    sim = Sim(eng)
    for _ in range(2):
        sim.hold("thumbs_up", 0.15).hold("fist", 0.12)
    assert sim.actions.count("play_pause") == 1  # 2nd tap lands inside the 0.35 s cooldown
    sim.hold("fist", 0.5).hold("thumbs_up", 0.15).hold("fist", 0.12)
    assert sim.actions.count("play_pause") == 2


def test_repeat_trigger():
    eng, _ = engine("desktop", overrides={
        "modes.desktop": [{"gesture": "thumbs_up", "repeat": 0.25, "action": "volume_up"}]})
    sim = Sim(eng).hold("thumbs_up", 1.3)
    assert 4 <= sim.actions.count("volume_up") <= 6
