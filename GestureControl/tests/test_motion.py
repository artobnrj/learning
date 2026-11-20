import numpy as np

from gesturectl.motion import Dial, Joystick, PinchZoom, SwipeDetector

SCALE = 0.15


def run_swipe(det, start, end, seconds, t0=0.0, pose="open_palm"):
    n = int(seconds * 30)
    events = []
    for i in range(n + 1):
        p = i / n
        c = np.array(start) + (np.array(end) - np.array(start)) * p
        ev = det.update(t0 + i / 30, c, SCALE, pose)
        if ev:
            events.append(ev)
    return events, t0 + seconds


def test_fast_flick_up_is_a_swipe():
    ev, _ = run_swipe(SwipeDetector(), (0.8, 0.7), (0.8, 0.4), 0.2)
    assert [e.direction for e in ev] == ["up"]
    assert ev[0].pose == "open_palm"


def test_slow_drift_is_not_a_swipe():
    ev, _ = run_swipe(SwipeDetector(), (0.8, 0.7), (0.8, 0.4), 1.5)
    assert ev == []


def test_diagonal_is_ignored():
    ev, _ = run_swipe(SwipeDetector(), (0.6, 0.7), (0.9, 0.4), 0.2)
    assert ev == []


def test_return_stroke_is_suppressed_but_a_new_swipe_is_not():
    det = SwipeDetector()
    up, t = run_swipe(det, (0.8, 0.7), (0.8, 0.4), 0.2)
    back, t = run_swipe(det, (0.8, 0.4), (0.8, 0.7), 0.2, t0=t + 0.05)
    assert [e.direction for e in up] == ["up"] and back == []
    again, _ = run_swipe(det, (0.8, 0.7), (0.8, 0.4), 0.2, t0=t + 1.0)
    assert [e.direction for e in again] == ["up"]


def test_swipe_distance_is_relative_to_hand_size():
    # Same pixel motion: a big (close) hand travels fewer palm lengths.
    det = SwipeDetector()
    events = []
    for i in range(7):
        ev = det.update(i / 30, np.array([0.8, 0.7 - 0.05 * i]), 0.4, "open_palm")
        events += [ev] if ev else []
    assert events == []


def test_joystick_deadzone_and_proportional_rate():
    j = Joystick(deadzone=0.25, gain=14, exponent=1.6)
    j.update(0.0, np.array([0.5, 0.5]), SCALE)
    small = sum(j.update(i / 30, np.array([0.5, 0.5 + 0.2 * SCALE]), SCALE) for i in range(1, 31))
    assert small == 0
    j.reset()
    j.update(0.0, np.array([0.5, 0.5]), SCALE)
    near = sum(j.update(i / 30, np.array([0.5, 0.5 + 0.6 * SCALE]), SCALE) for i in range(1, 31))
    j.reset()
    j.update(0.0, np.array([0.5, 0.5]), SCALE)
    far = sum(j.update(i / 30, np.array([0.5, 0.5 + 1.2 * SCALE]), SCALE) for i in range(1, 31))
    up = Joystick()
    up.update(0.0, np.array([0.5, 0.5]), SCALE)
    neg = sum(up.update(i / 30, np.array([0.5, 0.5 - 1.2 * SCALE]), SCALE) for i in range(1, 31))
    assert 0 < near < far and neg == -far


def test_dial_counts_steps_and_survives_wraparound():
    d = Dial(step_deg=18)
    angles = np.arange(170, 170 + 95, 5)  # crosses +180 -> -180
    steps = sum(d.update(((a + 180) % 360) - 180) for a in angles)
    assert steps == 5
    d.reset()
    assert sum(d.update(a) for a in np.arange(0, -60, -5)) == -3


def test_dial_ignores_tracking_glitch():
    d = Dial()
    d.update(0)
    assert d.update(120) == 0


def test_pinch_zoom_steps():
    z = PinchZoom(step_ratio=0.18)
    z.update([0.4, 0.5], [0.6, 0.5], SCALE)
    out = [z.update([0.4 - 0.01 * i, 0.5], [0.6 + 0.01 * i, 0.5], SCALE) for i in range(1, 11)]
    assert sum(out) >= 2 and min(out) == 0
