"""Dynamic gesture detectors: swipes, joystick scrolling, dials and two-hand zoom.

All distances are measured in *palm lengths* (hand size in the image), so a
swipe means the same thing whether you sit 40 cm or 1.5 m from the camera.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np


@dataclass
class SwipeEvent:
    direction: str  # "up" | "down" | "left" | "right"
    pose: str       # dominant pose during the stroke
    distance: float  # palm lengths travelled
    speed: float     # palm lengths / second


OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


class SwipeDetector:
    """Detects fast directional flicks of the hand.

    Algorithm (per frame):
      1. Keep a ~300 ms ring buffer of (t, palm centre, palm scale, pose).
      2. Displacement = newest - oldest sample, divided by mean palm scale.
      3. Fire when distance > `min_distance`, mean speed > `min_speed`, and
         the motion is clearly along one axis (`axis_ratio`).
      4. Then enter a refractory period, and block the *opposite* direction
         a little longer: after flicking up, your hand naturally travels back
         down - that return stroke must not count as a swipe down.
    """

    def __init__(self, window: float = 0.30, min_distance: float = 1.1, min_speed: float = 3.5,
                 axis_ratio: float = 1.8, refractory: float = 0.45, return_block: float = 0.9):
        self.window = window
        self.min_distance = min_distance
        self.min_speed = min_speed
        self.axis_ratio = axis_ratio
        self.refractory = refractory
        self.return_block = return_block
        self._buf: Deque[Tuple[float, np.ndarray, float, str]] = deque()
        self._quiet_until = 0.0
        self._blocked: Optional[str] = None
        self._blocked_until = 0.0

    def reset(self) -> None:
        self._buf.clear()

    def update(self, t: float, center: np.ndarray, scale: float, pose: str) -> Optional[SwipeEvent]:
        self._buf.append((t, np.asarray(center, dtype=float)[:2].copy(), scale, pose))
        while self._buf and t - self._buf[0][0] > self.window:
            self._buf.popleft()
        if t < self._quiet_until or len(self._buf) < 3:
            return None

        t0, c0, _, _ = self._buf[0]
        dt = t - t0
        if dt < 0.06:
            return None
        mean_scale = float(np.mean([s for _, _, s, _ in self._buf])) or 1e-6
        d = (self._buf[-1][1] - c0) / mean_scale
        dx, dy = float(d[0]), float(d[1])
        dist = math.hypot(dx, dy)
        speed = dist / dt
        if dist < self.min_distance or speed < self.min_speed:
            return None
        if abs(dx) > self.axis_ratio * abs(dy):
            direction = "right" if dx > 0 else "left"
        elif abs(dy) > self.axis_ratio * abs(dx):
            direction = "down" if dy > 0 else "up"  # image y grows downwards
        else:
            return None  # diagonal: ambiguous, ignore
        if self._blocked == direction and t < self._blocked_until:
            return None

        pose = Counter(p for *_, p in self._buf).most_common(1)[0][0]
        self._quiet_until = t + self.refractory
        self._blocked, self._blocked_until = OPPOSITE[direction], t + self.return_block
        self._buf.clear()
        return SwipeEvent(direction, pose, dist, speed)


class Joystick:
    """Displacement-from-anchor -> continuous rate (scrolling).

    When the pose starts, the current hand height becomes the anchor. Moving
    the hand above/below the anchor scrolls at a speed that grows with the
    offset (power curve), with a dead zone so small tremors do nothing.
    Fractional ticks accumulate so slow scrolling is smooth, not stuttery.
    """

    def __init__(self, deadzone: float = 0.25, gain: float = 14.0, exponent: float = 1.6,
                 max_rate: float = 40.0, axis: str = "vertical"):
        self.deadzone, self.gain, self.exponent, self.max_rate = deadzone, gain, exponent, max_rate
        self.axis = 1 if axis == "vertical" else 0
        self.anchor: Optional[np.ndarray] = None
        self._acc = 0.0
        self._t: Optional[float] = None
        self.offset = 0.0

    def reset(self) -> None:
        self.anchor, self._acc, self._t, self.offset = None, 0.0, None, 0.0

    def update(self, t: float, center: np.ndarray, scale: float) -> int:
        """Return signed integer ticks to emit now (+ = down/right)."""
        c = np.asarray(center, dtype=float)
        if self.anchor is None:
            self.anchor, self._t = c.copy(), t
            return 0
        dt = min(t - self._t, 0.1)
        self._t = t
        off = float(c[self.axis] - self.anchor[self.axis]) / (scale or 1e-6)
        self.offset = off
        mag = abs(off) - self.deadzone
        if mag <= 0:
            self._acc = 0.0
            return 0
        rate = min(self.gain * mag ** self.exponent, self.max_rate)  # ticks / second
        self._acc += math.copysign(rate * dt, off)
        ticks = int(self._acc)
        self._acc -= ticks
        return ticks


class Dial:
    """Turns in-plane hand rotation into discrete steps (a virtual knob).

    Uses the unwrapped angle so crossing +-180 deg does not cause a jump.
    """

    def __init__(self, step_deg: float = 18.0):
        self.step = step_deg
        self._last: Optional[float] = None
        self._acc = 0.0
        self.total = 0.0

    def reset(self) -> None:
        self._last, self._acc, self.total = None, 0.0, 0.0

    def update(self, angle_deg: float) -> int:
        """Return signed steps; + = clockwise as seen by the user."""
        if self._last is None:
            self._last = angle_deg
            return 0
        delta = (angle_deg - self._last + 180) % 360 - 180
        self._last = angle_deg
        if abs(delta) > 60:  # tracking glitch, not a real rotation
            return 0
        self._acc += delta
        self.total += delta
        steps = int(self._acc / self.step)
        self._acc -= steps * self.step
        return steps


class PinchZoom:
    """Two-hand pinch-and-spread -> zoom steps (like a touchpad, in the air)."""

    def __init__(self, step_ratio: float = 0.18):
        self.step_ratio = step_ratio
        self._ref: Optional[float] = None
        self.ratio = 1.0

    def reset(self) -> None:
        self._ref, self.ratio = None, 1.0

    def update(self, p1: np.ndarray, p2: np.ndarray, scale: float) -> int:
        d = float(np.linalg.norm(np.asarray(p1) - np.asarray(p2))) / (scale or 1e-6)
        if self._ref is None:
            self._ref = d
            return 0
        self.ratio = d / self._ref
        if self.ratio > 1 + self.step_ratio:
            self._ref = d
            return 1
        if self.ratio < 1 / (1 + self.step_ratio):
            self._ref = d
            return -1
        return 0
