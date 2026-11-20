"""Heads-up display drawn on the camera preview with OpenCV.

Shows the skeleton, current pose per hand, hold-to-confirm progress rings,
the scroll joystick, the mouse region, recent actions and a help overlay.
Feedback matters: a gesture system without it feels random.
"""
from __future__ import annotations

import time
from typing import List, Optional

import cv2
import numpy as np

from .actions import describe as describe_action
from .landmarks import FINGERTIPS, HAND_CONNECTIONS

# BGR
ACCENT = (255, 191, 0)
OK = (120, 220, 90)
WARN = (60, 170, 255)
DANGER = (80, 80, 240)
WHITE = (245, 245, 245)
DIM = (150, 150, 150)
FINGER_COLORS = [(66, 135, 245), (80, 220, 120), (255, 200, 60), (230, 120, 255), (90, 200, 255)]
DANGEROUS = {"sleep", "lock", "close_window", "quit_app", "close_tab"}


_ASCII = str.maketrans({"→": "->", "·": "|", "—": "-", "…": "..."})


def _text(img, s, org, scale=0.55, color=WHITE, thick=1, bg=True):
    """Draw text with a translucent backing box. OpenCV's Hershey fonts are
    ASCII-only, so typographic characters are transliterated first."""
    s = s.translate(_ASCII).encode("ascii", "replace").decode()
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), base = cv2.getTextSize(s, font, scale, thick)
    x, y = org
    if bg:
        overlay = img.copy()
        cv2.rectangle(overlay, (x - 5, y - h - 6), (x + w + 5, y + base + 3), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, s, (x, y), font, scale, color, thick, cv2.LINE_AA)
    return w


class HUD:
    def __init__(self):
        self.show_help = False
        self._fps = 0.0
        self._last = time.monotonic()

    def tick(self) -> float:
        now = time.monotonic()
        dt = now - self._last
        self._last = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps else 1.0 / dt
        return self._fps

    def draw(self, frame: np.ndarray, engine, latency_ms: Optional[float] = None) -> np.ndarray:
        img = frame
        h, w = img.shape[:2]

        if engine.continuous == "mouse" or any(b.trigger == "mouse" for b in engine.active_bindings()):
            x0, y0, x1, y1 = engine.mouse_region
            cv2.rectangle(img, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), DIM, 1)
            _text(img, "mouse area", (int(x0 * w) + 4, int(y0 * h) + 16), 0.4, DIM, bg=False)

        for slot in engine.visible:
            self._draw_hand(img, slot, primary=slot is engine.primary)

        for slots, frac, label in engine.progress:
            pts = [s.pixel_center(w, h) for s in slots]
            cx, cy = np.mean(pts, axis=0).astype(int)
            danger = any(a in label for a in DANGEROUS)
            color = DANGER if danger else ACCENT
            r = 46
            cv2.circle(img, (cx, cy), r, (40, 40, 40), 6, cv2.LINE_AA)
            cv2.ellipse(img, (cx, cy), (r, r), -90, 0, 360 * frac, color, 6, cv2.LINE_AA)
            _text(img, label, (cx - r, cy + r + 24), 0.5, color)

        p = engine.primary
        if engine.continuous == "scroll" and p is not None and engine.joy.anchor is not None:
            ax, ay = engine.joy.anchor
            ax, ay = int(ax / p.hand.aspect * w), int(ay * h)
            dz = int(engine.joy.deadzone * p.features.scale * h)
            cv2.line(img, (ax - 40, ay - dz), (ax + 40, ay - dz), DIM, 1)
            cv2.line(img, (ax - 40, ay + dz), (ax + 40, ay + dz), DIM, 1)
            cx, cy = p.pixel_center(w, h)
            cv2.arrowedLine(img, (ax, ay), (ax, cy), OK, 3, cv2.LINE_AA, tipLength=0.25)
            _text(img, f"scroll {engine.joy.offset:+.2f}", (ax + 50, ay), 0.5, OK)
        if engine.continuous == "dial" and p is not None:
            cx, cy = p.pixel_center(w, h)
            cv2.ellipse(img, (cx, cy), (60, 60), p.features.roll, -20, 20, ACCENT, 4, cv2.LINE_AA)
            _text(img, f"dial {engine.dial.total:+.0f} deg", (cx + 66, cy), 0.5, ACCENT)

        self._draw_status(img, engine, latency_ms)
        self._draw_events(img, engine)
        if self.show_help:
            self._draw_help(img, engine)
        return img

    def _draw_hand(self, img, slot, primary: bool) -> None:
        h, w = img.shape[:2]
        pts = (slot.hand.norm[:, :2] * [w, h]).astype(int)
        base = WHITE if primary else DIM
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), base, 2, cv2.LINE_AA)
        for i, tip in enumerate(FINGERTIPS):
            name = ("thumb", "index", "middle", "ring", "pinky")[i]
            ext = slot.features.extended[name]
            cv2.circle(img, tuple(pts[tip]), 8 if ext else 5, FINGER_COLORS[i], -1 if ext else 2,
                       cv2.LINE_AA)
        for j in range(21):
            if j not in FINGERTIPS:
                cv2.circle(img, tuple(pts[j]), 3, base, -1, cv2.LINE_AA)
        if slot.features.pinching:
            pp = ((pts[4] + pts[8]) / 2).astype(int)
            cv2.circle(img, tuple(pp), 14, OK, 2, cv2.LINE_AA)
        x, y = pts[:, 0].min(), pts[:, 1].min()
        label = f"{slot.key[0]}: {slot.pose}" + ("" if slot.raw == slot.pose else f" ({slot.raw})")
        _text(img, label, (int(x), max(20, int(y) - 12)), 0.6, ACCENT if primary else DIM, 2)

    def _draw_status(self, img, engine, latency_ms) -> None:
        h, w = img.shape[:2]
        state = ("PAUSED  (hold shaka to resume)", WARN) if engine.paused else ("ARMED", OK)
        x = 12
        x += _text(img, f"mode: {engine.mode.upper()}", (x, 28), 0.65, ACCENT, 2) + 18
        x += _text(img, state[0], (x, 28), 0.65, state[1], 2) + 18
        info = f"{self._fps:4.1f} fps" if self._fps else ""
        if latency_ms is not None:
            info += f"  {latency_ms:4.0f} ms"
        if info:
            _text(img, info, (x, 28), 0.5, DIM)
        _text(img, "[h] help  [m] mode  [p] pause  [q] quit", (12, h - 14), 0.45, DIM)

    def _draw_events(self, img, engine) -> None:
        h, w = img.shape[:2]
        now = engine.events[-1].t if engine.events else 0
        y = 64
        for ev in list(engine.events)[-5:][::-1]:
            age = now - ev.t
            color = DANGER if ev.action in DANGEROUS else (OK if age < 1.0 else DIM)
            s = f"{ev.trigger}  ->  {ev.action}"
            (tw, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            _text(img, s, (w - tw - 16, y), 0.5, color)
            y += 26

    def _draw_help(self, img, engine) -> None:
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
        _text(img, f"Gestures — mode '{engine.mode}'", (24, 60), 0.75, ACCENT, 2, bg=False)
        y = 96
        for b in engine.active_bindings():
            _text(img, b.label(), (28, y), 0.48, WHITE, bg=False)
            if b.actions and len(b.actions) == 1:
                _text(img, describe_action(b.actions[0]), (w // 2 + 40, y), 0.42, DIM, bg=False)
            y += 22
            if y > h - 30:
                break


def render_preview(frame: np.ndarray, engine, hud: Optional[HUD] = None) -> np.ndarray:
    """Render a static HUD image (used by `gesturectl annotate` and docs)."""
    return (hud or HUD()).draw(frame.copy(), engine)
