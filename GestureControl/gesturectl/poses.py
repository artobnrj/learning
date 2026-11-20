"""Static pose classification: features -> one of a fixed vocabulary.

Rule-based on purpose: it needs zero training data, is explainable, and is
easy to extend. User-trained poses (see custom.py) are layered on top.
"""
from __future__ import annotations

from collections import Counter, deque
from typing import Deque, Optional

from .features import HandFeatures

POSES = {
    "fist": "All fingers folded",
    "open_palm": "All five fingers extended",
    "four": "Four fingers extended, thumb tucked",
    "point": "Index finger only",
    "peace": "Index + middle (V sign)",
    "three": "Index + middle + ring",
    "thumbs_up": "Thumb up, other fingers folded",
    "thumbs_down": "Thumb down, other fingers folded",
    "rock": "Index + pinky (🤘)",
    "shaka": "Thumb + pinky (🤙 call-me / hang loose)",
    "L": "Thumb + index at a right angle",
    "pinch": "Thumb tip touching index tip",
    "ok": "Pinch with middle, ring, pinky extended (👌)",
    "none": "Unrecognised",
}


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def classify(f: HandFeatures) -> str:
    e = f.extended
    t, i, m, r, p = e["thumb"], e["index"], e["middle"], e["ring"], e["pinky"]

    if f.pinching:
        return "ok" if (m and r and p) else "pinch"

    if not (i or m or r or p):
        # Thumb must clearly stick out of the fist to count as a thumbs gesture.
        if t and f.thumb_out > 0.6:
            if _angle_diff(f.thumb_dir, -90) < 50:
                return "thumbs_up"
            if _angle_diff(f.thumb_dir, 90) < 50:
                return "thumbs_down"
        return "fist"

    if i and m and r and p:
        return "open_palm" if t else "four"
    if i and not m and not r and not p:
        return "L" if (t and f.thumb_out > 0.6) else "point"
    if i and m and not r and not p:
        return "peace"
    if i and m and r and not p:
        return "three"
    if i and p and not m and not r:
        return "rock"
    if p and t and not i and not m and not r:
        return "shaka"
    return "none"


class PoseSmoother:
    """Temporal majority vote with stickiness.

    A raw per-frame label flickers when a finger sits near a threshold. We
    keep the last `window` labels and only switch the reported pose when a
    new label owns at least `quorum` of them.
    """

    def __init__(self, window: int = 5, quorum: int = 3):
        self.window, self.quorum = window, quorum
        self._hist: Deque[str] = deque(maxlen=window)
        self.stable: str = "none"

    def update(self, label: str) -> str:
        self._hist.append(label)
        best, count = Counter(self._hist).most_common(1)[0]
        if count >= self.quorum:
            self.stable = best
        return self.stable

    def reset(self) -> None:
        self._hist.clear()
        self.stable = "none"


def describe(pose: Optional[str]) -> str:
    return POSES.get(pose or "none", pose or "")
