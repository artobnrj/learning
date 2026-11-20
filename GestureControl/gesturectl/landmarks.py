"""MediaPipe hand-landmark topology and the `Hand` container used everywhere.

MediaPipe returns 21 landmarks per hand:

    0 wrist
    1-4   thumb   (CMC, MCP, IP, TIP)
    5-8   index   (MCP, PIP, DIP, TIP)
    9-12  middle  (MCP, PIP, DIP, TIP)
    13-16 ring    (MCP, PIP, DIP, TIP)
    17-20 pinky   (MCP, PIP, DIP, TIP)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_JOINTS = {
    "thumb": (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}
FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass
class Hand:
    """One detected hand in one frame.

    Attributes:
        image: (21, 3) landmarks in *isotropic image units*: x is multiplied
            by the frame aspect ratio so that a unit in x equals a unit in y
            (1.0 == frame height). This keeps distances and angles honest on
            a 16:9 webcam. z is MediaPipe's relative depth (same scale as x).
        norm: (21, 3) raw MediaPipe normalised landmarks (x, y in [0, 1]).
            Used for mapping to screen coordinates.
        world: (21, 3) metric 3-D landmarks (metres, hand-centred) or None.
            Rotation/scale robust, so preferred for pose geometry.
        handedness: "Left" or "Right" from the *user's* point of view.
        score: handedness confidence.
    """

    image: np.ndarray
    norm: np.ndarray
    world: Optional[np.ndarray]
    handedness: str = "Right"
    score: float = 1.0
    aspect: float = 4 / 3
    meta: dict = field(default_factory=dict)

    @property
    def geometry(self) -> np.ndarray:
        """Best available 3-D point set for shape analysis."""
        return self.world if self.world is not None else self.image

    @classmethod
    def from_normalized(cls, norm, world=None, handedness="Right", score=1.0, aspect=4 / 3):
        norm = np.asarray(norm, dtype=np.float64).reshape(21, 3)
        image = norm.copy()
        image[:, 0] *= aspect
        image[:, 2] *= aspect
        w = None if world is None else np.asarray(world, dtype=np.float64).reshape(21, 3)
        return cls(image=image, norm=norm, world=w, handedness=handedness, score=score, aspect=aspect)

    def to_json(self) -> dict:
        return {
            "norm": np.round(self.norm, 5).tolist(),
            "world": None if self.world is None else np.round(self.world, 5).tolist(),
            "handedness": self.handedness,
            "score": round(float(self.score), 4),
            "aspect": self.aspect,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Hand":
        return cls.from_normalized(d["norm"], d.get("world"), d.get("handedness", "Right"),
                                   d.get("score", 1.0), d.get("aspect", 4 / 3))
