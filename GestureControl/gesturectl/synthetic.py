"""Procedural hand generator (forward kinematics) for tests and demos.

Builds anatomically-plausible 21-point hands for any pose in the
vocabulary, at any position, size, in-plane rotation and handedness. This
lets the whole engine - classifier, motion detectors, bindings - be tested
deterministically without a camera.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from .landmarks import Hand

# Palm layout in palm-length units, right hand, palm facing camera, fingers up (y-down).
_MCP = {"index": (-0.32, -0.95), "middle": (0.0, -1.0), "ring": (0.27, -0.94), "pinky": (0.50, -0.82)}
_SEG = {"index": (0.45, 0.27, 0.22), "middle": (0.50, 0.30, 0.23), "ring": (0.46, 0.28, 0.22),
        "pinky": (0.36, 0.22, 0.20)}
_THUMB_CMC = np.array([-0.22, -0.22, 0.0])
_THUMB_SEG = (0.38, 0.32, 0.27)

STRAIGHT = (4.0, 4.0, 3.0)
CURLED = (75.0, 100.0, 65.0)

POSE_FINGERS: Dict[str, Tuple[bool, bool, bool, bool, bool]] = {
    # thumb, index, middle, ring, pinky
    "fist": (False, False, False, False, False),
    "open_palm": (True, True, True, True, True),
    "four": (False, True, True, True, True),
    "point": (False, True, False, False, False),
    "peace": (False, True, True, False, False),
    "three": (False, True, True, True, False),
    "rock": (False, True, False, False, True),
    "shaka": (True, False, False, False, True),
    "L": (True, True, False, False, False),
    "thumbs_up": (True, False, False, False, False),
    "thumbs_down": (True, False, False, False, False),
}


def _rot_x(v: np.ndarray, deg: float) -> np.ndarray:
    """Bend towards the palm (rotate the y-z plane)."""
    a = math.radians(deg)
    y, z = v[1], v[2]
    return np.array([v[0], y * math.cos(a) + z * math.sin(a), -y * math.sin(a) + z * math.cos(a)])


def _finger(name: str, bends) -> np.ndarray:
    mcp = np.array([*_MCP[name], 0.0])
    d = mcp / np.linalg.norm(mcp)
    pts = [mcp]
    for seg, bend in zip(_SEG[name], bends):
        d = _rot_x(d, -bend)  # negative: tips curl towards the camera / palm
        pts.append(pts[-1] + d * seg)
    return np.array(pts)


def _thumb(extended: bool) -> np.ndarray:
    if extended:
        d = np.array([-0.85, -0.53, 0.0])
        bends = (5, 5, 5)
    else:
        d = np.array([-0.35, -0.9, -0.2])
        bends = (25, 70, 70)
    d /= np.linalg.norm(d)
    pts = [_THUMB_CMC]
    for i, (seg, bend) in enumerate(zip(_THUMB_SEG, bends)):
        if i > 0 and not extended:
            # fold across the palm: rotate direction around z towards +x
            a = math.radians(bend)
            d = np.array([d[0] * math.cos(a) - d[1] * math.sin(a),
                          d[0] * math.sin(a) + d[1] * math.cos(a), d[2] - 0.2])
            d /= np.linalg.norm(d)
        pts.append(pts[-1] + d * seg)
    return np.array(pts)


def _rotate_z(pts: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    r = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    return pts @ r.T


def hand_points(pose: str) -> np.ndarray:
    """(21, 3) points in palm-length units for `pose` (right hand, upright)."""
    pts = np.zeros((21, 3))
    if pose in ("pinch", "ok"):
        rest = CURLED if pose == "pinch" else STRAIGHT
        idx = _finger("index", (30.0, 45.0, 25.0))
        pts[5:9] = idx
        for i, name in enumerate(("middle", "ring", "pinky")):
            pts[9 + 4 * i:13 + 4 * i] = _finger(name, rest)
        tip = idx[-1] + np.array([0.03, 0.02, 0.0])
        cmc = _THUMB_CMC
        mid = (cmc + tip) / 2 + np.array([-0.15, 0.0, 0.0])
        pts[1:5] = [cmc, (cmc + mid) / 2 + [-0.05, 0, 0], (mid + tip) / 2 + [-0.03, 0, 0], tip]
        return pts
    fingers = POSE_FINGERS[pose]
    pts[1:5] = _thumb(fingers[0])
    for i, name in enumerate(("index", "middle", "ring", "pinky")):
        pts[5 + 4 * i:9 + 4 * i] = _finger(name, STRAIGHT if fingers[i + 1] else CURLED)
    if pose in ("thumbs_up", "thumbs_down"):
        v = pts[4, :2] - pts[2, :2]
        current = math.degrees(math.atan2(v[1], v[0]))
        target = -90.0 if pose == "thumbs_up" else 90.0
        pts = _rotate_z(pts, target - current)
    return pts


def make_hand(pose: str, center=(0.5, 0.5), size: float = 0.18, roll: float = 0.0,
              handedness: str = "Right", aspect: float = 16 / 9, noise: float = 0.0,
              rng: Optional[np.random.Generator] = None) -> Hand:
    """Create a `Hand` showing `pose`.

    center: palm centre in isotropic image units (x in [0, aspect], y in [0, 1]).
    size: palm length as a fraction of frame height.
    roll: extra in-plane rotation in degrees (clockwise on screen).
    """
    pts = hand_points(pose)
    if roll:
        pts = _rotate_z(pts, roll)
    if handedness == "Left":
        pts = pts * np.array([-1, 1, 1])
    if noise:
        rng = rng or np.random.default_rng(0)
        pts = pts + rng.normal(0, noise, pts.shape)
    world = pts * 0.09  # ~9 cm palm
    palm_c = pts[[0, 5, 9, 17], :2].mean(axis=0)
    iso = np.empty_like(pts)
    iso[:, 0] = center[0] + (pts[:, 0] - palm_c[0]) * size
    iso[:, 1] = center[1] + (pts[:, 1] - palm_c[1]) * size
    iso[:, 2] = pts[:, 2] * size
    norm = iso.copy()
    norm[:, 0] /= aspect
    norm[:, 2] /= aspect
    return Hand.from_normalized(norm, world, handedness, 0.99, aspect)
