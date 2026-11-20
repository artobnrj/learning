"""Geometric feature extraction from 21 hand landmarks.

Design principle: every feature is a *ratio* or an *angle*, never an
absolute distance, so it is invariant to how far the hand is from the
camera, to camera resolution, and (for angles on 3-D world landmarks) to
hand rotation.

Thresholds below were calibrated on real MediaPipe output (see
docs/TECHNICAL.md, "Calibration data"): extended fingers measure a joint
curl of ~40-55 deg, folded fingers >= 145 deg, so the hysteresis band sits
in the empty space between them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .filters import Hysteresis
from .landmarks import (FINGER_JOINTS, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP,
                        PINKY_MCP, THUMB_MCP, THUMB_TIP, WRIST, Hand)

# Finger is "extended" when total curl drops below ON, "folded" above OFF.
FINGER_CURL_ON, FINGER_CURL_OFF = 105.0, 125.0
THUMB_CURL_ON, THUMB_CURL_OFF = 68.0, 82.0  # real data: extended <= 38, folded >= 86
# Pinch = thumb tip to index tip, in palm lengths.
PINCH_ON, PINCH_OFF = 0.22, 0.32


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at b (degrees). 180 = perfectly straight."""
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return 180.0
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def finger_curl(g: np.ndarray, finger: str) -> float:
    """Sum of bend angles along a finger chain, starting at the wrist.

    0 deg = ruler straight; a tight fist is ~200-240 deg.
    """
    a, b, c, d = FINGER_JOINTS[finger]
    chain = (g[WRIST], g[a], g[b], g[c], g[d])
    return sum(180.0 - joint_angle(chain[i - 1], chain[i], chain[i + 1]) for i in range(1, 4))


def palm_length(g: np.ndarray) -> float:
    return float(np.linalg.norm(g[WRIST] - g[MIDDLE_MCP])) or 1e-6


@dataclass
class HandFeatures:
    curl: Dict[str, float]
    extended: Dict[str, bool]
    pinch: float                 # thumb-index tip distance / palm length
    pinch_middle: float          # thumb-middle tip distance / palm length
    pinching: bool
    pinching_middle: bool
    thumb_dir: float             # angle of thumb (MCP->TIP) in image, deg; -90 = straight up
    thumb_out: float             # thumb tip to index MCP / palm length
    index_reach: float           # wrist to index tip / palm length (fist vs pinch guard)
    palm_center: np.ndarray      # isotropic image coords (x, y)
    scale: float                 # palm length in image units (for motion normalisation)
    roll: float                  # in-plane hand rotation (deg) for the volume dial
    palm_facing: bool            # True when the palm (not the back) faces the camera
    pinch_point: np.ndarray = field(default_factory=lambda: np.zeros(2))

    @property
    def n_extended(self) -> int:
        return sum(self.extended.values())


class FeatureExtractor:
    """Stateful per-hand feature extractor (state = hysteresis memory)."""

    def __init__(self):
        self._finger = {f: (Hysteresis(THUMB_CURL_ON, THUMB_CURL_OFF, low_is_on=True) if f == "thumb"
                            else Hysteresis(FINGER_CURL_ON, FINGER_CURL_OFF, low_is_on=True))
                        for f in FINGER_JOINTS}
        self._pinch = Hysteresis(PINCH_ON, PINCH_OFF, low_is_on=True)
        self._pinch_mid = Hysteresis(PINCH_ON, PINCH_OFF, low_is_on=True)

    def __call__(self, hand: Hand) -> HandFeatures:
        g = hand.geometry
        im = hand.image
        palm = palm_length(g)

        curl = {f: finger_curl(g, f) for f in FINGER_JOINTS}
        extended = {f: self._finger[f].update(curl[f]) for f in FINGER_JOINTS}

        pinch = float(np.linalg.norm(g[THUMB_TIP] - g[INDEX_TIP]) / palm)
        pinch_mid = float(np.linalg.norm(g[THUMB_TIP] - g[MIDDLE_TIP]) / palm)
        index_reach = float(np.linalg.norm(g[INDEX_TIP] - g[WRIST]) / palm)
        # A curled index in a fist can sit close to the thumb; real pinches keep
        # the index finger reaching out, so gate on reach.
        pinching = self._pinch.update(pinch) and index_reach > 1.1
        pinching_mid = self._pinch_mid.update(pinch_mid) and not pinching

        tv = im[THUMB_TIP, :2] - im[THUMB_MCP, :2]
        thumb_dir = math.degrees(math.atan2(tv[1], tv[0]))  # y-down image: -90 is up
        thumb_out = float(np.linalg.norm(g[THUMB_TIP] - g[INDEX_MCP]) / palm)

        palm_pts = im[[WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP], :2]
        palm_center = palm_pts.mean(axis=0)
        scale = float(np.linalg.norm(im[WRIST, :2] - im[MIDDLE_MCP, :2])) or 1e-6
        rv = im[MIDDLE_MCP, :2] - im[WRIST, :2]
        roll = math.degrees(math.atan2(rv[1], rv[0]))

        # Palm normal via cross product; its sign flips between left/right hands.
        n = np.cross(im[INDEX_MCP] - im[WRIST], im[PINKY_MCP] - im[WRIST])
        palm_facing = bool((n[2] < 0) if hand.handedness == "Right" else (n[2] > 0))

        pinch_point = (im[THUMB_TIP, :2] + im[INDEX_TIP, :2]) / 2
        return HandFeatures(curl, extended, pinch, pinch_mid, pinching, pinching_mid,
                            thumb_dir, thumb_out, index_reach, palm_center, scale, roll,
                            palm_facing, pinch_point)
