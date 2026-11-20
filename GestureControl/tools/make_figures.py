"""Regenerate the figures in docs/images from synthetic hands (no camera needed).

    python tools/make_figures.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from gesturectl.hud import HUD, _text  # noqa: E402
from gesturectl.landmarks import FINGERTIPS, HAND_CONNECTIONS  # noqa: E402
from gesturectl.synthetic import make_hand  # noqa: E402
from helpers import Sim, engine  # noqa: E402

OUT = ROOT / "docs" / "images"
BG = (32, 26, 24)
COLORS = [(66, 135, 245), (80, 220, 120), (255, 200, 60), (230, 120, 255), (90, 200, 255)]


def draw_skeleton(img, hand, extended):
    h, w = img.shape[:2]
    pts = (hand.norm[:, :2] * [w, h]).astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), (235, 235, 235), 2, cv2.LINE_AA)
    for i, tip in enumerate(FINGERTIPS):
        cv2.circle(img, tuple(pts[tip]), 7 if extended[i] else 4, COLORS[i], -1 if extended[i] else 2,
                   cv2.LINE_AA)


def poses_grid():
    from gesturectl.features import FeatureExtractor
    from gesturectl.poses import classify

    poses = ["open_palm", "fist", "point", "peace", "three", "four", "L",
             "thumbs_up", "thumbs_down", "rock", "shaka", "pinch", "ok"]
    cols, cell = 7, 200
    rows = (len(poses) + cols - 1) // cols
    img = np.full((rows * cell, cols * cell, 3), BG, np.uint8)
    for k, pose in enumerate(poses):
        r, c = divmod(k, cols)
        tile = np.full((cell, cell, 3), BG, np.uint8)
        hand = make_hand(pose, center=(0.5, 0.52), size=0.30, aspect=1.0)
        f = FeatureExtractor()(hand)
        assert classify(f) == pose
        draw_skeleton(tile, hand, [f.extended[n] for n in ("thumb", "index", "middle", "ring", "pinky")])
        _text(tile, pose, (10, cell - 12), 0.55, (255, 191, 0), 1)
        img[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = tile
    cv2.imwrite(str(OUT / "poses.png"), img)


def hud_demo():
    eng, _ = engine("reels")
    sim = Sim(eng)
    # Some history for the event list: two swipes and a like.
    for direction in ("up", "up", "down"):
        dy = -0.3 if direction == "up" else 0.3
        sim.hold("open_palm", 0.3, center=(1.2, 0.55))
        sim.move("open_palm", 0.2, (1.2, 0.55), (1.2, 0.55 + dy))
        sim.none(0.6)
    sim.hold("thumbs_up", 0.6, center=(1.2, 0.5)).none(0.4)
    # Now: building up a hold-to-confirm "sleep" (2.5 s fist).
    sim.hold("fist", 1.6, center=(1.3, 0.5), size=0.22)
    frame = np.full((540, 960, 3), (48, 40, 38), np.uint8)
    cv2.imwrite(str(OUT / "hud.png"), HUD().draw(frame, eng))

    # Peace-sign joystick: hand lowered below its anchor -> continuous scroll.
    eng, _ = engine("desktop")
    sim = Sim(eng)
    sim.hold("peace", 0.3, center=(0.9, 0.35), size=0.2)
    sim.frames(0.5, lambda p: [make_hand("peace", center=(0.9, 0.35 + 0.16 * p), size=0.2)])
    frame = np.full((540, 960, 3), (48, 40, 38), np.uint8)
    cv2.imwrite(str(OUT / "hud_scroll.png"), HUD().draw(frame, eng))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    poses_grid()
    hud_demo()
    print("wrote", *sorted(p.name for p in OUT.glob("*.png")))
