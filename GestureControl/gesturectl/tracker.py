"""Hand landmark detection via the MediaPipe Tasks `HandLandmarker`.

The model is a two-stage pipeline: a palm detector (SSD-style, runs only
when tracking is lost) followed by a landmark regressor that outputs 21
3-D keypoints, handedness and world coordinates. In VIDEO mode MediaPipe
re-uses the previous frame's landmarks as the next region of interest, so
the expensive detector rarely runs and we get real-time speeds on a CPU.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import List, Optional

import numpy as np

from .landmarks import Hand

log = logging.getLogger(__name__)

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
CACHE_DIR = Path(os.environ.get("GESTURECTL_CACHE", Path.home() / ".cache" / "gesturectl"))


def ensure_model(path: Optional[str] = None) -> Path:
    """Return a local path to hand_landmarker.task, downloading it once if needed."""
    if path:
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"model not found: {p}")
        return p
    env = os.environ.get("GESTURECTL_MODEL")
    if env:
        return ensure_model(env)
    target = CACHE_DIR / "hand_landmarker.task"
    if target.exists() and target.stat().st_size > 1_000_000:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading hand landmark model (~7.5 MB) -> %s", target)
    print(f"[gesturectl] downloading hand landmark model -> {target}")
    with urllib.request.urlopen(MODEL_URL, timeout=60) as r, \
            tempfile.NamedTemporaryFile(delete=False, dir=target.parent) as tmp:
        shutil.copyfileobj(r, tmp)
    os.replace(tmp.name, target)  # atomic: never leave a half-written model behind
    return target


class HandTracker:
    """Thin wrapper turning BGR frames into a list of `Hand` objects."""

    def __init__(self, model_path: Optional[str] = None, max_hands: int = 2,
                 min_detection_confidence: float = 0.6, min_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5, mode: str = "video",
                 mirrored_input: bool = True):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self._mp = mp
        self.mode = mode
        # MediaPipe labels handedness assuming a mirrored (selfie) image. If
        # the caller does not mirror, we swap labels so "Right" always means
        # the user's right hand.
        self.mirrored_input = mirrored_input
        running_mode = vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=running_mode,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, t: float = 0.0) -> List[Hand]:
        import cv2

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self.mode == "video":
            ts = int(t * 1000)
            if ts <= self._last_ts:  # VIDEO mode demands strictly increasing timestamps
                ts = self._last_ts + 1
            self._last_ts = ts
            result = self._landmarker.detect_for_video(image, ts)
        else:
            result = self._landmarker.detect(image)

        hands: List[Hand] = []
        for i, lms in enumerate(result.hand_landmarks):
            norm = np.array([[p.x, p.y, p.z] for p in lms])
            world = None
            if result.hand_world_landmarks and i < len(result.hand_world_landmarks):
                world = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[i]])
            label, score = "Right", 1.0
            if result.handedness and i < len(result.handedness):
                label = result.handedness[i][0].category_name
                score = result.handedness[i][0].score
            if not self.mirrored_input:
                label = "Left" if label == "Right" else "Right"
            hands.append(Hand.from_normalized(norm, world, label, score, aspect=w / h))
        return hands

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
