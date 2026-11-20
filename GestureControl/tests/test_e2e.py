"""Full-stack test: video -> camera thread -> MediaPipe -> engine -> pynput -> X server.

A short video is assembled from real hand photos, played through
`gesturectl run --camera <video>`, and a pynput keyboard *listener* checks
that the synthesised key events actually reach the display server.

Needs an X display (run under `xvfb-run -a pytest`); skipped otherwise.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from test_real_images import _fetch

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="needs an X display (xvfb-run)")


def _video(path):
    import cv2

    def frame(name):
        img = np.full((540, 960, 3), 235, np.uint8)
        if name:
            im = cv2.imread(str(_fetch(name + ".jpg")))
            h, w = im.shape[:2]
            im = cv2.resize(im, (int(w * 500 / h), 500))
            x = (960 - im.shape[1]) // 2
            img[20:520, x:x + im.shape[1]] = im
        return img

    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (960, 540))
    for name, secs in [(None, 0.5), ("thumbs_up", 1.2), (None, 0.6), ("pointing_up", 1.4), (None, 0.5)]:
        f = frame(name)
        for _ in range(int(secs * 30)):
            vw.write(f)
    vw.release()


def test_video_to_real_keystrokes(tmp_path):
    pytest.importorskip("mediapipe")
    from pynput import keyboard

    video = tmp_path / "demo.avi"
    _video(video)
    received = []
    listener = keyboard.Listener(on_press=lambda k: received.append(str(k)))
    listener.start()
    time.sleep(0.5)
    proc = subprocess.run([sys.executable, "-m", "gesturectl", "run", "--camera", str(video), "--no-preview"],
                          cwd=ROOT, capture_output=True, text=True, timeout=120,
                          env={**os.environ, "GLOG_minloglevel": "2"})
    time.sleep(0.5)
    listener.stop()
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "thumbs_up hold 0.5s -> play_pause" in proc.stdout
    assert "point hold 0.9s -> switch_app" in proc.stdout
    assert "Key.media_play_pause" in received          # thumbs-up held
    assert ["Key.alt", "Key.tab"] == [k for k in received if k in ("Key.alt", "Key.tab")]  # alt+tab
