"""Low-latency threaded webcam capture.

OpenCV's `read()` returns the *oldest* buffered frame. If inference is a
little slower than the camera, a queue of stale frames builds up and the
system feels laggy. A background thread that always overwrites a single
"latest frame" slot fixes that: the vision loop always sees the newest
image, and slow frames are dropped instead of delayed.
"""
from __future__ import annotations

import platform
import threading
import time
from typing import Optional, Tuple

import numpy as np


class Camera:
    def __init__(self, index=0, width: int = 960, height: int = 540, fps: int = 30,
                 mirror: bool = True):
        """`index` is a webcam number, or a path to a video file (for demos/tests)."""
        import cv2

        self._cv2 = cv2
        self.is_file = isinstance(index, str) and not index.isdigit()
        if isinstance(index, str) and index.isdigit():
            index = int(index)
        if self.is_file:
            self.cap = cv2.VideoCapture(index)
        else:
            backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            if self.is_file:
                raise RuntimeError(f"cannot open video file '{index}'")
            raise RuntimeError(f"cannot open camera {index}. Is another app using it? "
                               "On macOS grant camera access to your terminal.")
        # Video files are paced at their native frame rate so timing-based
        # gestures (holds, swipes) behave exactly as they would live.
        self._period = 1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or fps) if self.is_file else 0.0
        self.finished = False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.mirror = mirror
        self._frame: Optional[np.ndarray] = None
        self._t = 0.0
        self._seq = 0
        self._cond = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        next_t = time.monotonic()
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file:
                    self.finished = True
                    with self._cond:
                        self._cond.notify_all()
                    return
                time.sleep(0.01)
                continue
            if self._period:
                next_t += self._period
                time.sleep(max(0.0, next_t - time.monotonic()))
            if self.mirror:
                frame = self._cv2.flip(frame, 1)
            with self._cond:
                self._frame, self._t = frame, time.monotonic()
                self._seq += 1
                self._cond.notify_all()

    def read(self, last_seq: int = -1, timeout: float = 1.0) -> Tuple[int, float, Optional[np.ndarray]]:
        """Block until a frame newer than `last_seq` exists; return (seq, t, frame)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq or not self._running or self.finished,
                                timeout)
            if self.finished and self._seq == last_seq:
                raise EOFError("end of video")
            return self._seq, self._t, self._frame

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1)
        self.cap.release()
