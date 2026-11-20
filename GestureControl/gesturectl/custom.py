"""Few-shot user-defined poses with a k-nearest-neighbour classifier.

Teach a new pose with ~100 frames (about 4 seconds of video):

    gesturectl train spiderman

then bind it like any built-in pose: `{gesture: "custom:spiderman", ...}`.

Embedding: the 21 3-D landmarks are translated so the wrist is the origin,
left hands are mirrored onto right hands, and everything is scaled by palm
length. That makes the 63-D vector invariant to position, distance and
handedness while keeping orientation (so thumbs-up != thumbs-down).

Open-set rejection: a pose is only reported when the nearest neighbours
agree *and* are closer than a per-model distance threshold learnt from
leave-one-out distances in the training data, so random hand shapes are
rejected instead of being forced into the nearest class.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .features import palm_length
from .landmarks import WRIST, Hand


def embed(hand: Hand) -> np.ndarray:
    g = hand.geometry.copy()
    g = g - g[WRIST]
    if hand.handedness == "Left":
        g[:, 0] *= -1
    g /= palm_length(hand.geometry)
    return g.reshape(-1)


class CustomGestureModel:
    def __init__(self, X: Optional[np.ndarray] = None, y: Optional[List[str]] = None, k: int = 5):
        self.X = np.zeros((0, 63)) if X is None else np.asarray(X, dtype=np.float64)
        self.y = np.array([] if y is None else list(y), dtype=object)
        self.k = k
        self.threshold = np.inf
        if len(self.X):
            self._fit_threshold()

    @property
    def labels(self) -> List[str]:
        return sorted(set(self.y.tolist()))

    def __len__(self) -> int:
        return len(self.X)

    def _fit_threshold(self) -> None:
        if len(self.X) < 2:
            self.threshold = np.inf
            return
        d = np.linalg.norm(self.X[:, None, :] - self.X[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        nn = d.min(axis=1)
        # Generous margin over the typical intra-class spread.
        self.threshold = float(np.percentile(nn, 95) * 3.0 + 0.15)

    def add(self, label: str, samples: np.ndarray) -> None:
        samples = np.atleast_2d(samples)
        self.X = np.vstack([self.X, samples])
        self.y = np.concatenate([self.y, np.array([label] * len(samples), dtype=object)])
        self._fit_threshold()

    def remove(self, label: str) -> int:
        keep = self.y != label
        n = int((~keep).sum())
        self.X, self.y = self.X[keep], self.y[keep]
        self._fit_threshold()
        return n

    def predict(self, vec: np.ndarray) -> Tuple[Optional[str], float]:
        if len(self.X) == 0:
            return None, 0.0
        d = np.linalg.norm(self.X - vec, axis=1)
        k = min(self.k, len(d))
        idx = np.argpartition(d, k - 1)[:k]
        label, votes = Counter(self.y[idx].tolist()).most_common(1)[0]
        mean_d = float(d[idx][self.y[idx] == label].mean())
        conf = votes / k
        if conf < 0.6 or mean_d > self.threshold:
            return None, conf
        return label, conf

    def save(self, path) -> None:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=self.X, y=self.y.astype(str))

    @classmethod
    def load(cls, path) -> "CustomGestureModel":
        path = Path(path).expanduser()
        if not path.exists():
            return cls()
        data = np.load(path, allow_pickle=False)
        return cls(data["X"], data["y"].tolist())
