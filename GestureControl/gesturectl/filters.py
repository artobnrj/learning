"""Signal filters.

The One Euro Filter (Casiez, Roussel & Vogel, CHI 2012) is an adaptive
low-pass filter: at low speed it uses a low cutoff (kills jitter while you
hold still), at high speed it raises the cutoff (kills lag while you move).
That trade-off is exactly what a hand-driven cursor needs.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _alpha(dt: float, cutoff: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Vectorised One Euro filter; works on scalars or numpy arrays.

    Args:
        min_cutoff: cutoff (Hz) when the signal is still. Lower = smoother.
        beta: how fast the cutoff grows with speed. Higher = less lag.
        d_cutoff: cutoff for the derivative estimate.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: Optional[np.ndarray] = None
        self._dx: Optional[np.ndarray] = None
        self._t: Optional[float] = None

    def __call__(self, x, t: float):
        x = np.asarray(x, dtype=np.float64)
        if self._x is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self._t
        if dt <= 0:
            return self._x.copy()
        a_d = _alpha(dt, self.d_cutoff)
        dx = (x - self._x) / dt
        dx_hat = a_d * dx + (1 - a_d) * self._dx
        # One cutoff per element so a fast x does not un-smooth a still y.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        tau = 1.0 / (2.0 * math.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        x_hat = a * x + (1 - a) * self._x
        self._x, self._dx, self._t = x_hat, dx_hat, t
        return x_hat.copy()


class Hysteresis:
    """Boolean with separate on/off thresholds (a Schmitt trigger).

    `low_is_on=True` means the state turns on when the value drops below
    `on` and turns off when it rises above `off` (e.g. pinch distance).
    """

    def __init__(self, on: float, off: float, low_is_on: bool = False):
        self.on, self.off, self.low_is_on = on, off, low_is_on
        self.state = False

    def update(self, value: float) -> bool:
        if self.low_is_on:
            if not self.state and value < self.on:
                self.state = True
            elif self.state and value > self.off:
                self.state = False
        else:
            if not self.state and value > self.on:
                self.state = True
            elif self.state and value < self.off:
                self.state = False
        return self.state
