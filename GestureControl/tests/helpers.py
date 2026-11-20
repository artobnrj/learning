import copy

from gesturectl.actions import ActionRunner
from gesturectl.actions.backend import DryRunBackend
from gesturectl.config import load_config
from gesturectl.engine import Engine
from gesturectl.synthetic import make_hand

FPS = 30.0
DT = 1.0 / FPS


def engine(mode="desktop", overrides=None, custom=None, os_name="linux"):
    cfg = load_config()
    cfg["custom_gestures"] = None
    cfg["control"]["start_mode"] = mode
    for path, value in (overrides or {}).items():
        node = cfg
        keys = path.split(".")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = copy.deepcopy(value)
    backend = DryRunBackend()
    runner = ActionRunner(backend, synchronous=True, os_name=os_name)
    return Engine(cfg, runner, custom), backend


class Sim:
    """Drives an engine with synthetic frames at 30 fps."""

    def __init__(self, eng, t0=100.0):
        self.eng = eng
        self.t = t0
        self.events = []

    def frames(self, seconds, hands_fn):
        """hands_fn(progress 0..1) -> list of Hand."""
        n = max(1, int(round(seconds * FPS)))
        for i in range(n):
            self.t += DT
            self.events += self.eng.update(hands_fn(i / max(1, n - 1)), self.t)
        return self

    def hold(self, pose, seconds, **kw):
        return self.frames(seconds, lambda p: [make_hand(pose, **kw)])

    def move(self, pose, seconds, start, end, **kw):
        def fn(p):
            c = (start[0] + (end[0] - start[0]) * p, start[1] + (end[1] - start[1]) * p)
            return [make_hand(pose, center=c, **kw)]
        return self.frames(seconds, fn)

    def none(self, seconds):
        return self.frames(seconds, lambda p: [])

    @property
    def actions(self):
        return [e.action for e in self.events]
