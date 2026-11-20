"""Configuration loading, deep-merging and binding validation."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .actions import known_actions
from .poses import POSES

DEFAULT_PATH = Path(__file__).with_name("default_config.yaml")
USER_PATH = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "gesturectl" / "config.yaml"

TRIGGERS = ("hold", "repeat", "tap", "swipe", "joystick", "dial", "zoom", "mouse")
INTERNAL = ("toggle_pause", "next_mode")
ACTION_PREFIXES = ("hotkey:", "key:", "open:", "run:", "type:", "scroll:", "mode:")


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: Optional[str] = None) -> dict:
    with open(DEFAULT_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    candidate = Path(path).expanduser() if path else USER_PATH
    if path and not candidate.exists():
        raise ConfigError(f"config file not found: {candidate}")
    if candidate.exists():
        with open(candidate, "r", encoding="utf-8") as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    return cfg


@dataclass
class Binding:
    id: str
    trigger: str
    poses: Tuple[str, ...] = ()
    two_hands: bool = False
    value: Any = None                     # hold seconds, swipe direction, ...
    actions: Tuple[str, ...] = ()
    cw: Optional[str] = None              # dial / zoom
    ccw: Optional[str] = None
    when_paused: bool = False
    options: Dict[str, Any] = field(default_factory=dict)

    def matches(self, pose: str) -> bool:
        return "any" in self.poses or pose in self.poses

    def label(self) -> str:
        who = ("both hands " if self.two_hands else "") + "/".join(self.poses or ("hand",))
        if self.trigger in ("hold", "repeat", "tap"):
            how = f"{self.trigger} {self.value}s"
        elif self.trigger == "swipe":
            how = f"swipe {self.value}"
        else:
            how = self.trigger
        what = ", ".join(self.actions) if self.actions else (
            f"{self.cw} / {self.ccw}" if self.cw else self.trigger)
        return f"{who} · {how} → {what}"


def _check_action(a: str, where: str) -> None:
    if a in INTERNAL or a in known_actions() or a.startswith(ACTION_PREFIXES):
        return
    raise ConfigError(f"{where}: unknown action '{a}'. Known: {', '.join(known_actions())}")


def _check_pose(p: str, where: str) -> None:
    if p in POSES or p == "any" or p.startswith("custom:"):
        return
    raise ConfigError(f"{where}: unknown gesture '{p}'. Known: {', '.join(POSES)}")


def parse_bindings(raw: List[dict], scope: str) -> List[Binding]:
    out: List[Binding] = []
    for i, b in enumerate(raw or []):
        where = f"{scope}[{i}]"
        if not isinstance(b, dict):
            raise ConfigError(f"{where}: binding must be a mapping")
        triggers = [t for t in TRIGGERS if t in b]
        if len(triggers) != 1:
            raise ConfigError(f"{where}: need exactly one trigger of {TRIGGERS}, got {triggers}")
        trig = triggers[0]
        two = "two_hands" in b
        poses = b.get("two_hands") if two else b.get("gesture", "any" if trig == "mouse" else None)
        if poses is None:
            raise ConfigError(f"{where}: missing 'gesture' or 'two_hands'")
        poses = tuple(poses) if isinstance(poses, list) else (poses,)
        for p in poses:
            _check_pose(p, where)
        acts = b.get("action", [])
        acts = tuple(acts) if isinstance(acts, list) else ((acts,) if acts else ())
        cw, ccw = (b.get("in"), b.get("out")) if trig == "zoom" else (b.get("cw"), b.get("ccw"))
        if trig in ("dial", "zoom"):
            if not (cw and ccw):
                raise ConfigError(f"{where}: {trig} needs both "
                                  f"{'in/out' if trig == 'zoom' else 'cw/ccw'} actions")
            for a in (cw, ccw):
                _check_action(a, where)
        elif trig == "joystick":
            acts = acts or ("scroll",)
        elif trig != "mouse":
            if not acts:
                raise ConfigError(f"{where}: missing 'action'")
            for a in acts:
                _check_action(a, where)
        if trig == "swipe" and b["swipe"] not in ("up", "down", "left", "right"):
            raise ConfigError(f"{where}: swipe must be up/down/left/right")
        opts = {k: v for k, v in b.items()
                if k not in (*TRIGGERS, "gesture", "two_hands", "action", "cw", "ccw", "in", "out",
                             "when_paused")}
        out.append(Binding(id=where, trigger=trig, poses=poses, two_hands=two, value=b[trig],
                           actions=acts, cw=cw, ccw=ccw, when_paused=bool(b.get("when_paused")),
                           options=opts))
    return out


@dataclass
class Bindings:
    global_: List[Binding]
    modes: Dict[str, List[Binding]]

    @classmethod
    def from_config(cls, cfg: dict) -> "Bindings":
        modes = {name: parse_bindings(lst, f"modes.{name}") for name, lst in (cfg.get("modes") or {}).items()}
        if not modes:
            raise ConfigError("at least one mode is required")
        return cls(parse_bindings(cfg.get("global"), "global"), modes)
