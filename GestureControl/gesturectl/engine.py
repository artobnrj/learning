"""The gesture engine: hands in, actions out.

Per frame:
    hands --> per-hand slot (features -> pose -> temporal smoothing -> motion)
          --> choose the primary hand (largest = closest to the camera)
          --> two-hand bindings (if both hands agree on a pose they win the frame)
          --> single-hand bindings for global + current mode
          --> actions (debounced, with per-action cooldowns)

The engine is pure logic with an injected clock and action runner, so the
whole thing is unit-testable with synthetic landmarks and no camera.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Binding, Bindings
from .custom import CustomGestureModel, embed
from .features import FeatureExtractor, HandFeatures
from .filters import OneEuroFilter
from .landmarks import INDEX_MCP, MIDDLE_MCP, PINKY_MCP, WRIST, Hand
from .motion import Dial, Joystick, PinchZoom, SwipeDetector, SwipeEvent
from .poses import PoseSmoother, classify

log = logging.getLogger(__name__)


@dataclass
class Event:
    t: float
    trigger: str
    action: str
    magnitude: int = 1

    def __str__(self) -> str:
        mag = f" x{self.magnitude}" if abs(self.magnitude) != 1 else ""
        return f"{self.trigger} -> {self.action}{mag}"


@dataclass
class _Hold:
    start: Optional[float] = None
    last_match: float = 0.0
    fired: bool = False
    next_fire: Optional[float] = None


class HandSlot:
    """Everything we remember about one physical hand across frames."""

    def __init__(self, cfg: dict, key: str):
        self.key = key
        self.extract = FeatureExtractor()
        self.smoother = PoseSmoother()
        self.swipe = SwipeDetector(**cfg["swipe"])
        self.hand: Optional[Hand] = None
        self.features: Optional[HandFeatures] = None
        self.raw = self.pose = "none"
        self.speed = 0.0
        self.swipe_event: Optional[SwipeEvent] = None
        self._last_c: Optional[np.ndarray] = None
        self._last_t: Optional[float] = None

    def update(self, hand: Hand, t: float, custom: Optional[CustomGestureModel]) -> None:
        f = self.extract(hand)
        raw = classify(f)
        if custom is not None and len(custom):
            label, _ = custom.predict(embed(hand))
            if label:
                raw = f"custom:{label}"
        self.hand, self.features, self.raw = hand, f, raw
        self.pose = self.smoother.update(raw)
        if self._last_c is not None and t > self._last_t:
            inst = float(np.linalg.norm(f.palm_center - self._last_c)) / (t - self._last_t) / f.scale
            self.speed = 0.6 * self.speed + 0.4 * inst
        self._last_c, self._last_t = f.palm_center.copy(), t
        self.swipe_event = self.swipe.update(t, f.palm_center, f.scale, self.pose)

    def lost(self) -> None:
        self.smoother.reset()
        self.swipe.reset()
        self.pose = self.raw = "none"
        self.hand = self.features = None
        self._last_c = self._last_t = None
        self.speed = 0.0
        self.swipe_event = None

    def pixel_center(self, w: int, h: int) -> Tuple[int, int]:
        n = self.hand.norm[[WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP], :2].mean(axis=0)
        return int(n[0] * w), int(n[1] * h)


class Engine:
    def __init__(self, cfg: dict, runner, custom: Optional[CustomGestureModel] = None):
        self.cfg = cfg
        self.runner = runner
        self.custom = custom
        self.bindings = Bindings.from_config(cfg)
        self.mode_names = list(self.bindings.modes)
        ctl = cfg["control"]
        start = ctl.get("start_mode")
        self.mode = start if start in self.bindings.modes else self.mode_names[0]
        self.paused = bool(ctl.get("start_paused"))
        self.min_hand_size = ctl["min_hand_size"]
        self.cooldown = ctl["action_cooldown"]
        self.grace = ctl["hold_grace"]
        self.still_speed = ctl["still_speed"]

        sc = dict(cfg["scroll"])
        self.scroll_sign = -1 if sc.pop("invert", False) else 1
        self.joy = Joystick(**sc)
        self.dial = Dial(cfg["dial"]["step_deg"])
        self.zoom = PinchZoom()
        m = cfg["mouse"]
        self.cursor_filter = OneEuroFilter(m["min_cutoff"], m["beta"])
        self.mouse_region = m["region"]
        self.click_freeze = m["click_freeze"]

        self.slots: Dict[str, HandSlot] = {}
        self.visible: List[HandSlot] = []
        self.primary: Optional[HandSlot] = None
        self.events: Deque[Event] = deque(maxlen=8)
        self.progress: List[tuple] = []   # (slots, fraction, label) for the HUD
        self.continuous: Optional[str] = None   # "scroll" | "dial" | "zoom" | "mouse" for the HUD
        self.cursor: Optional[Tuple[float, float]] = None
        self._holds: Dict[str, _Hold] = {}
        self._taps: Dict[str, _Hold] = {}
        self._cool: Dict[str, float] = {}
        self._mouse_down = False
        self._prev_pinch = False
        self._prev_pinch_mid = False
        self._freeze_until = 0.0
        self._frame_events: List[Event] = []

    # ------------------------------------------------------------------ API
    def set_mode(self, name: str) -> None:
        if name not in self.bindings.modes:
            log.warning("unknown mode %s", name)
            return
        if name != self.mode:
            self.mode = name
            self._reset_continuous()
            self._release_mouse()

    def next_mode(self) -> None:
        i = self.mode_names.index(self.mode)
        self.set_mode(self.mode_names[(i + 1) % len(self.mode_names)])

    def active_bindings(self) -> List[Binding]:
        bs = [b for b in self.bindings.global_ if self.mode not in b.options.get("except_modes", ())]
        bs += self.bindings.modes[self.mode]
        return [b for b in bs if b.when_paused] if self.paused else bs

    def update(self, hands: Sequence[Hand], t: float) -> List[Event]:
        self._frame_events = []
        self.progress = []
        self.continuous = None
        evaluated: set = set()

        self._assign_slots(hands, t)
        active = self.active_bindings()
        two = [b for b in active if b.two_hands]
        one = [b for b in active if not b.two_hands]

        if self._two_hand_step(two, t, evaluated):
            self._release_mouse()
            self._reset_continuous(keep_zoom=True)
        elif self.primary is not None:
            self.zoom.reset()
            self._single_hand_step(one, self.primary, t, evaluated)
        else:
            self.zoom.reset()
            self._reset_continuous()
            self._release_mouse()

        # Holds that were not evaluated this frame (other mode, paused, hand
        # gone) must restart from zero next time, never resume.
        for key in list(self._holds):
            if key not in evaluated:
                del self._holds[key]
        for key in list(self._taps):
            if key not in evaluated:
                del self._taps[key]
        return self._frame_events

    # -------------------------------------------------------------- stages
    def _assign_slots(self, hands: Sequence[Hand], t: float) -> None:
        seen = set()
        valid = []
        for hand in hands:
            key = hand.handedness if hand.handedness not in seen else hand.handedness + "#2"
            seen.add(key)
            slot = self.slots.get(key) or self.slots.setdefault(key, HandSlot(self.cfg, key))
            slot.update(hand, t, self.custom)
            if slot.features.scale >= self.min_hand_size:
                valid.append(slot)
        for key, slot in self.slots.items():
            if key not in seen and slot.hand is not None:
                slot.lost()
        valid.sort(key=lambda s: s.features.scale, reverse=True)
        self.visible = valid
        self.primary = valid[0] if valid else None

    def _two_hand_step(self, bindings: List[Binding], t: float, evaluated: set) -> bool:
        if len(self.visible) < 2 or not bindings:
            return False
        a, b = self.visible[0], self.visible[1]
        engaged = False
        still = max(a.speed, b.speed) < self.still_speed
        for bd in bindings:
            match = bd.matches(a.pose) and bd.matches(b.pose)
            if bd.trigger in ("hold", "repeat"):
                evaluated.add(bd.id)
                self._hold_step(bd, match, t, still, (a, b))
            elif bd.trigger == "zoom" and match:
                self.continuous = "zoom"
                scale = (a.features.scale + b.features.scale) / 2
                step = self.zoom.update(a.features.pinch_point, b.features.pinch_point, scale)
                if step:
                    self._fire(f"both {bd.poses[0]} zoom", (bd.cw if step > 0 else bd.ccw,), t,
                               cooldown=False)
            engaged |= match
        return engaged

    def _single_hand_step(self, bindings: List[Binding], slot: HandSlot, t: float,
                          evaluated: set) -> None:
        pose = slot.pose
        swipe = slot.swipe_event
        still = slot.speed < self.still_speed
        joy_used = dial_used = False
        mouse_binding = None
        swiped = False

        for bd in bindings:
            m = bd.matches(pose)
            if bd.trigger in ("hold", "repeat"):
                evaluated.add(bd.id)
                self._hold_step(bd, m, t, still or bd.trigger == "repeat", (slot,))
            elif bd.trigger == "tap":
                evaluated.add(bd.id)
                self._tap_step(bd, m, t)
            elif bd.trigger == "swipe":
                if swipe and swipe.direction == bd.value and bd.matches(swipe.pose):
                    self._fire(f"{swipe.pose} swipe {swipe.direction}", bd.actions, t)
                    swiped = True
            elif bd.trigger == "joystick" and m and not joy_used:
                joy_used = True
                self.continuous = "scroll"
                ticks = self.joy.update(t, slot.features.palm_center, slot.features.scale)
                if ticks:
                    for a in bd.actions:
                        if a == "scroll":
                            self.runner.backend.scroll(0, -ticks * self.scroll_sign)
                        else:
                            self._fire(f"{pose} joystick", (a,), t, magnitude=ticks, cooldown=False)
            elif bd.trigger == "dial" and m and not dial_used:
                dial_used = True
                self.continuous = "dial"
                steps = self.dial.update(slot.features.roll)
                if steps:
                    act = bd.cw if steps > 0 else bd.ccw
                    self._fire(f"{pose} dial {'cw' if steps > 0 else 'ccw'}", (act,), t,
                               magnitude=abs(steps), cooldown=False)
            elif bd.trigger == "mouse":
                mouse_binding = bd

        if swiped:  # a flick cancels any hold that was building up
            for key in list(self._holds):
                if not self._holds[key].fired:
                    del self._holds[key]
            self.progress = []
        if not joy_used:
            self.joy.reset()
        if not dial_used:
            self.dial.reset()
        if mouse_binding is not None:
            self._mouse_step(mouse_binding, slot, t, suspended=joy_used)
        else:
            self._release_mouse()

    # ------------------------------------------------------------ triggers
    def _hold_step(self, bd: Binding, match: bool, t: float, still: bool, slots: tuple) -> None:
        st = self._holds.setdefault(bd.id, _Hold())
        if not match:
            if st.start is not None and t - st.last_match > self.grace:
                self._holds[bd.id] = _Hold()
            return
        st.last_match = t
        if st.start is None:
            st.start = t
        name = ("both " if bd.two_hands else "") + "/".join(bd.poses)
        if bd.trigger == "hold":
            if st.fired:
                return
            if not still:
                st.start = t  # moving hand: restart the countdown
            frac = (t - st.start) / max(bd.value, 1e-6)
            self.progress.append((slots, min(frac, 1.0), f"{name} → {', '.join(bd.actions)}"))
            if frac >= 1.0:
                st.fired = True
                self._fire(f"{name} hold {bd.value}s", bd.actions, t)
        else:  # repeat
            if st.next_fire is None:
                st.next_fire = st.start + 0.2  # brief debounce before the first fire
            if t >= st.next_fire:
                st.next_fire = t + bd.value
                self._fire(f"{name} repeat", bd.actions, t, cooldown=False)

    def _tap_step(self, bd: Binding, match: bool, t: float) -> None:
        st = self._taps.setdefault(bd.id, _Hold())
        if match:
            if st.start is None:
                st.start = t
            st.last_match = t
        elif st.start is not None:
            if st.last_match - st.start <= bd.value:
                self._fire(f"{'/'.join(bd.poses)} tap", bd.actions, t)
            self._taps[bd.id] = _Hold()

    def _mouse_step(self, bd: Binding, slot: HandSlot, t: float, suspended: bool) -> None:
        clutch = bd.options.get("clutch", "fist")
        if suspended or slot.pose == clutch:
            self._release_mouse()
            return
        self.continuous = "mouse"
        f, b = slot.features, self.runner.backend
        x0, y0, x1, y1 = self.mouse_region
        nx, ny = slot.hand.norm[INDEX_MCP, :2]  # knuckle: stays put while you pinch
        u = float(np.clip((nx - x0) / (x1 - x0), 0, 1))
        v = float(np.clip((ny - y0) / (y1 - y0), 0, 1))
        su, sv = self.cursor_filter([u, v], t)
        if f.pinching != self._prev_pinch:
            self._freeze_until = t + self.click_freeze
        if t >= self._freeze_until:
            w, h = b.screen_size()
            b.move(int(su * (w - 1)), int(sv * (h - 1)))
            self.cursor = (su, sv)
        if f.pinching and not self._mouse_down:
            b.press("left")
            self._mouse_down = True
            self._log_event(Event(t, "pinch", "mouse down"))
        elif not f.pinching and self._mouse_down:
            b.release("left")
            self._mouse_down = False
            self._log_event(Event(t, "release", "mouse up"))
        if f.pinching_middle and not self._prev_pinch_mid:
            b.click("right", 1)
            self._log_event(Event(t, "thumb+middle pinch", "right click"))
        self._prev_pinch, self._prev_pinch_mid = f.pinching, f.pinching_middle

    # -------------------------------------------------------------- helpers
    def _fire(self, trigger: str, actions: Sequence[str], t: float, magnitude: int = 1,
              cooldown: bool = True) -> None:
        for a in actions:
            if cooldown and t - self._cool.get(a, -1e9) < self.cooldown:
                continue
            self._cool[a] = t
            if a == "toggle_pause":
                self.paused = not self.paused
                self._reset_continuous()
                self._release_mouse()
            elif a == "next_mode":
                self.next_mode()
            elif a.startswith("mode:"):
                self.set_mode(a[5:])
            else:
                self.runner(a, magnitude)
            self._log_event(Event(t, trigger, a, magnitude))

    def _log_event(self, ev: Event) -> None:
        log.info("%s", ev)
        self.events.append(ev)
        self._frame_events.append(ev)

    def _release_mouse(self) -> None:
        if self._mouse_down:
            self.runner.backend.release("left")
            self._mouse_down = False
        self._prev_pinch = self._prev_pinch_mid = False

    def _reset_continuous(self, keep_zoom: bool = False) -> None:
        self.joy.reset()
        self.dial.reset()
        if not keep_zoom:
            self.zoom.reset()
