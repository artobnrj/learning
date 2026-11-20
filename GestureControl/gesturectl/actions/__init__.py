"""Semantic desktop actions resolved per operating system.

Bindings refer to actions by *meaning* ("close_window") rather than by key
("alt+f4"), and this module resolves the meaning for Windows, macOS or
Linux. Action strings:

    close_window              built-in semantic action (see ACTIONS)
    hotkey:ctrl+shift+t       any key combination
    key:down                  single key
    open:https://...          open a URL / file with the default app
    run:<shell command>       run a command (opt-in, see config)
    type:<text>               type text
    scroll:<n>                scroll n ticks (+ = down)
"""
from __future__ import annotations

import logging
import platform
import queue
import shlex
import subprocess
import sys
import threading
import webbrowser
from typing import Callable, Dict, Optional

log = logging.getLogger(__name__)

OS = {"Windows": "windows", "Darwin": "darwin"}.get(platform.system(), "linux")

# action -> {os: hotkey} | {os: ["cmd", ...]} | callable name
HOTKEYS: Dict[str, Dict[str, str]] = {
    "close_window":   {"windows": "alt+f4",        "darwin": "cmd+w",          "linux": "alt+f4"},
    "quit_app":       {"windows": "alt+f4",        "darwin": "cmd+q",          "linux": "ctrl+q"},
    "minimize_window": {"windows": "win+down",     "darwin": "cmd+m",          "linux": "super+h"},
    "maximize_window": {"windows": "win+up",       "darwin": "ctrl+cmd+f",     "linux": "super+up"},
    "show_desktop":   {"windows": "win+d",         "darwin": "cmd+f3",         "linux": "super+d"},
    "task_view":      {"windows": "win+tab",       "darwin": "ctrl+up",        "linux": "super"},
    "switch_app":     {"windows": "alt+tab",       "darwin": "cmd+tab",        "linux": "alt+tab"},
    "next_desktop":   {"windows": "ctrl+win+right", "darwin": "ctrl+right",    "linux": "ctrl+alt+right"},
    "prev_desktop":   {"windows": "ctrl+win+left", "darwin": "ctrl+left",      "linux": "ctrl+alt+left"},
    "new_window":     {"windows": "ctrl+n",        "darwin": "cmd+n",          "linux": "ctrl+n"},
    "next_tab":       {"windows": "ctrl+tab",      "darwin": "ctrl+tab",       "linux": "ctrl+tab"},
    "prev_tab":       {"windows": "ctrl+shift+tab", "darwin": "ctrl+shift+tab", "linux": "ctrl+shift+tab"},
    "close_tab":      {"windows": "ctrl+w",        "darwin": "cmd+w",          "linux": "ctrl+w"},
    "reopen_tab":     {"windows": "ctrl+shift+t",  "darwin": "cmd+shift+t",    "linux": "ctrl+shift+t"},
    "zoom_in":        {"windows": "ctrl+=",        "darwin": "cmd+=",          "linux": "ctrl+="},
    "zoom_out":       {"windows": "ctrl+-",        "darwin": "cmd+-",          "linux": "ctrl+-"},
    "zoom_reset":     {"windows": "ctrl+0",        "darwin": "cmd+0",          "linux": "ctrl+0"},
    "screenshot":     {"windows": "win+print_screen", "darwin": "cmd+shift+3", "linux": "print_screen"},
    "snip":           {"windows": "win+shift+s",   "darwin": "cmd+shift+4",    "linux": "shift+print_screen"},
    "fullscreen":     {"windows": "f11",           "darwin": "ctrl+cmd+f",     "linux": "f11"},
    "browser_back":   {"windows": "alt+left",      "darwin": "cmd+[",          "linux": "alt+left"},
    "browser_forward": {"windows": "alt+right",    "darwin": "cmd+]",          "linux": "alt+right"},
    "volume_up":      {"*": "media_volume_up"},
    "volume_down":    {"*": "media_volume_down"},
    "mute":           {"*": "media_volume_mute"},
    "play_pause":     {"*": "media_play_pause"},
    "next_track":     {"*": "media_next"},
    "prev_track":     {"*": "media_previous"},
    # Short-video feeds (Instagram Reels, YouTube Shorts, TikTok web) all
    # treat arrow keys as next/previous video.
    "reels_next":     {"*": "down"},
    "reels_prev":     {"*": "up"},
    "page_down":      {"*": "page_down"},
    "page_up":        {"*": "page_up"},
    "escape":         {"*": "esc"},
    "enter":          {"*": "enter"},
}

COMMANDS: Dict[str, Dict[str, str]] = {
    "sleep": {
        "windows": "rundll32.exe powrprof.dll,SetSuspendState 0,1,0",
        "darwin": "pmset sleepnow",
        "linux": "systemctl suspend",
    },
    "lock": {
        "windows": "rundll32.exe user32.dll,LockWorkStation",
        "darwin": "pmset displaysleepnow",
        "linux": "loginctl lock-session",
    },
}

# Mouse-based semantic actions.
MOUSE_ACTIONS = {"like", "click", "double_click", "right_click", "click_center"}

DESCRIPTIONS = {
    "like": "double-click the centre of the screen (double-tap-to-like on Reels/TikTok/Shorts)",
    "click_center": "click the centre of the screen (pause/play a reel)",
    "sleep": "suspend the computer",
    "lock": "lock the screen",
}


def known_actions():
    return sorted(set(HOTKEYS) | set(COMMANDS) | MOUSE_ACTIONS)


def describe(action: str) -> str:
    if action in DESCRIPTIONS:
        return DESCRIPTIONS[action]
    if action in HOTKEYS:
        k = HOTKEYS[action].get(OS) or HOTKEYS[action].get("*")
        return f"press {k}"
    if action in COMMANDS:
        return f"run `{COMMANDS[action][OS]}`"
    return action


class ActionRunner:
    """Executes action strings on a background worker thread.

    Running actions off the vision thread means a slow `open:` or a
    blocking system call never drops camera frames. Order is preserved by
    the single FIFO queue.
    """

    def __init__(self, backend, allow_run: bool = False, os_name: str = OS,
                 on_error: Optional[Callable[[str], None]] = None, synchronous: bool = False):
        self.backend = backend
        self.allow_run = allow_run
        self.os = os_name
        self.on_error = on_error or (lambda m: log.warning(m))
        self.synchronous = synchronous
        self._q: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._thread = None
        if not synchronous:
            self._thread = threading.Thread(target=self._worker, name="actions", daemon=True)
            self._thread.start()

    def __call__(self, action: str, magnitude: int = 1) -> None:
        if self.synchronous:
            self._execute(action, magnitude)
        else:
            self._q.put((action, magnitude))

    def close(self):
        if self._thread:
            self._q.put(None)
            self._thread.join(timeout=2)

    def _worker(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            self._execute(*item)

    def _execute(self, action: str, magnitude: int = 1) -> None:
        try:
            self._dispatch(action, magnitude)
        except Exception as exc:  # never let one bad action kill the loop
            self.on_error(f"action '{action}' failed: {exc}")

    def _dispatch(self, action: str, magnitude: int) -> None:
        from .backend import parse_combo

        b = self.backend
        kind, _, arg = action.partition(":")
        n = max(1, abs(magnitude))
        if kind == "hotkey" or kind == "key":
            b.tap(parse_combo(arg), n)
        elif kind == "scroll":
            ticks = int(arg or 1) * (magnitude or 1)
            b.scroll(0, -ticks)  # pynput: positive dy scrolls up
        elif kind == "open":
            if isinstance(b, _dry_types()):
                b._rec("open", arg)
            else:
                webbrowser.open(arg)
        elif kind == "type":
            b.type_text(arg)
        elif kind == "run":
            if not self.allow_run:
                raise PermissionError("run: actions are disabled (set actions.allow_run: true)")
            self._shell(arg)
        elif action in HOTKEYS:
            combo = HOTKEYS[action].get(self.os) or HOTKEYS[action]["*"]
            b.tap(parse_combo(combo), n)
        elif action in COMMANDS:
            self._shell(COMMANDS[action][self.os])
        elif action in MOUSE_ACTIONS:
            w, h = b.screen_size()
            center = (w // 2, h // 2)
            if action == "like":
                b.click("left", 2, at=center)
            elif action == "click_center":
                b.click("left", 1, at=center)
            elif action == "double_click":
                b.click("left", 2)
            elif action == "right_click":
                b.click("right", 1)
            else:
                b.click("left", 1)
        else:
            raise ValueError(f"unknown action '{action}'")

    def _shell(self, cmd: str) -> None:
        if isinstance(self.backend, _dry_types()):
            self.backend._rec("shell", cmd)
            return
        log.info("running: %s", cmd)
        if sys.platform.startswith("win"):
            subprocess.Popen(cmd, shell=True)
        else:
            subprocess.Popen(shlex.split(cmd))


def _dry_types():
    from .backend import DryRunBackend
    return (DryRunBackend,)
