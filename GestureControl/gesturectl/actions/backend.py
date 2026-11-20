"""Input-injection backends.

`PynputBackend` synthesises real keyboard/mouse events (Win32 SendInput,
macOS Quartz CGEvents, Linux X11 XTest). `DryRunBackend` records calls
instead - used by --dry-run, the replay tool and the test-suite.
"""
from __future__ import annotations

import logging
import platform
import subprocess
from typing import List, Sequence, Tuple

log = logging.getLogger(__name__)

KEY_ALIASES = {
    "win": "cmd", "super": "cmd", "meta": "cmd", "command": "cmd", "windows": "cmd",
    "option": "alt", "opt": "alt", "control": "ctrl", "return": "enter", "escape": "esc",
    "prtsc": "print_screen", "print": "print_screen", "pgdn": "page_down", "pgup": "page_up",
    "del": "delete", "plus": "=", "minus": "-", "equal": "=",
    "volume_up": "media_volume_up", "volume_down": "media_volume_down",
    "volume_mute": "media_volume_mute", "play_pause": "media_play_pause",
    "next_track": "media_next", "prev_track": "media_previous",
}


def parse_combo(combo: str) -> List[str]:
    """'Ctrl+Shift+T' -> ['ctrl', 'shift', 't'] with aliases resolved."""
    parts = [p.strip().lower() for p in combo.replace(" ", "").split("+")]
    if combo.endswith("++"):  # literal plus key, e.g. "ctrl++"
        parts = [p for p in parts if p] + ["="]
    return [KEY_ALIASES.get(p, p) for p in parts if p]


def screen_size() -> Tuple[int, int]:
    """Primary screen size in pixels, with per-OS fallbacks."""
    system = platform.system()
    try:
        if system == "Windows":
            import ctypes
            user32 = ctypes.windll.user32
            try:
                user32.SetProcessDPIAware()
            except Exception:
                pass
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        if system == "Darwin":
            out = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True,
                                 text=True, timeout=5).stdout
            for line in out.splitlines():
                if "Resolution" in line:
                    nums = [int(s) for s in line.replace("x", " ").split() if s.isdigit()]
                    if len(nums) >= 2:
                        # Quartz uses points; Retina panels report 2x pixels.
                        w, h = nums[0], nums[1]
                        return (w // 2, h // 2) if "Retina" in line else (w, h)
        else:
            from Xlib import display  # installed with pynput on Linux
            s = display.Display().screen()
            return s.width_in_pixels, s.height_in_pixels
    except Exception as exc:  # pragma: no cover - depends on host
        log.debug("screen size probe failed: %s", exc)
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        size = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return size
    except Exception:
        return 1920, 1080


class DryRunBackend:
    """Records every call. Nothing touches the real OS."""

    def __init__(self, size: Tuple[int, int] = (1920, 1080), echo: bool = False):
        self.calls: List[tuple] = []
        self._size = size
        self.echo = echo

    def _rec(self, *call):
        self.calls.append(call)
        if self.echo:
            print("[dry-run]", *call)

    def tap(self, keys: Sequence[str], repeat: int = 1):
        self._rec("tap", "+".join(keys), repeat)

    def scroll(self, dx: int, dy: int):
        self._rec("scroll", dx, dy)

    def move(self, x: int, y: int):
        self.calls.append(("move", x, y))  # too chatty to echo

    def click(self, button: str = "left", count: int = 1, at=None):
        self._rec("click", button, count, at)

    def press(self, button: str = "left"):
        self._rec("press", button)

    def release(self, button: str = "left"):
        self._rec("release", button)

    def type_text(self, text: str):
        self._rec("type", text)

    def screen_size(self):
        return self._size


class PynputBackend:
    def __init__(self):
        from pynput import keyboard, mouse  # imported lazily: needs a display server
        self._kb = keyboard.Controller()
        self._mouse = mouse.Controller()
        self._Key = keyboard.Key
        self._KeyCode = keyboard.KeyCode
        self._Button = mouse.Button
        self._size = screen_size()

    def _key(self, name: str):
        if len(name) == 1:
            return self._KeyCode.from_char(name)
        try:
            return getattr(self._Key, name)
        except AttributeError as exc:
            raise ValueError(f"unknown key '{name}'") from exc

    def tap(self, keys: Sequence[str], repeat: int = 1):
        resolved = [self._key(k) for k in keys]
        for _ in range(max(1, repeat)):
            for k in resolved:
                self._kb.press(k)
            for k in reversed(resolved):
                self._kb.release(k)

    def scroll(self, dx: int, dy: int):
        self._mouse.scroll(dx, dy)

    def move(self, x: int, y: int):
        self._mouse.position = (int(x), int(y))

    def click(self, button: str = "left", count: int = 1, at=None):
        if at is not None:
            self._mouse.position = at
        self._mouse.click(getattr(self._Button, button), count)

    def press(self, button: str = "left"):
        self._mouse.press(getattr(self._Button, button))

    def release(self, button: str = "left"):
        self._mouse.release(getattr(self._Button, button))

    def type_text(self, text: str):
        self._kb.type(text)

    def screen_size(self):
        return self._size
