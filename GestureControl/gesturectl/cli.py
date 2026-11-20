"""Command-line interface: `gesturectl <command>` (or `python -m gesturectl`)."""
from __future__ import annotations

import argparse
import logging
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

from . import __version__
from .config import Bindings, ConfigError, load_config

WINDOW = "gesturectl"


def _build(cfg, dry_run: bool, synchronous: bool = False):
    from .actions import ActionRunner
    from .actions.backend import DryRunBackend, PynputBackend
    from .custom import CustomGestureModel
    from .engine import Engine

    backend = DryRunBackend(echo=True) if dry_run else PynputBackend()
    runner = ActionRunner(backend, allow_run=cfg["actions"]["allow_run"], synchronous=synchronous)
    custom = CustomGestureModel.load(cfg["custom_gestures"]) if cfg.get("custom_gestures") else None
    return Engine(cfg, runner, custom), runner


def _tracker(cfg, mode="video"):
    from .tracker import HandTracker

    tr = cfg["tracking"]
    return HandTracker(tr.get("model_path"), tr["max_hands"], tr["min_detection_confidence"],
                       tr["min_presence_confidence"], tr["min_tracking_confidence"], mode=mode,
                       mirrored_input=cfg["camera"]["mirror"] if mode == "video" else False)


def _camera(cfg):
    from .camera import Camera

    c = cfg["camera"]
    return Camera(c["index"], c["width"], c["height"], c["fps"], c["mirror"])


# ---------------------------------------------------------------- commands
def cmd_run(args, cfg) -> int:
    import cv2

    from .hud import HUD
    from .recorder import Recorder

    if args.mode:
        cfg["control"]["start_mode"] = args.mode
    if args.camera is not None:
        cfg["camera"]["index"] = args.camera
    if args.paused:
        cfg["control"]["start_paused"] = True

    try:
        cam = _camera(cfg)  # open the camera first: cheapest thing to fail
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    engine, runner = _build(cfg, args.dry_run)
    tracker = _tracker(cfg)
    hud = HUD()
    rec = Recorder(args.record) if args.record else None
    show = not args.no_preview
    if show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, args.preview_width, int(args.preview_width * 9 / 16))
        try:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
    print(f"gesturectl {__version__} — mode '{engine.mode}'"
          f"{' (DRY RUN: no real input is sent)' if args.dry_run else ''}. "
          "Show your hand to the camera. Ctrl+C or 'q' in the preview to quit.")

    seq = -1
    try:
        while True:
            seq, t, frame = cam.read(seq)
            if frame is None:
                continue
            hands = tracker.process(frame, t)
            if rec:
                rec.write(t, hands)
            events = engine.update(hands, t)
            hud.tick()
            if not show:
                for ev in events:
                    print(f"[{time.strftime('%H:%M:%S')}] {ev}")
                continue
            latency = (time.monotonic() - t) * 1000
            cv2.imshow(WINDOW, hud.draw(frame, engine, latency))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("h"):
                hud.show_help = not hud.show_help
            elif key == ord("m"):
                engine.next_mode()
            elif key == ord("p"):
                engine.paused = not engine.paused
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        engine._release_mouse()
        runner.close()
        cam.close()
        tracker.close()
        if rec:
            rec.close()
        if show:
            cv2.destroyAllWindows()
    return 0


def cmd_replay(args, cfg) -> int:
    from .recorder import read_session

    engine, runner = _build(cfg, dry_run=True, synchronous=True)
    runner.backend.echo = False
    counts: Counter = Counter()
    t0 = None
    for t, hands in read_session(args.file):
        t0 = t if t0 is None else t0
        for ev in engine.update(hands, t):
            counts[ev.action] += 1
            print(f"{t - t0:8.2f}s  {ev}")
    print("\nsummary:", dict(counts) or "no actions")
    return 0


def cmd_train(args, cfg) -> int:
    import cv2
    import numpy as np

    from .custom import CustomGestureModel, embed
    from .hud import _text

    path = cfg["custom_gestures"]
    model = CustomGestureModel.load(path)
    if args.list:
        for label, n in Counter(model.y.tolist()).items():
            print(f"custom:{label:20s} {n} samples")
        return 0
    if args.remove:
        n = model.remove(args.name)
        model.save(path)
        print(f"removed {n} samples of '{args.name}'")
        return 0

    tracker, cam = _tracker(cfg), _camera(cfg)
    samples, seq = [], -1
    start = time.monotonic()
    try:
        while len(samples) < args.samples:
            seq, t, frame = cam.read(seq)
            if frame is None:
                continue
            hands = tracker.process(frame, t)
            countdown = 3 - (time.monotonic() - start)
            if countdown <= 0 and hands:
                hand = max(hands, key=lambda h: np.ptp(h.norm[:, 1]))
                samples.append(embed(hand))
            msg = (f"Get ready: show '{args.name}' in {countdown:.0f}" if countdown > 0 else
                   f"Recording '{args.name}': {len(samples)}/{args.samples} — vary angle & distance")
            _text(frame, msg, (16, 36), 0.7, (0, 200, 255), 2)
            for h in hands:
                for p in (h.norm[:, :2] * frame.shape[1::-1]).astype(int):
                    cv2.circle(frame, tuple(p), 3, (255, 191, 0), -1)
            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                print("aborted")
                return 1
    finally:
        cam.close()
        tracker.close()
        cv2.destroyAllWindows()
    model.remove(args.name)  # re-training replaces old samples
    model.add(args.name, np.array(samples))
    model.save(path)
    print(f"saved {len(samples)} samples. Bind it with:  {{gesture: \"custom:{args.name}\", hold: 0.6, action: ...}}")
    return 0


def cmd_gestures(args, cfg) -> int:
    from .actions import describe
    from .poses import POSES

    print("Poses:")
    for k, v in POSES.items():
        if k != "none":
            print(f"  {k:12s} {v}")
    b = Bindings.from_config(cfg)
    for scope, lst in [("global (all modes)", b.global_)] + [(f"mode '{m}'", l) for m, l in b.modes.items()]:
        print(f"\n{scope}:")
        for bd in lst:
            extra = f"   [{describe(bd.actions[0])}]" if len(bd.actions) == 1 else ""
            print(f"  {bd.label()}{extra}")
    return 0


def cmd_annotate(args, cfg) -> int:
    import cv2

    from .hud import HUD

    img = cv2.imread(args.image)
    if img is None:
        print(f"cannot read {args.image}", file=sys.stderr)
        return 1
    engine, _ = _build(cfg, dry_run=True, synchronous=True)
    tracker = _tracker(cfg, mode="image")
    hands = tracker.process(img)
    tracker.close()
    # Feed the same frame a few times so the temporal smoother settles.
    for i in range(6):
        engine.update(hands, i * 0.033)
    for s in engine.visible:
        print(f"{s.key:6s} pose={s.pose:12s} fingers={[k for k, v in s.features.extended.items() if v]}")
    if not hands:
        print("no hands found")
    if args.output:
        cv2.imwrite(args.output, HUD().draw(img.copy(), engine))
        print(f"wrote {args.output}")
    return 0


def cmd_doctor(args, cfg) -> int:
    ok = True

    def line(status, name, detail=""):
        nonlocal ok
        ok &= status != "FAIL"
        print(f"[{status:4s}] {name}{(': ' + detail) if detail else ''}")

    line("OK", "python", sys.version.split()[0])
    line("OK", "platform", f"{platform.system()} {platform.release()}")
    try:
        import mediapipe
        line("OK", "mediapipe", mediapipe.__version__)
    except Exception as e:
        line("FAIL", "mediapipe", str(e))
    try:
        import cv2
        gui = "GUI: NONE" not in cv2.getBuildInformation().replace("  ", " ")
        line("OK" if gui else "WARN", "opencv", cv2.__version__ +
             ("" if gui else " (headless build: no preview window; pip install opencv-python)"))
    except Exception as e:
        line("FAIL", "opencv", str(e))
    try:
        from .tracker import ensure_model
        line("OK", "hand model", str(ensure_model(cfg["tracking"].get("model_path"))))
    except Exception as e:
        line("FAIL", "hand model", str(e))
    try:
        from .actions.backend import PynputBackend
        b = PynputBackend()
        line("OK", "input injection", f"pynput, screen {b.screen_size()[0]}x{b.screen_size()[1]}")
    except Exception as e:
        line("FAIL", "input injection", f"{e} (needs a desktop session)")
    if os.environ.get("XDG_SESSION_TYPE") == "wayland":
        line("WARN", "wayland", "synthetic input only reaches XWayland apps; log in with an "
             "X11/Xorg session for full control")
    if platform.system() == "Darwin":
        line("INFO", "macOS", "grant Camera + Accessibility permission to your terminal app "
             "(System Settings > Privacy & Security)")
    try:
        Bindings.from_config(cfg)
        line("OK", "config", "bindings valid")
    except ConfigError as e:
        line("FAIL", "config", str(e))
    if args.camera:
        try:
            cam = _camera(cfg)
            _, _, f = cam.read(timeout=3)
            cam.close()
            line("OK" if f is not None else "FAIL", "camera",
                 f"{f.shape[1]}x{f.shape[0]}" if f is not None else "no frames")
        except Exception as e:
            line("FAIL", "camera", str(e))
    return 0 if ok else 1


def cmd_download(args, cfg) -> int:
    from .tracker import ensure_model
    print(ensure_model(cfg["tracking"].get("model_path")))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gesturectl", description="Control your desktop with hand gestures.")
    p.add_argument("--config", help="YAML config (default: ~/.config/gesturectl/config.yaml)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="start gesture control (default)")
    r.add_argument("--dry-run", action="store_true", help="print actions instead of performing them")
    r.add_argument("--mode", help="start in this mode (desktop, reels, mouse, ...)")
    r.add_argument("--camera", help="webcam index, or a video file to replay")
    r.add_argument("--paused", action="store_true", help="start paused (hold shaka to arm)")
    r.add_argument("--no-preview", action="store_true", help="no camera window; log actions to console")
    r.add_argument("--preview-width", type=int, default=560)
    r.add_argument("--record", metavar="FILE", help="record landmarks to a .jsonl session")

    rp = sub.add_parser("replay", help="replay a recorded session through the engine (dry run)")
    rp.add_argument("file")

    t = sub.add_parser("train", help="teach a custom pose from your webcam")
    t.add_argument("name", nargs="?", default="")
    t.add_argument("--samples", type=int, default=120)
    t.add_argument("--remove", action="store_true")
    t.add_argument("--list", action="store_true")

    sub.add_parser("gestures", help="list poses and current bindings")
    a = sub.add_parser("annotate", help="detect hands in an image and draw the HUD")
    a.add_argument("image")
    a.add_argument("-o", "--output")
    d = sub.add_parser("doctor", help="check your setup")
    d.add_argument("--camera", action="store_true", help="also try to open the webcam")
    sub.add_parser("download-model", help="fetch the MediaPipe hand model")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd is None:
        args = p.parse_args(list(sys.argv[1:] if argv is None else argv) + ["run"])
    if args.cmd == "train" and not (args.name or args.list):
        p.error("train needs a NAME (or --list)")
    try:
        cfg = load_config(args.config)
        Bindings.from_config(cfg)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    os.environ.setdefault("GLOG_minloglevel", "2")  # quieten MediaPipe's C++ logging
    return {
        "run": cmd_run, "replay": cmd_replay, "train": cmd_train, "gestures": cmd_gestures,
        "annotate": cmd_annotate, "doctor": cmd_doctor, "download-model": cmd_download,
    }[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
