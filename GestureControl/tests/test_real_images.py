"""Full pipeline on real photographs: MediaPipe -> features -> pose.

Uses MediaPipe's public sample images. They are downloaded to the cache on
first run; the test is skipped when offline or when MediaPipe cannot load.
"""
import os
import urllib.request
from pathlib import Path

import pytest

CACHE = Path(os.environ.get("GESTURECTL_CACHE", Path.home() / ".cache" / "gesturectl")) / "test-images"
BASE = "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/"
CASES = {"thumbs_up.jpg": "thumbs_up", "thumbs_down.jpg": "thumbs_down",
         "victory.jpg": "peace", "pointing_up.jpg": "point"}


def _fetch(name):
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(BASE + name, path)
        except Exception as exc:
            pytest.skip(f"cannot download test image: {exc}")
    return path


@pytest.fixture(scope="module")
def tracker():
    try:
        from gesturectl.tracker import HandTracker
        tr = HandTracker(mode="image", mirrored_input=False)
    except Exception as exc:
        pytest.skip(f"MediaPipe unavailable: {exc}")
    yield tr
    tr.close()


@pytest.mark.parametrize("image, expected", CASES.items())
def test_real_photo_pose(tracker, image, expected):
    import cv2

    from gesturectl.features import FeatureExtractor
    from gesturectl.poses import classify

    hands = tracker.process(cv2.imread(str(_fetch(image))))
    assert len(hands) == 1
    assert classify(FeatureExtractor()(hands[0])) == expected
