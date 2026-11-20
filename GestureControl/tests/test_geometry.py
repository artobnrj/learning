import numpy as np
import pytest

from gesturectl.custom import CustomGestureModel, embed
from gesturectl.features import FeatureExtractor
from gesturectl.filters import Hysteresis, OneEuroFilter
from gesturectl.poses import PoseSmoother, classify
from gesturectl.synthetic import POSE_FINGERS, make_hand

ALL_POSES = list(POSE_FINGERS) + ["pinch", "ok"]


@pytest.mark.parametrize("pose", ALL_POSES)
@pytest.mark.parametrize("handedness", ["Right", "Left"])
@pytest.mark.parametrize("size", [0.08, 0.18, 0.35])
@pytest.mark.parametrize("roll", [-25, 0, 25])
def test_pose_invariant_to_scale_rotation_and_hand(pose, handedness, size, roll):
    if pose.startswith("thumbs"):
        roll = roll / 2  # thumbs up/down are *defined* by orientation
    hand = make_hand(pose, size=size, roll=roll, handedness=handedness, center=(0.7, 0.4))
    assert classify(FeatureExtractor()(hand)) == pose


@pytest.mark.parametrize("pose", ALL_POSES)
def test_pose_robust_to_landmark_noise(pose):
    """~2 mm of Gaussian noise on every landmark, every frame.

    Single frames may misfire on short-segment fingers (the thumb), but the
    streaming pipeline (hysteresis + majority smoother) must stay locked on.
    """
    rng = np.random.default_rng(42)
    single = sum(classify(FeatureExtractor()(make_hand(pose, noise=0.025, rng=rng))) == pose
                 for _ in range(100))
    assert single >= 80, f"{pose}: single-frame {single}%"

    fe, sm = FeatureExtractor(), PoseSmoother()
    stream = [sm.update(classify(fe(make_hand(pose, noise=0.025, rng=rng)))) for _ in range(200)]
    assert sum(s == pose for s in stream[5:]) / 195 >= 0.97


def test_thumbs_rotated_too_far_is_not_thumbs_up():
    hand = make_hand("thumbs_up", roll=80)
    assert classify(FeatureExtractor()(hand)) == "fist"


def test_finger_hysteresis_prevents_flicker():
    h = Hysteresis(on=105, off=125, low_is_on=True)
    states = [h.update(v) for v in (150, 110, 100, 115, 120, 124, 130, 110)]
    assert states == [False, False, True, True, True, True, False, False]


def test_pose_smoother_needs_quorum():
    s = PoseSmoother(window=5, quorum=3)
    seq = ["fist"] * 5 + ["open_palm", "fist", "open_palm", "open_palm"]
    out = [s.update(x) for x in seq]
    assert out[4] == "fist"
    assert out[6] == "fist"          # a single blip is ignored
    assert out[-1] == "open_palm"    # a sustained change wins


def test_one_euro_smooths_jitter_but_tracks_motion():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=0.8, beta=4.0)  # shipped mouse defaults
    t = np.arange(0, 2, 1 / 30)
    noisy = 0.5 + rng.normal(0, 0.01, len(t))
    out = np.array([f(x, ti) for x, ti in zip(noisy, t)])
    # Perceived cursor shake = frame-to-frame movement.
    assert np.abs(np.diff(out[15:])).mean() < np.abs(np.diff(noisy[15:])).mean() / 4

    f.reset()
    ramp = np.linspace(0, 1, len(t))
    out = np.array([f(x, ti) for x, ti in zip(ramp, t)])
    assert abs(out[-1] - ramp[-1]) < 0.03  # low lag on a fast, steady move


def test_custom_knn_learns_and_rejects_unknown():
    rng = np.random.default_rng(1)
    model = CustomGestureModel()
    model.add("rock_on", np.array([embed(make_hand("rock", noise=0.02, rng=rng)) for _ in range(40)]))
    model.add("hang", np.array([embed(make_hand("shaka", noise=0.02, rng=rng)) for _ in range(40)]))
    assert model.predict(embed(make_hand("rock", noise=0.02, rng=rng)))[0] == "rock_on"
    assert model.predict(embed(make_hand("shaka", handedness="Left", size=0.3)))[0] == "hang"
    assert model.predict(embed(make_hand("open_palm")))[0] is None


def test_custom_model_roundtrip(tmp_path):
    model = CustomGestureModel()
    model.add("x", np.array([embed(make_hand("rock"))] * 5))
    model.save(tmp_path / "m.npz")
    loaded = CustomGestureModel.load(tmp_path / "m.npz")
    assert loaded.labels == ["x"] and len(loaded) == 5
