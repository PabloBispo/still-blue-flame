"""Tests for the BASS video reader. Pure-function tests run anywhere; the
regression test on a real PSI-25 file runs only when BASS_VIDEO_DIR is set."""
import glob
import os

import numpy as np
import pandas as pd
import pytest

from flame.video import bass
from flame.video.summarize import camera_flame, fan_plateaus, rad_decay_onset, label_from_filename as parse_label


def test_label_accepts_both_spellings():
    assert parse_label("PSI-25_Raw Data_BASS_GMT49_SAMPLE_207_TEST_O1.mp4") == "O1"
    assert parse_label("PSI-25_Raw Data_BASS_GMT183_SAMPLE_150_TESTB16.mp4") == "B16"
    assert parse_label("PSI-25_Raw Data_BASS_GMT164_SAMPLE_205_TEST_O3b.mp4") == "O3b"
    assert parse_label("PSI-25_Raw Data_Fan calibration data.xlsx") is None


def test_fan_plateaus_ignores_short_transitions():
    fan = [89] * 20 + [75, 68] + [62] * 15 + [61] * 3 + [50] * 10
    sec = pd.DataFrame({"t_s": range(len(fan)), "fan_display": fan})
    got = [(p["fan_display"], p["t_start_s"], p["t_end_s"]) for p in fan_plateaus(sec)]
    assert got == [(89.0, 0, 19), (62.0, 22, 39), (50.0, 40, 49)]


def test_rad_decay_onset_skips_igniter_spike_and_slow_drift():
    rad = [0.0] * 5 + [1.4, 1.3, 0.9] + [1.0] * 20 + list(np.linspace(1.0, 0.95, 10)) + [0.8, 0.6, 0.45, 0.3, 0.25, 0.2, 0.2, 0.2]
    sec = pd.DataFrame({"t_s": range(len(rad)), "rad": rad})
    onset = rad_decay_onset(sec)
    assert 36 <= onset <= 39          # the fall starts at t = 38


def test_rad_decay_onset_none_when_flat():
    sec = pd.DataFrame({"t_s": range(40), "rad": [1.0] * 40})
    assert rad_decay_onset(sec) is None


def test_camera_flame_drops_igniter_flash_and_lights_on():
    n = 60
    dim = np.full(n, 30.0); lit = np.zeros(n); bg = np.zeros(n)
    dim[:8] = 150000; bg[:8] = 26          # chamber lights on
    dim[12:15] = 5000                      # igniter flash, 3 s
    dim[20:45] = 900; lit[20:45] = 700     # flame
    sec = pd.DataFrame({"t_s": range(n), "dim_px": dim, "lit_px": lit, "background": bg})
    cam = camera_flame(sec)
    assert cam["camera_runs"] == [(20, 44)]
    assert cam["camera_extinction_s"] == 44.0 and cam["camera_flame_kind"] == "bright"


def test_minus_sign_and_shift():
    cell = np.full((1, 13, 24), 22, np.uint8)
    cell[0, 5:8, 2:22] = 140
    assert bass._is_minus(cell)[0]
    digit = np.full((1, 13, 24), 22, np.uint8)
    digit[0, 0:13, 10:14] = 150
    assert not bass._is_minus(digit)[0]
    t = np.zeros((10, 13, 24), np.float32); t[:, :, 5] = 1
    assert bass._shifted(t, 2)[0, 0, 7] == 1 and bass._shifted(t, -2)[0, 0, 3] == 1


VIDEO_DIR = os.environ.get("BASS_VIDEO_DIR")


@pytest.mark.skipif(not VIDEO_DIR, reason="set BASS_VIDEO_DIR to the PSI-25 Raw Data folder")
def test_o1_regression():
    path = glob.glob(os.path.join(VIDEO_DIR, "*TEST_O1.mp4"))[0]
    r = bass.read_video(path)
    assert r.clock_selfcheck > 0.99 and r.clock_step_ok > 0.99
    f = pd.DataFrame(r.frames)
    row = f.iloc[5996]                     # frame read by eye: FUEL -2, AIR 0, FAN 40, RAD 0.73, 16:30:25
    assert (row.fuel, row.air_cm_s, row.fan_display, round(row.rad, 2)) == (-2, 0, 40, 0.73)
    assert row.clock_tod_s == 16 * 3600 + 30 * 60 + 25


def _cal():
    return pd.DataFrame({
        "calibration_date": ["2014-02-18"] * 3 + ["2014-06-23"] * 3,
        "calibration_label": ["a"] * 3 + ["b"] * 3,
        "flow_restrictor": [2] * 6,
        "fan_display": [38.0, 55.0, 89.0, 38.0, 42.0, 55.0],
        "air_velocity_cm_s": [0.0, 7.5, 20.0, 0.0, 3.0, 8.0],
    })


def test_fan_to_flow_uses_closest_date_and_does_not_extrapolate():
    from flame.video.units import add_flow
    sec = pd.DataFrame({"t_s": range(4), "fan_display": [89.0, 38.0, 37.5, 122.0]})
    out, meta = add_flow(sec, _cal(), "2014-02-19", 2)
    assert meta["flow_calibration_date"] == "2014-02-18" and meta["flow_calibration_gap_days"] == 1
    assert out.flow_cm_s.iloc[0] == 20.0 and out.flow_cm_s.iloc[1] == 0.0
    assert out.flow_cm_s.iloc[2] == 0.0            # within the 1-unit tolerance of a stopped fan
    assert np.isnan(out.flow_cm_s.iloc[3]) and bool(out.flow_out_of_range.iloc[3])
    out2, meta2 = add_flow(sec, _cal(), "2014-06-20", 2)
    assert meta2["flow_calibration_date"] == "2014-06-23"
    assert np.isnan(out2.flow_cm_s.iloc[0])        # 89 is above that day's calibrated range


def test_no_calibration_for_restrictor_gives_null_flow():
    from flame.video.units import add_flow
    sec = pd.DataFrame({"t_s": range(2), "fan_display": [50.0, 60.0]})
    out, meta = add_flow(sec, _cal(), "2014-02-18", 1)
    assert out.flow_cm_s.isna().all() and meta["flow_calibration_date"] is None
