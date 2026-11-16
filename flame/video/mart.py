"""Join the video summary to the BASS-II rows of the gold dataset.

    python -m flame.video.mart --tests data/processed/tests.parquet \
        --video-dir data/processed/video --out data/processed/marts/bass2_video.parquet

One row per BASS-II test (all of them). `has_video` marks the tests whose video
was processed; video columns are null for the others.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

TEST_COLS = ["record_uid", "test_label", "test_date", "pi", "fuel_raw", "fuel", "geometry", "thickness_mm",
             "char_diameter_mm", "o2_mole_frac", "o2_mole_frac_final", "flow_velocity_cm_s", "flags"]
VIDEO_COLS = ["duration_s", "clock_date", "clock_start", "clock_minus_test_date_days", "fan_plateaus", "fan_first",
              "fan_last", "air_max_cm_s", "air_min_cm_s", "rad_max", "rad_saturated", "rad_decay_onset_s", "rad_decay_censored", "fan_jump_at_rad_decay", "rad_decay_after_fan_step_up",
              "end_of_test_kind", "video_accepted", "accept_note", "clock_selfcheck",
              "fan_at_rad_decay", "air_at_rad_decay_cm_s", "last_fan_step_down", "rad_decay_lag_after_fan_step_s",
              "camera_visible_s", "camera_extinction_s", "camera_reliable", "camera_flame_kind", "fan_at_camera_extinction",
              "flow_first_cm_s", "flow_last_cm_s", "flow_at_rad_decay_cm_s", "flow_at_camera_extinction_cm_s",
              "flow_calibration_date", "flow_calibration_gap_days", "flow_step_near_zero_cm_s",
              "air_minus_flow_n_s", "air_minus_flow_mae_cm_s", "notes"]


def build(tests: str, video_dir: str, gas: str | None = None) -> pd.DataFrame:
    t = pd.read_parquet(tests)
    t = t[(t["psi_id"] == "PSI-25") & (t["table_id"] == "bass2")][TEST_COLS]
    v = pd.read_parquet(os.path.join(video_dir, "video_summary.parquet"))[["record_uid"] + VIDEO_COLS]
    sec = pd.read_parquet(os.path.join(video_dir, "video_seconds.parquet"))
    # flame size in the 30 s before the radiometer starts to fall (steady burning, not the igniter)
    rows = []
    for uid, onset in zip(v["record_uid"], v["rad_decay_onset_s"]):
        s = sec[sec["record_uid"] == uid]
        w = s[(s["t_s"] >= onset - 30) & (s["t_s"] < onset)] if pd.notna(onset) else s.iloc[0:0]
        rows.append({"record_uid": uid,
                     "flame_area_px_before_fall": float(w["flame_area_px"].median()) if len(w) else np.nan,
                     "flame_radius_px_before_fall": float(w["flame_radius_px"].median()) if len(w) else np.nan})
    v = v.merge(pd.DataFrame(rows), on="record_uid", how="left")
    out = t.merge(v, on="record_uid", how="left")
    out.insert(1, "has_video", out["duration_s"].notna())
    out["video_accepted"] = out["video_accepted"].fillna(False).astype(bool)
    if gas and os.path.exists(gas):
        g = pd.read_parquet(gas)
        keep = [c for c in g.columns if c == "record_uid" or c not in out.columns]
        out = out.merge(g[keep], on="record_uid", how="left")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default="data/processed/tests.parquet")
    ap.add_argument("--video-dir", default="data/processed/video")
    ap.add_argument("--gas", default="data/processed/marts/bass2_gas.parquet")
    ap.add_argument("--out", default="data/processed/marts/bass2_video.parquet")
    a = ap.parse_args()
    m = build(a.tests, a.video_dir, a.gas)
    m.to_parquet(a.out, index=False)
    print(f"{len(m)} tests, {int(m.has_video.sum())} with video -> {a.out}")


if __name__ == "__main__":
    main()
