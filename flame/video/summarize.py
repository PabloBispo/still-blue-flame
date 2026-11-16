"""Per-test summary from the frame-level BASS video table.

Three independent signals are kept apart on purpose, because they do not
measure the same thing and do not always agree:

* camera: lit pixels below the overlay. Blind to dim blue flames near the
  quenching limit (the video camera does not see them; the still camera does).
* radiometer (RAD on the overlay): responds to flame and to the hot sample,
  saturates at 4.10 on the display.
* operator input (FAN on the overlay): the commanded flow steps.

Nothing here converts FAN display units to cm/s. That needs the PSI-25 fan
calibration sheet.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

LIGHTS_ON_BG = 10.0      # median ROI level above this means the chamber lights are on
DIM_MARGIN_PX = 150      # camera sees a flame when dim_px exceeds the dark baseline by this much
MIN_VISIBLE_RUN_S = 5    # shorter lit runs are igniter flashes, not a flame
RAD_SATURATION = 4.10
ONSET_SLOPE_SHARE = 0.15  # the fall starts where the slope reaches this share of its steepest value
LABEL_RE = re.compile(r"TEST_?([A-Za-z]{1,2}\d+[a-z]?)", re.IGNORECASE)


FAN_JUMP_WINDOW = (60, 10)   # seconds before the radiometer fall used as the "before" fan level
FAN_JUMP_MIN = 5.0           # display units; a rise of this size or more just before the fall is a blow-out


def label_from_filename(file_name: str) -> str | None:
    """Test label from a PSI-25 video file name. Accepts TEST_B16 and TESTB16."""
    m = LABEL_RE.search(file_name)
    return m.group(1) if m else None


def per_second(frames: pd.DataFrame) -> pd.DataFrame:
    """Median overlay values and mean camera signal per second of video."""
    f = frames.assign(t_s=np.floor(frames["t_s"]).astype(int))
    g = f.groupby("t_s")
    out = g.agg(
        fuel=("fuel", "median"), air_cm_s=("air_cm_s", "median"), fan_display=("fan_display", "median"),
        rad=("rad_1dp", "median"), rad_2dp=("rad", "median"), clock_tod_s=("clock_tod_s", "median"),
        dim_px=("dim_px", "mean"), lit_px=("lit_px", "mean"), core_px=("core_px", "mean"),
        blue_share=("blue_share", "mean"), background=("background", "median"),
        flame_area_px=("flame_area_px", "median"), flame_radius_px=("flame_radius_px", "median"),
        flame_cx=("flame_cx", "median"), flame_cy=("flame_cy", "median"),
        flame_w=("flame_w", "median"), flame_h=("flame_h", "median"),
        core_area_px=("core_area_px", "median"), peak_level=("peak_level", "median"),
        head_x=("head_x", "median"), head_y=("head_y", "median"), head_radius_px=("head_radius_px", "median"),
    ).reset_index()
    # best radiometer value: two decimals where the cut digit was readable, else one
    out["rad_best"] = out["rad_2dp"].fillna(out["rad"])
    return out


def fan_plateaus(sec: pd.DataFrame, min_s: int = 5, tol: float = 1.0) -> list[dict]:
    """Runs where the FAN display stays within `tol` of the run's first value for at least `min_s` seconds."""
    runs, cur = [], None
    for t, v in zip(sec["t_s"], sec["fan_display"]):
        if not np.isfinite(v):
            continue
        if cur is not None and abs(v - cur["_first"]) <= tol:
            cur["t_end_s"] = int(t)
            cur["_vals"].append(v)
        else:
            cur = {"_first": v, "_vals": [v], "t_start_s": int(t), "t_end_s": int(t)}
            runs.append(cur)
    out = []
    for r in runs:
        if r["t_end_s"] - r["t_start_s"] + 1 >= min_s:
            out.append({"fan_display": float(np.median(r["_vals"])), "t_start_s": r["t_start_s"], "t_end_s": r["t_end_s"]})
    return out


def rad_decay_onset(sec: pd.DataFrame, span: int = 2) -> int | None:
    """Second at which the radiometer starts its final fall.

    Looks only at the final descent (the tail after which RAD never recovers),
    finds the steepest point, and walks back while the slope stays above a
    share (ONSET_SLOPE_SHARE) of that steepest slope. This ignores the igniter spike and a slow
    drift before the fall.
    """
    col = "rad_best" if "rad_best" in sec else "rad"
    r = sec[col].interpolate(limit_direction="both").to_numpy(float)
    n = len(r)
    if n < 4 * span + 2:
        return None
    sm = pd.Series(r).rolling(3, center=True, min_periods=1).mean().to_numpy()
    slope = np.full(n, 0.0)
    slope[span:n - span] = sm[2 * span:] - sm[:n - 2 * span]
    # start of the final descent: walk back from the end while the running maximum keeps (weakly) rising
    tail = n - 1
    runmax = sm[-1]
    for i in range(n - 2, -1, -1):
        if sm[i] + 0.05 < runmax:
            break
        runmax = max(runmax, sm[i])
        tail = i
    seg = slope[tail:]
    if seg.size == 0 or seg.min() >= -0.05:
        return None
    k = tail + int(np.argmin(seg))
    limit = ONSET_SLOPE_SHARE * slope[k]
    while k > tail and slope[k - 1] <= limit:
        k -= 1
    return int(sec["t_s"].iloc[min(k + 1, n - 1)])


def camera_flame(sec: pd.DataFrame, frames: pd.DataFrame | None = None) -> dict:
    """What the video camera sees: visible seconds and the end of the last sustained lit run."""
    dark = sec["background"] < LIGHTS_ON_BG
    base = float(sec.loc[dark, "dim_px"].min()) if dark.any() else 0.0   # darkest second = noise floor
    on = (dark & (sec["dim_px"] > base + DIM_MARGIN_PX)).to_numpy()
    on = pd.Series(on).rolling(5, center=True, min_periods=1).mean().to_numpy() > 0.5
    t = sec["t_s"].to_numpy()
    runs, start = [], None
    for i, v in enumerate(np.append(on, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= MIN_VISIBLE_RUN_S:
                runs.append((int(t[start]), int(t[i - 1])))
            start = None
    out = {"camera_baseline_px": base, "camera_visible_s": int(sum(b - a + 1 for a, b in runs)),
           "camera_runs": runs, "camera_extinction_s": None, "camera_censored": False, "camera_flame_kind": None}
    if runs:
        a, b = runs[-1]
        out["camera_censored"] = bool(b >= t[-1] - 1)
        ext = None if out["camera_censored"] else float(b)
        if ext is not None and frames is not None:      # refine to the last lit frame around the end of the run
            w = frames[(frames["t_s"] >= b - 3) & (frames["t_s"] <= b + 8) & (frames["dim_px"] > base + DIM_MARGIN_PX)]
            if len(w):
                ext = round(float(w["t_s"].iloc[-1]), 2)
        out["camera_extinction_s"] = ext
        last = sec[(sec["t_s"] >= max(a, b - 10)) & (sec["t_s"] <= b)]
        out["camera_flame_kind"] = "bright" if last["lit_px"].mean() > 500 else "dim"
    return out


def summarize(record_uid: str, frames: pd.DataFrame, test_date: str | None = None,
              sec: pd.DataFrame | None = None, flow_meta: dict | None = None) -> dict:
    """`sec` may be passed in when it already carries `flow_cm_s` (see flame.video.units)."""
    sec = per_second(frames) if sec is None else sec
    has_flow = "flow_cm_s" in sec
    plateaus = fan_plateaus(sec)
    onset = rad_decay_onset(sec)
    cam = camera_flame(sec, frames)

    def at(t, col):
        if t is None:
            return None
        v = sec.loc[sec["t_s"] == int(t), col]
        return float(v.iloc[0]) if len(v) and np.isfinite(v.iloc[0]) else None

    # last FAN step-down before the radiometer starts to fall
    step = None
    if onset is not None:
        for a, b in zip(plateaus[:-1], plateaus[1:]):
            if b["t_start_s"] <= onset + 5 and b["fan_display"] < a["fan_display"]:
                step = {"t_s": b["t_start_s"], "from": a["fan_display"], "to": b["fan_display"]}

    tod0 = sec["clock_tod_s"].dropna()
    start = None
    if len(tod0):
        s = int(round(float((tod0 - sec.loc[tod0.index, "t_s"]).median()))) % 86400
        start = f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"
    mo, dd, yy = (frames[c].median() for c in ("clock_month", "clock_day", "clock_year"))
    clock_date = f"20{int(yy):02d}-{int(mo):02d}-{int(dd):02d}" if np.isfinite([mo, dd, yy]).all() else None
    date_offset = None
    if clock_date and test_date:
        date_offset = int((pd.Timestamp(clock_date) - pd.Timestamp(test_date)).days)

    rad_max = float(sec["rad_best"].max())
    notes = []
    # the video may end while the radiometer is still falling: then the onset is a lower bound, not an extinction
    censored = None
    if onset is not None:
        r_on = at(onset, "rad_best")
        r_end = float(sec["rad_best"].dropna().iloc[-3:].median())
        censored = bool(r_on is not None and r_on > 0 and r_end > 0.5 * r_on)
        if censored:
            notes.append("video ends before the radiometer has fallen to half: extinction not captured")
    # did the operator raise the fan just before the radiometer fell? Then the flame was blown out at the end of
    # the test and the flow at the fall is the new fan setting, not a low-flow extinction limit
    fan_jump = None
    if onset is not None:
        before = sec[(sec["t_s"] >= onset - FAN_JUMP_WINDOW[0]) & (sec["t_s"] < onset - FAN_JUMP_WINDOW[1])]["fan_display"].dropna()
        f_on = at(onset, "fan_display")
        if len(before) and f_on is not None:
            fan_jump = float(f_on - before.median())
    step_up = bool(fan_jump is not None and fan_jump >= FAN_JUMP_MIN)
    if step_up:
        notes.append("radiometer fell right after the fan was raised: blown out by the operator, not a low-flow extinction")
    if onset is None:
        end_kind = "no_fall_detected"
    elif censored:
        end_kind = "video_ends_before_fall"
    elif step_up:
        end_kind = "fall_after_fan_step_up"
    else:
        end_kind = "fall_at_steady_fan"
    cam["camera_last_visible_s"] = cam["camera_extinction_s"]
    cam["camera_reliable"] = True
    if onset is not None and (not cam["camera_runs"] or cam["camera_runs"][-1][1] < onset - 30):
        # the camera lost the flame long before the radiometer fell: its last lit frame is not an extinction
        notes.append("flame not visible to the video camera while the radiometer is still high; camera extinction not reported")
        cam["camera_reliable"] = False
        cam["camera_extinction_s"] = None
    cam_ext = cam["camera_extinction_s"]
    if onset is not None and cam_ext is not None and not (-5 <= cam_ext - onset <= 30):
        # e.g. the chamber lights come back on after the test: the last lit run is not the flame
        notes.append("camera and radiometer disagree on the end of the flame; camera extinction not reported")
        cam["camera_reliable"] = False
        cam["camera_extinction_s"] = cam_ext = None
    if has_flow:
        for p in plateaus:
            w = sec[(sec["t_s"] >= p["t_start_s"]) & (sec["t_s"] <= p["t_end_s"])]["flow_cm_s"]
            p["flow_cm_s"] = round(float(w.median()), 2) if w.notna().any() else None
    if rad_max >= RAD_SATURATION:
        notes.append("radiometer saturated at 4.10")
    return {
        "record_uid": record_uid,
        "duration_s": float(frames["t_s"].iloc[-1]),
        "clock_date": clock_date, "clock_start": start, "clock_minus_test_date_days": date_offset,
        "fan_plateaus": plateaus,
        "fan_first": plateaus[0]["fan_display"] if plateaus else None,
        "fan_last": plateaus[-1]["fan_display"] if plateaus else None,
        "air_max_cm_s": float(sec["air_cm_s"].max()), "air_min_cm_s": float(sec["air_cm_s"].min()),
        "rad_max": rad_max, "rad_saturated": bool(rad_max >= RAD_SATURATION),
        "rad_decay_onset_s": onset, "rad_decay_censored": censored,
        "fan_at_rad_decay": at(onset, "fan_display"), "air_at_rad_decay_cm_s": at(onset, "air_cm_s"),
        "fan_jump_at_rad_decay": fan_jump, "rad_decay_after_fan_step_up": step_up, "end_of_test_kind": end_kind,
        "last_fan_step_down": step,
        "rad_decay_lag_after_fan_step_s": (onset - step["t_s"]) if (step and onset is not None) else None,
        **cam,
        "fan_at_camera_extinction": at(cam_ext, "fan_display"),
        "air_at_camera_extinction_cm_s": at(cam_ext, "air_cm_s"),
        "flow_at_rad_decay_cm_s": at(onset, "flow_cm_s") if has_flow else None,
        "flow_at_camera_extinction_cm_s": at(cam_ext, "flow_cm_s") if has_flow else None,
        "flow_first_cm_s": plateaus[0].get("flow_cm_s") if plateaus else None,
        "flow_last_cm_s": plateaus[-1].get("flow_cm_s") if plateaus else None,
        **(flow_meta or {}),
        "notes": notes,
    }
