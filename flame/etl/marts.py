"""Cross-source marts: script logs against PI tables, and extinction points for solids."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SUSTAINED = {"flow_terminated": True, "aborted_temperature_limit": True, "not_sustained": False,
             "self_extinguished": False, "no_ignition": False}


def _key(s: pd.Series) -> pd.Series:
    return s.str.replace(" ", "", regex=False).str.upper()


def join_logs(tests: pd.DataFrame, camp: pd.DataFrame, meas: pd.DataFrame, test_tables: list[str], log_table: str) -> pd.DataFrame:
    t = tests[tests.table_id.isin(test_tables)].assign(key=lambda d: _key(d.test_label))
    lg = camp[camp.table_id == log_table].assign(key=lambda d: _key(d.test_label))
    dur = meas[meas.variable == "ignition_to_end"].set_index("record_uid").value_num
    lg = lg.assign(log_ignition_to_end_s=lg.record_uid.map(dur))[["key", "record_uid", "outcome", "log_ignition_to_end_s", "test_date"]]
    lg.columns = ["key", "log_record_uid", "log_outcome", "log_ignition_to_end_s", "log_test_date"]
    return t.merge(lg, on="key", how="left").drop(columns="key")


def validate_log_rule(joined: pd.DataFrame) -> dict:
    """Agreement between the script-log outcome rule and the outcome written by the PI (CFI-G)."""
    j = joined[joined.log_outcome.notna() & joined.outcome.notna()]
    pi = j.outcome.map(SUSTAINED)
    lg = j.log_outcome.map(SUSTAINED)
    tab = pd.crosstab(j.outcome, j.log_outcome)
    d = (j.log_ignition_to_end_s - j.burn_time_s).dropna()
    return {
        "n_tests": int(len(joined)), "n_with_log": int(joined.log_record_uid.notna().sum()), "n_compared": int(len(j)),
        "binary_agreement_pct": round(100 * float((pi == lg).mean()), 1), "binary_agreement_n": [int((pi == lg).sum()), int(len(j))],
        "log_says_not_sustained": {"n": int((~lg).sum()), "pi_agrees": int(((~lg) & (~pi)).sum())},
        "log_says_sustained": {"n": int(lg.sum()), "pi_agrees": int((lg & pi).sum())},
        "pi_no_ignition_seen_by_log_as": {k: int(v) for k, v in j[j.outcome == "no_ignition"].log_outcome.value_counts().items()},
        "crosstab_pi_rows_log_cols": {str(r): {str(c): int(tab.loc[r, c]) for c in tab.columns} for r in tab.index},
        "log_window_minus_pi_burn_time_s": {"n": int(len(d)), "median": float(d.median()), "q25": float(d.quantile(.25)), "q75": float(d.quantile(.75))},
        "reading": "The script detects that the flame is gone; it cannot tell a failed ignition from a self-extinction, hence 'not_sustained'. "
                   "The log window starts at the ignition command, so it is longer than the PI burn time.",
    }


def _calib_curve(fan: pd.DataFrame, date: str | None, restrictor: int | None):
    f = fan[fan.air_velocity_cm_s.notna()].copy()
    f["d"] = pd.to_datetime(f.calibration_date, errors="coerce")
    f = f[f.d.notna()]
    if restrictor is not None:
        f = f[f.flow_restrictor == restrictor]
    if f.empty or not date:
        return None
    gap = (f.d - pd.Timestamp(date)).dt.days.abs()
    best = f[gap == gap.min()]
    best = best[best.calibration_label == best.calibration_label.iloc[0]]
    return best.sort_values("fan_display"), int(gap.min())


def fan_to_velocity(fan: pd.DataFrame, fan_display: float | None, date: str | None, restrictor: int | None) -> dict:
    """Interpolate duct air speed from a fan display reading, using the calibration closest in date."""
    out = {"velocity_from_fan_cm_s": None, "calibration_date": None, "calibration_gap_days": None, "calibration_note": None}
    if fan_display is None or pd.isna(fan_display):
        return out
    got = _calib_curve(fan, date, restrictor)
    if got is None:
        out["calibration_note"] = "no calibration for this restrictor / date"
        return out
    curve, gap = got
    col = "pot_setting" if fan_display < 20 else "fan_display"
    x, y = curve[col].to_numpy(float), curve.air_velocity_cm_s.to_numpy(float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    out.update(calibration_date=curve.calibration_date.iloc[0], calibration_gap_days=gap)
    if len(x) < 2:
        out["calibration_note"] = "calibration has fewer than 2 points"
        return out
    if fan_display < x.min() - 1 or fan_display > x.max() + 1:
        # no extrapolation: a reading outside the calibrated range (1 display unit of tolerance) has no speed
        out["calibration_note"] = f"reading outside the calibrated range ({x.min():g} to {x.max():g}); not converted"
        return out
    elif col == "pot_setting":
        out["calibration_note"] = "reading is a potentiometer setting"
    out["velocity_from_fan_cm_s"] = round(float(np.interp(fan_display, x, y)), 2)
    return out


def bass2_extinction_points(tests: pd.DataFrame, meas: pd.DataFrame, fan: pd.DataFrame | None, video: pd.DataFrame | None) -> pd.DataFrame:
    """One row per (test, evidence source) where the flame is reported to go out at a known flow setting."""
    s = tests[tests.table_id == "bass2"].set_index("record_uid")
    rawj = s.raw_json.map(json.loads)
    restr = meas[meas.variable == "flow_restrictor"].set_index("record_uid").value_text
    rows = []

    def base(uid):
        r = s.loc[uid]
        k = restr.get(uid)
        return {"record_uid": uid, "test_label": r.test_label, "pi": r.pi, "test_date": r.test_date, "fuel": r.fuel, "geometry": r.geometry,
                "thickness_mm": r.thickness_mm, "char_diameter_mm": r.char_diameter_mm, "o2_mole_frac": r.o2_mole_frac,
                "flow_restrictor": int(k) if isinstance(k, str) and k.isdigit() else None}

    for uid, d in rawj.items():
        air, fanc = d.get("Air display", ""), d.get("Fan display", "")
        if not re.search(r"ext", air + " " + fanc, re.I):
            continue
        def at_ext(var: str, cell: str):
            """Value of the sequence element that carries the word 'ext'; None when the cell only says the flow was lowered until extinction."""
            if not re.search("ext", cell, re.I):
                return None, None
            el = meas[(meas.record_uid == uid) & (meas.variable == var)].sort_values("seq_index")
            hit = el[el.value_text.str.contains("ext", case=False)]
            if hit.empty:
                return None, "extinction word not attached to a sequence element"
            h = hit.iloc[0]
            if re.search(r"down|decr|slow", h.value_text, re.I):
                return None, "lowered continuously to extinction; value at extinction not written"
            if pd.notna(h.value_num):
                return float(h.value_num), None
            prev = el[(el.seq_index < h.seq_index) & el.value_num.notna()]
            if len(prev) and not re.search(r"down|decr|slow", str(prev.value_text.iloc[-1]), re.I):
                return float(prev.value_num.iloc[-1]), "last step before the extinction word"
            return None, "value at extinction not written"

        b = base(uid)
        av, an = at_ext("flow_velocity_seq", air)
        fv, fn = at_ext("fan_display_seq", fanc)
        b.update(source="test matrix cell", air_at_extinction_cm_s=av, fan_at_extinction=fv, extinction_time_s=None,
                 evidence="; ".join(x for x in (f"Air display: {air}" if re.search("ext", air, re.I) else "",
                                               f"Fan display: {fanc}" if re.search("ext", fanc, re.I) else "", an or "", fn or "") if x))
        rows.append(b)
    if video is not None:
        for v in video.itertuples():
            if v.record_uid not in s.index:
                continue
            for src, fan_v, air_v, t in (("video: radiometer decay", v.fan_at_rad_decay, v.air_at_rad_decay_cm_s, v.rad_decay_onset_s),
                                         ("video: camera", v.fan_at_camera_extinction, v.air_at_camera_extinction_cm_s, v.camera_extinction_s)):
                # the camera loses faint blue flames early; keep its point only where the video session marks it reliable
                if src == "video: camera" and (not bool(getattr(v, "camera_reliable", True)) or pd.isna(t)):
                    continue
                if src == "video: radiometer decay":
                    kind = getattr(v, "end_of_test_kind", None)
                    if pd.isna(t) or kind == "video_ends_before_fall":
                        continue          # no fall, or the video ends before it: not an extinction
                    if kind == "fall_after_fan_step_up":
                        src = "video: blown out by fan step-up"   # the operator raised the fan; not a low-flow limit
                b = base(v.record_uid)
                b.update(source=src, air_at_extinction_cm_s=None if pd.isna(air_v) else float(air_v),
                         fan_at_extinction=None if pd.isna(fan_v) else float(fan_v), extinction_time_s=None if pd.isna(t) else float(t),
                         evidence="; ".join(json.loads(v.notes)) if v.notes else "")
                rows.append(b)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if fan is not None:
        conv = [fan_to_velocity(fan, r.fan_at_extinction, r.test_date, r.flow_restrictor) for r in out.itertuples()]
        out = pd.concat([out, pd.DataFrame(conv)], axis=1)
    return out.sort_values(["record_uid", "source"], kind="stable").reset_index(drop=True)


def video_measurements(video: pd.DataFrame | None, tests: pd.DataFrame, variables: dict, columns: tuple) -> pd.DataFrame:
    """Values read from the BASS-II videos by flame.video, in the long format of measurements.parquet.

    Kept as a mart, not appended to measurements.parquet: these values come from our reader of the video overlay,
    not from a table published on PSI. Every row carries the flag 'from_video'. Nothing is computed here.
    """
    rows = []
    if video is None:
        return pd.DataFrame(rows, columns=list(columns))
    known = set(tests.record_uid)

    def add(uid, var, value, src, seq=0, text=None, flags=()):
        if value is None or pd.isna(value):
            return
        v = variables[var]
        rows.append({"record_uid": uid, "variable": var, "category": v["category"], "seq_index": seq, "value_num": float(value),
                     "value_lo": None, "value_hi": None, "qualifier": "eq", "unit": v.get("unit"), "value_text": text, "unit_raw": None,
                     "source_column": f"video_summary.{src}", "flags": ["from_video", *flags]})

    for v in video.itertuples():
        if v.record_uid not in known:
            continue
        sat = ("radiometer_saturated",) if bool(getattr(v, "rad_saturated", False)) else ()
        for i, pl in enumerate(json.loads(v.fan_plateaus) if v.fan_plateaus else []):
            span = f"t = {pl['t_start_s']:g} to {pl['t_end_s']:g} s"
            add(v.record_uid, "fan_display_video_seq", pl.get("fan_display"), "fan_plateaus", i, span)
            add(v.record_uid, "flow_velocity_video_seq", pl.get("flow_cm_s"), "fan_plateaus", i, span)
        cens = ("censored",) if bool(getattr(v, "rad_decay_censored", False)) else ()
        add(v.record_uid, "radiometer_decay_onset", v.rad_decay_onset_s, "rad_decay_onset_s", flags=sat + cens)
        add(v.record_uid, "flow_at_radiometer_decay", getattr(v, "flow_at_rad_decay_cm_s", None), "flow_at_rad_decay_cm_s", flags=cens)
        if bool(getattr(v, "camera_reliable", False)):
            add(v.record_uid, "camera_extinction_time", v.camera_extinction_s, "camera_extinction_s")
        add(v.record_uid, "video_duration", v.duration_s, "duration_s")
    out = pd.DataFrame(rows, columns=list(columns))
    return out.sort_values(["record_uid", "variable", "seq_index"], kind="stable").reset_index(drop=True)


def load_video_summary(out_dir: Path) -> pd.DataFrame | None:
    p = out_dir / "video" / "video_summary.parquet"
    if not p.exists():
        return None
    v = pd.read_parquet(p)
    # the video session marks which files passed its reading checks (clock self-check, overlay date): keep only those
    if "video_accepted" in v.columns:
        v = v[v["video_accepted"].fillna(False).astype(bool)].reset_index(drop=True)
    return v
