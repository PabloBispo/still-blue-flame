"""Physical units for the BASS-II video table: FAN display to duct air speed (cm/s).

The fan display is not a speed. PSI-25 publishes one calibration per operations
day and flow restrictor (display 38 is a stopped fan; the same display reads
different speeds on different days). This uses the calibration closest in date
for the test's restrictor, the same rule as `flame.etl.marts.fan_to_velocity`,
applied to a whole series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RANGE_TOLERANCE = 1.0   # display units; the stopped-fan reading flickers between 37 and 39


def calibration_curve(fan_cal: pd.DataFrame, date: str | None, restrictor: int | None):
    """(curve, gap_days) for the calibration closest in date, or None."""
    f = fan_cal[fan_cal["air_velocity_cm_s"].notna() & fan_cal["fan_display"].notna()].copy()
    f["d"] = pd.to_datetime(f["calibration_date"], errors="coerce")
    f = f[f["d"].notna()]
    if restrictor is not None:
        f = f[f["flow_restrictor"] == restrictor]
    if f.empty or not date:
        return None
    gap = (f["d"] - pd.Timestamp(date)).dt.days.abs()
    best = f[gap == gap.min()]
    best = best[best["calibration_label"] == best["calibration_label"].iloc[0]]
    return best.sort_values("fan_display"), int(gap.min())


def fan_to_flow(fan_display, curve: pd.DataFrame):
    """Air speed (cm/s) for an array of display readings, and a mask of readings outside the calibrated range."""
    x = curve["fan_display"].to_numpy(float)
    y = curve["air_velocity_cm_s"].to_numpy(float)
    v = np.asarray(fan_display, float)
    out = np.interp(v, x, y)
    oor = (v < x.min() - RANGE_TOLERANCE) | (v > x.max() + RANGE_TOLERANCE)
    out[~np.isfinite(v) | oor] = np.nan      # no extrapolation: a reading outside the calibrated range has no speed
    return out, oor


def add_flow(sec: pd.DataFrame, fan_cal: pd.DataFrame, date: str | None, restrictor: int | None) -> tuple[pd.DataFrame, dict]:
    """Add `flow_cm_s` to a per-second table. Returns the table and the calibration used."""
    meta = {"flow_calibration_date": None, "flow_calibration_label": None, "flow_calibration_gap_days": None,
            "flow_restrictor": restrictor, "flow_calibration_points": None, "flow_step_near_zero_cm_s": None}
    got = calibration_curve(fan_cal, date, restrictor)
    sec = sec.copy()
    if got is None:
        sec["flow_cm_s"] = np.nan
        sec["flow_out_of_range"] = False
        return sec, meta
    curve, gap = got
    if curve["fan_display"].nunique() < 2:
        sec["flow_cm_s"] = np.nan
        sec["flow_out_of_range"] = False
        meta.update(flow_calibration_date=str(curve["calibration_date"].iloc[0]), flow_calibration_gap_days=gap,
                    flow_calibration_points=int(len(curve)))
        return sec, meta
    flow, oor = fan_to_flow(sec["fan_display"], curve)
    sec["flow_cm_s"] = np.round(flow, 2)
    sec["flow_out_of_range"] = oor
    y = np.sort(curve["air_velocity_cm_s"].to_numpy(float))
    meta.update(flow_calibration_date=str(curve["calibration_date"].iloc[0]), flow_calibration_label=str(curve["calibration_label"].iloc[0]),
                flow_calibration_gap_days=gap, flow_calibration_points=int(len(curve)),
                flow_calibrated_display_range=[float(curve["fan_display"].min()), float(curve["fan_display"].max())],
                flow_out_of_range_share=round(float(np.mean(oor)), 3),
                # first calibrated speed above zero: below it the conversion is a straight line between two points
                flow_step_near_zero_cm_s=float(y[y > 0][0]) if (y > 0).any() else None)
    return sec, meta


def air_vs_fan(sec: pd.DataFrame, plateaus: list[dict], settle_s: int = 45) -> dict:
    """Compare the anemometer (AIR on the overlay) with the speed converted from FAN.

    AIR is slow (about one minute), so only seconds at least `settle_s` into a FAN plateau count.
    """
    keep = np.zeros(len(sec), bool)
    t = sec["t_s"].to_numpy()
    for p in plateaus:
        keep |= (t >= p["t_start_s"] + settle_s) & (t <= p["t_end_s"])
    d = (sec.loc[keep, "air_cm_s"] - sec.loc[keep, "flow_cm_s"]).dropna()
    if d.empty:
        return {"air_minus_flow_n_s": 0, "air_minus_flow_median_cm_s": None, "air_minus_flow_mae_cm_s": None}
    return {"air_minus_flow_n_s": int(len(d)), "air_minus_flow_median_cm_s": round(float(d.median()), 2),
            "air_minus_flow_mae_cm_s": round(float(d.abs().mean()), 2)}
