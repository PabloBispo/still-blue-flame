"""Export small JSON datasets for the sample dashboard (EDA + BASS-II video telemetry).

    uv run python -m flame.analysis.dash_export --data data/processed

Each file is {"rows": [...], "meta": [...]}: rows for the charts, meta saying how they were produced.
Reads only gold tables, marts and the video session's outputs; computes nothing that is not a count or a copy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CMD = "uv run python -m flame.analysis.dash_export --data data/processed"


def _meta(inputs: str, note: str) -> list[dict]:
    return [{"label": "Produced by", "value": CMD}, {"label": "Input", "value": inputs}, {"label": "What it is", "value": note}]


def _clean(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.replace({np.nan: None}).to_json(orient="records", double_precision=5))


def run(data: Path) -> dict[str, dict]:
    t = pd.read_parquet(data / "tests.parquet")
    c = pd.read_parquet(data / "campaigns.parquet")
    m = pd.read_parquet(data / "measurements.parquet")
    inv = json.loads((data / "investigations.json").read_text())
    out: dict[str, dict] = {}

    amb = t[(t.oxidizer_location == "ambient") & t.o2_mole_frac.notna()]
    solids = amb[amb.fuel_phase == "solid"]
    out["summary"] = {"rows": [{
        "tests": int(len(t)), "measurements": int(len(m)), "investigations": len(inv),
        "investigations_with_tests": int(t.psi_id.nunique()), "campaign_records": int(len(c)),
        "tests_with_outcome": int(t.has_outcome_label.sum()), "tests_gas": int((t.fuel_phase == "gas").sum()),
        "tests_liquid": int((t.fuel_phase == "liquid").sum()), "tests_solid": int((t.fuel_phase == "solid").sum()),
        "solid_o2_max_pct": round(100 * float(solids.o2_mole_frac.max()), 1),
        "solid_tests_o2_ge_30": int((solids.o2_mole_frac >= 0.2999).sum()), "solid_tests_with_o2": int(len(solids)),
    }], "meta": _meta("tests.parquet, campaigns.parquet, measurements.parquet, investigations.json", "Counts of the unified dataset.")}

    edges = np.arange(0.10, 0.4201, 0.02)
    rows = []
    for ph, d in amb.groupby("fuel_phase"):
        h, _ = np.histogram(d.o2_mole_frac.clip(upper=0.4199), bins=edges)
        rows += [{"fuel_phase": ph, "o2_from_pct": int(round(100 * a)), "o2_to_pct": int(round(100 * a)) + 2, "tests": int(n)} for a, n in zip(edges[:-1], h)]
    out["o2_hist"] = {"rows": rows, "meta": _meta("tests.parquet", "Tests per 2-point bin of chamber oxygen, by fuel phase. Ambient-oxidizer tests only.")}

    pts = amb.dropna(subset=["pressure_kpa"])
    g = (pts.assign(o2_pct=(100 * pts.o2_mole_frac).round(1), pressure_kpa=pts.pressure_kpa.round(0))
         .groupby(["fuel_phase", "o2_pct", "pressure_kpa"]).size().reset_index(name="tests"))
    out["o2_pressure"] = {"rows": _clean(g), "meta": _meta("tests.parquet", "Tests per (oxygen, pressure) condition, by fuel phase. Ambient-oxidizer tests with both values.")}
    out["targets"] = {"rows": [
        {"atmosphere": "ISS cabin", "o2_pct": 21.0, "pressure_kpa": 101.3, "basis": "21% O2 at 14.7 psia"},
        {"atmosphere": "Challenge statement", "o2_pct": 30.0, "pressure_kpa": 70.3, "basis": "30% O2 at 10.2 psia"},
        {"atmosphere": "Exploration atmosphere", "o2_pct": 34.0, "pressure_kpa": 56.5, "basis": "34% O2 at 8.2 psia"},
    ], "meta": [{"label": "What it is", "value": "Target habitat atmospheres named in the challenge statement and NASA exploration atmosphere work. Reference values, not data."}]}

    rows = []
    cf = pd.read_parquet(data / "marts/model_cfig.parquet")
    cf = cf[(cf.flame_type == "normal") & cf.outcome.notna()]
    cf_group = cf.outcome.map({"self_extinguished": "Not sustained", "no_ignition": "Not sustained", "flow_terminated": "Sustained"})
    for (band, grp), d in cf.groupby([np.where(cf.o2_mole_frac < 0.30, "Below 30% O2", "30% O2 or more"), cf_group]):
        rows.append({"investigation": "ACME CFI-G", "label_source": "PI table", "o2_band": band, "result": grp, "tests": int(len(d))})
    p = data / "marts/model_flame_design.parquet"
    if p.exists():
        fd = pd.read_parquet(p)
        fd = fd[(fd.flame_type == "normal") & fd.label_sustained.notna() & fd.o2_mole_frac.notna()]
        grp = fd.label_sustained.map({True: "Sustained", False: "Not sustained"})
        for (band, gname), d in fd.groupby([np.where(fd.o2_mole_frac < 0.30, "Below 30% O2", "30% O2 or more"), grp]):
            rows.append({"investigation": "ACME Flame Design", "label_source": "script log rule", "o2_band": band, "result": gname, "tests": int(len(d))})
    out["sustain"] = {"rows": rows, "meta": _meta("marts/model_cfig.parquet, marts/model_flame_design.parquet",
                                                    "Normal (fuel-into-oxidizer) spherical gas flames by chamber oxygen band. CFI-G: outcome written by the PI "
                                                    "(self-extinguished and no ignition = not sustained). Flame Design: outcome from the script-log rule, which agrees "
                                                    "with the PI in 94.7% of 284 CFI-G tests.")}

    p = data / "marts/bass2_extinction_points.parquet"
    if p.exists():
        e = pd.read_parquet(p)
        e["velocity_cm_s"] = e.air_at_extinction_cm_s.where(e.air_at_extinction_cm_s.notna() & (e.air_at_extinction_cm_s > 0), e.get("velocity_from_fan_cm_s"))
        e["velocity_basis"] = np.where(e.air_at_extinction_cm_s.notna() & (e.air_at_extinction_cm_s > 0), "air display",
                                       np.where(e.get("velocity_from_fan_cm_s").notna(), "fan reading + calibration", "not available"))
        e["o2_pct"] = (100 * e.o2_mole_frac).round(1)
        keep = e[["test_label", "pi", "fuel", "geometry", "o2_pct", "source", "velocity_cm_s", "velocity_basis", "fan_at_extinction",
                  "air_at_extinction_cm_s", "calibration_date", "calibration_gap_days", "extinction_time_s"]]
        out["extinction_points"] = {"rows": _clean(keep), "meta": _meta("marts/bass2_extinction_points.parquet",
                                                                       "BASS-II tests where the flame is reported to go out at a known flow setting, one row per evidence source.")}

    vs = data / "video/video_seconds.parquet"
    if vs.exists():
        v = pd.read_parquet(vs)
        v["test"] = v.record_uid.str.split(":").str[-1]
        keep = v[["test", "t_s", "fan_display", "air_cm_s", "rad_best", "lit_px"]].rename(columns={"rad_best": "radiometer", "lit_px": "camera_lit_px"})
        keep["camera_lit_px"] = keep.camera_lit_px.round(1)
        out["video_seconds"] = {"rows": _clean(keep), "meta": _meta("video/video_seconds.parquet (video session, flame/video)",
                                                                   f"Overlay telemetry read frame by frame from {keep.test.nunique() if 'test' in keep else len(keep)} BASS-II videos, one row per second: fan display, air speed, radiometer, and lit pixels seen by the camera.")}
        s = pd.read_parquet(data / "video/video_summary.parquet")
        tt = t.set_index("record_uid")
        s["test"] = s.record_uid.str.split(":").str[-1]
        s["material"] = s.record_uid.map(tt.fuel_raw)
        s["o2_pct"] = (100 * s.record_uid.map(tt.o2_mole_frac)).round(1)
        s["fan_steps"] = s.fan_plateaus.map(lambda x: ", ".join(f"{p['fan_display']:g}" for p in json.loads(x)))
        keep = s[["test", "material", "o2_pct", "duration_s", "fan_steps", "rad_decay_onset_s", "camera_extinction_s", "camera_flame_kind", "rad_saturated"]]
        out["video_tests"] = {"rows": _clean(keep), "meta": _meta("video/video_summary.parquet, tests.parquet",
                                                                 f"Per-test summary of the {len(keep)} videos joined to the test matrix on record_uid.")}
    d = data / "dash"
    d.mkdir(exist_ok=True)
    for k, v in out.items():
        (d / f"{k}.json").write_text(json.dumps(v, ensure_ascii=False, allow_nan=False) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=Path("data/processed"), type=Path)
    out = run(ap.parse_args().data)
    print({k: len(v["rows"]) for k, v in out.items()})


if __name__ == "__main__":
    main()
