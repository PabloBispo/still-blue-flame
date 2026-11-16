"""Run the BASS video reader over a folder of PSI-25 videos.

    python -m flame.video.pipeline --videos ../flame-freefall/data/raw/files/PSI-25/"Raw Data" \
        --out data/processed/video [--tests data/processed/tests.parquet] [--jobs 2]

Outputs (all keyed by record_uid, the same key as tests.parquet):

    frames/<label>.parquet    one row per video frame: overlay values and camera signal
    video_seconds.parquet     the same, one row per second, all tests
    video_summary.parquet     one row per test (list columns as JSON strings)
    video_summary.json        the same with nested fields
    video_quality.json        reader self-checks per file
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from flame.video.bass import read_video
from flame.video.summarize import fan_plateaus, label_from_filename, per_second, summarize
from flame.video.units import add_flow, air_vs_fan

PSI_ID = "PSI-25"
TABLE_ID = "bass2"


def _file_key(path: str, duplicated: set) -> str:
    """Cache key of a video. Two PSI-25 files can carry the same test label (B15 on GMT183 and GMT188)."""
    name = os.path.basename(path)
    label = label_from_filename(name)
    if label in duplicated:
        m = re.search(r"GMT(\d+)", name)
        return f"{label}@GMT{m.group(1) if m else 'x'}"
    return label


def _date_offset(frames: pd.DataFrame, test_date) -> int | None:
    mo, dd, yy = (frames[c].median() for c in ("clock_month", "clock_day", "clock_year"))
    if not test_date or pd.isna(mo) or pd.isna(dd) or pd.isna(yy):
        return None
    try:
        return int((pd.Timestamp(f"20{int(yy):02d}-{int(mo):02d}-{int(dd):02d}") - pd.Timestamp(test_date)).days)
    except ValueError:
        return None


def _one(path: str, out: str | None = None, key: str | None = None) -> tuple[str, pd.DataFrame, dict]:
    label = label_from_filename(os.path.basename(path))
    key = key or label
    uid = f"{PSI_ID}:{TABLE_ID}:{label}"
    try:
        r = read_video(path)
    except Exception as exc:  # noqa: BLE001  a damaged file must not stop the batch
        return uid, None, {"record_uid": uid, "source_file": os.path.basename(path), "error": f"{type(exc).__name__}: {exc}"}
    frames = pd.DataFrame(r.frames)
    frames.insert(0, "record_uid", uid)
    quality = {
        "record_uid": uid, "source_file": os.path.basename(path), "fps": r.fps, "n_frames": r.n_frames,
        "duration_s": r.duration_s, "overlay_dx": r.overlay_dx, "overlay_dy": r.overlay_dy,
        "clock_selfcheck": r.clock_selfcheck, "overlay_present": r.overlay_present, "clock_step_ok": r.clock_step_ok,
        "edge_digit_check": r.edge_digit_check, "edge_visible_px": r.edge_visible_px, "readable": r.readable,
    }
    if out:                                  # save as soon as the file is read, so a later failure loses nothing
        frames.to_parquet(os.path.join(out, "frames", key + ".parquet"), index=False)
        with open(os.path.join(out, "frames", key + ".quality.json"), "w") as fh:
            json.dump(quality, fh)
    return uid, frames, quality


def _cached(path: str, out: str, key: str):
    """Frames already extracted for this file, or None."""
    fp = os.path.join(out, "frames", f"{key}.parquet")
    qp = os.path.join(out, "frames", f"{key}.quality.json")
    if os.path.exists(fp) and os.path.exists(qp):
        with open(qp) as fh:
            q = json.load(fh)
        if q.get("source_file") == os.path.basename(path):
            return q["record_uid"], pd.read_parquet(fp), q
    return None


ACCEPT_MIN_SELFCHECK = 0.98
# files accepted although the overlay date is not test date + 1 day, with the reason (checked by hand on 2026-11-15)
DATE_OFFSET_EXCEPTIONS = {
    "PSI-25:bass2:T9": "test_date typo in the PSI table (2018 vs 2014 on the overlay)",
    "PSI-25:bass2:T18": "test_date typo in the PSI table (2018 vs 2014 on the overlay)",
    "PSI-25:bass2:M16": "run crossed GMT midnight (overlay date is test date + 2)",
}


def _accept(uid: str, s: dict) -> tuple[bool, str | None]:
    """Per-file acceptance of the overlay reading: clock self-check and overlay date against the test date."""
    check = s.get("clock_selfcheck")
    if not s.get("in_tests_table"):
        return False, "not attached to a test of the table"
    if check is None or check < ACCEPT_MIN_SELFCHECK:
        return False, f"clock self-check below {ACCEPT_MIN_SELFCHECK}"
    if s.get("clock_minus_test_date_days") == 1:
        return True, None
    if uid in DATE_OFFSET_EXCEPTIONS:
        return True, DATE_OFFSET_EXCEPTIONS[uid]
    return False, "overlay date is not test date + 1 day"


def run(videos: str, out: str, tests: str | None = None, jobs: int = 2, fan_calibration: str | None = None,
        measurements: str | None = None, reuse: bool = True) -> dict:
    files = sorted(f for f in glob.glob(os.path.join(videos, "*.mp4"))
                   if label_from_filename(os.path.basename(f)) and not any(x in f for x in ("Still_Sequence", "2ND_VIEW", "LOS-portion")))
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    dates = {}
    if tests and os.path.exists(tests):
        t = pd.read_parquet(tests, columns=["record_uid", "test_date"])
        dates = dict(zip(t["record_uid"], t["test_date"]))
    fan_cal = pd.read_parquet(fan_calibration) if fan_calibration and os.path.exists(fan_calibration) else None
    restrictor = {}
    if measurements and os.path.exists(measurements):
        m = pd.read_parquet(measurements, columns=["record_uid", "variable", "value_text"])
        m = m[m["variable"] == "flow_restrictor"]
        restrictor = {u: int(v) for u, v in zip(m["record_uid"], m["value_text"]) if isinstance(v, str) and v.isdigit()}
    labels = [label_from_filename(os.path.basename(f)) for f in files]
    duplicated = {x for x in labels if labels.count(x) > 1}
    keys = {f: _file_key(f, duplicated) for f in files}
    cached = {f: _cached(f, out, keys[f]) for f in files} if reuse else {}
    todo = [f for f in files if not cached.get(f)]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        fresh = dict(zip(todo, ex.map(_one, todo, [out] * len(todo), [keys[f] for f in todo])))
    results = [cached.get(f) or fresh[f] for f in files]
    seconds, summaries, quality = [], [], []
    for path, (uid, frames, q) in zip(files, results):
        if frames is None:
            quality.append(q)
            continue
        label = uid.split(":")[-1]
        dup_note = None
        if label in duplicated and _date_offset(frames, dates.get(uid)) != 1:
            # same label, other recording day: keep the data, but do not attach it to the test
            dup_note = f"label {label} is shared by two videos; this file ({keys[path]}) does not match the test date"
            uid = f"{uid}?{keys[path].split('@')[-1]}"
            frames = frames.assign(record_uid=uid)
            q = dict(q, record_uid=uid)
        sec = per_second(frames)
        flow_meta = None
        if fan_cal is not None:
            sec, flow_meta = add_flow(sec, fan_cal, dates.get(uid), restrictor.get(uid))
            flow_meta.update(air_vs_fan(sec, fan_plateaus(sec)))
        sec.insert(0, "record_uid", uid)
        seconds.append(sec)
        s = summarize(uid, frames, dates.get(uid), sec=sec, flow_meta=flow_meta)
        s["in_tests_table"] = (uid in dates) if dates else None
        s["clock_selfcheck"] = q.get("clock_selfcheck")
        s["overlay_present"] = q.get("overlay_present")
        s["video_accepted"], s["accept_note"] = _accept(uid, s)
        s["file_key"] = keys[path]
        if dup_note:
            s["notes"].append(dup_note)
        summaries.append(s)
        quality.append(q)
    pd.concat(seconds, ignore_index=True).to_parquet(os.path.join(out, "video_seconds.parquet"), index=False)
    flat = pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in s.items()}
                         for s in summaries])
    flat.to_parquet(os.path.join(out, "video_summary.parquet"), index=False)
    flat[["record_uid", "file_key", "video_accepted", "accept_note", "clock_selfcheck", "overlay_present",
          "clock_minus_test_date_days"]].to_csv(os.path.join(out, "video_acceptance.csv"), index=False)
    with open(os.path.join(out, "video_summary.json"), "w") as fh:
        json.dump(summaries, fh, indent=1)
    with open(os.path.join(out, "video_quality.json"), "w") as fh:
        json.dump(quality, fh, indent=1)
    return {"n_videos": len(files), "summaries": summaries, "quality": quality}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", default="data/processed/video")
    ap.add_argument("--tests", default="data/processed/tests.parquet")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--fan-calibration", default="data/processed/reference/bass2_fan_calibration.parquet")
    ap.add_argument("--measurements", default="data/processed/measurements.parquet")
    ap.add_argument("--no-reuse", action="store_true", help="re-read videos even when their frames are already extracted")
    a = ap.parse_args()
    res = run(a.videos, a.out, a.tests, a.jobs, a.fan_calibration, a.measurements, reuse=not a.no_reuse)
    for q in res["quality"]:
        if "error" in q:
            print(f'{q["record_uid"]}: NOT READ, {q["error"]}')
    for s, q in zip(res["summaries"], [q for q in res["quality"] if "error" not in q]):
        print(f'{s["record_uid"]}: clock check {q["clock_selfcheck"]:.3f}, '
              f'rad decay at {s["rad_decay_onset_s"]} s (FAN {s["fan_at_rad_decay"]}, {s.get("flow_at_rad_decay_cm_s")} cm/s), '
              f'camera extinction {s["camera_extinction_s"]} s; {"; ".join(s["notes"]) or "no notes"}')


if __name__ == "__main__":
    main()
