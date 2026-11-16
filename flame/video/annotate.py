"""Render a BASS video with the measurements drawn on it.

    python -m flame.video.annotate --video <file.mp4> --frames data/processed/video/frames/O1.parquet --out O1_annotated.mp4

Draws the telemetry box that was read, the outline of the largest lit region,
the flame head (largest inscribed circle, green cross at its centre), and a panel with the values read
from the overlay plus the flame area and FAN history. Needs ffmpeg for the
final H.264 encode.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile

import cv2
import numpy as np
import pandas as pd

from flame.video import bass

PANEL_H = 150
CYAN, YELLOW, WHITE, GREY, ORANGE = (255, 220, 0), (0, 230, 255), (240, 240, 240), (120, 120, 120), (0, 150, 255)


def _fmt(v, nd=0):
    return "--" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def _spark(panel, x0, y0, w, h, series, upto, color, vmax):
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (50, 50, 50), 1)
    n = len(series)
    xs = (np.arange(n) / max(n - 1, 1) * w + x0).astype(int)
    ys = y0 + h - (np.nan_to_num(series) / vmax * (h - 2)).clip(0, h - 2).astype(int)
    pts = np.stack([xs, ys], 1)
    cv2.polylines(panel, [pts], False, (70, 70, 70), 1, cv2.LINE_AA)
    if upto > 1:
        cv2.polylines(panel, [pts[:upto]], False, color, 1, cv2.LINE_AA)
    cv2.line(panel, (int(xs[min(upto, n - 1)]), y0), (int(xs[min(upto, n - 1)]), y0 + h), WHITE, 1)


def annotate(video: str, frames_parquet: str, out: str, stride: int = 1) -> None:
    """`stride` keeps one frame in every N (output frame rate drops by N; playback speed stays real time)."""
    f = pd.read_parquet(frames_parquet)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dx, dy = float(f.overlay_dx.iloc[0]), int(f.overlay_dy.iloc[0])
    box = (bass._cell_x(-1, dx) - 4, bass.ROWS_LABELS[0] + dy - 4, bass.VISIBLE_RIGHT, bass.ROWS_CLOCK[1] + dy + 3)
    area = f.flame_area_px.fillna(0).to_numpy()
    fan = f.fan_display.ffill().bfill().to_numpy()
    rad = f.rad_1dp.ffill().bfill().to_numpy()
    amax, fmax, rmax = max(area.max(), 1), max(np.nanmax(fan), 1), max(np.nanmax(rad), 0.1)
    step = max(1, len(f) // 660)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (w, h + PANEL_H))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(f)):
        if i % stride:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        row = f.iloc[i]
        cv2.rectangle(frame, box[:2], box[2:], YELLOW, 1)
        cv2.putText(frame, "telemetry read", (box[0], box[3] + 13), font, 0.4, YELLOW, 1, cv2.LINE_AA)
        if row.lights_on:
            cv2.putText(frame, "chamber lights on: flame not measured", (20, h - 14), font, 0.45, ORANGE, 1, cv2.LINE_AA)
        elif np.isfinite(row.flame_area_px):
            mask, _ = bass.flame_region(frame)
            if mask is not None:
                cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(frame, cs, -1, CYAN, 1, cv2.LINE_AA)
                hc = (int(round(row.head_x)), int(round(row.head_y)))
                cv2.circle(frame, hc, int(round(row.head_radius_px)), WHITE, 1, cv2.LINE_AA)
                cv2.drawMarker(frame, hc, (0, 255, 0), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
        panel = np.zeros((PANEL_H, w, 3), np.uint8)
        tod = row.clock_tod_s
        clock = "--:--:--" if not np.isfinite(tod) else f"{int(tod)//3600:02d}:{int(tod)%3600//60:02d}:{int(tod)%60:02d}"
        l1 = f"t {row.t_s:7.2f} s   clock {clock}   FUEL {_fmt(row.fuel)}   AIR {_fmt(row.air_cm_s)} cm/s   FAN {_fmt(row.fan_display)}   RAD {_fmt(row.rad, 2)}"
        l2 = (f"flame area {_fmt(row.flame_area_px)} px   head ({_fmt(row.head_x)}, {_fmt(row.head_y)})   "
              f"head radius {_fmt(row.head_radius_px, 1)} px   size {_fmt(row.flame_w)} x {_fmt(row.flame_h)} px")
        cv2.putText(panel, l1, (10, 20), font, 0.45, YELLOW, 1, cv2.LINE_AA)
        cv2.putText(panel, l2, (10, 42), font, 0.45, CYAN, 1, cv2.LINE_AA)
        k = i // step
        for j, (name, ser, col, vmax) in enumerate((("flame area", area, CYAN, amax), ("FAN", fan, YELLOW, fmax), ("RAD", rad, ORANGE, rmax))):
            y = 54 + j * 31
            cv2.putText(panel, name, (10, y + 18), font, 0.4, col, 1, cv2.LINE_AA)
            _spark(panel, 90, y, w - 100, 27, ser[::step], k, col, vmax)
        vw.write(np.vstack([frame, panel]))
    vw.release(); cap.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                    "-movflags", "+faststart", out], check=True)
    os.remove(tmp)


def _job(args) -> str:
    video, frames, out, stride = args
    annotate(video, frames, out, stride)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="one video file")
    ap.add_argument("--frames", help="its frames parquet")
    ap.add_argument("--out", help="output mp4 (single video) or output folder (with --videos)")
    ap.add_argument("--videos", help="folder of PSI-25 videos: annotate every one that has a frames parquet")
    ap.add_argument("--video-dir", default="data/processed/video")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    if a.video:
        annotate(a.video, a.frames, a.out, a.stride)
        return
    import glob
    from concurrent.futures import ProcessPoolExecutor
    from flame.video.summarize import label_from_filename
    out = a.out or os.path.join(a.video_dir, "annotated")
    os.makedirs(out, exist_ok=True)
    jobs = []
    for v in sorted(glob.glob(os.path.join(a.videos, "*.mp4"))):
        if any(x in v for x in ("Still_Sequence", "2ND_VIEW", "LOS-portion")):
            continue
        label = label_from_filename(os.path.basename(v))
        fp = os.path.join(a.video_dir, "frames", f"{label}.parquet")
        dest = os.path.join(out, f"{label}_annotated.mp4")
        if label and os.path.exists(fp) and not os.path.exists(dest):
            jobs.append((v, fp, dest, a.stride))
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for done in ex.map(_job, jobs):
            print("wrote", done, flush=True)


if __name__ == "__main__":
    main()
