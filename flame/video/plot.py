"""Overview figure: FAN, radiometer and camera signal per test.

    python -m flame.video.plot --video-dir data/processed/video --out docs/img/video_overview.png
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", default="data/processed/video")
    ap.add_argument("--out", default="docs/img/video_overview.png")
    a = ap.parse_args()
    sec = pd.read_parquet(os.path.join(a.video_dir, "video_seconds.parquet"))
    summ = {s["record_uid"]: s for s in json.load(open(os.path.join(a.video_dir, "video_summary.json")))}
    uids = sorted(summ)
    fig, axes = plt.subplots(len(uids), 1, figsize=(9, 2.3 * len(uids)), constrained_layout=True)
    for ax, uid in zip(np.atleast_1d(axes), uids):
        d, s = sec[sec.record_uid == uid], summ[uid]
        ax.step(d.t_s, d.fan_display, where="post", color="#1f4e79", lw=1.6, label="FAN display")
        ax.set_ylabel("FAN", color="#1f4e79")
        ax.set_ylim(30, 95)
        ax2 = ax.twinx()
        ax2.plot(d.t_s, d.rad_best, color="#c0392b", lw=1.4, label="RAD (radiometer)")
        cam = np.log10(d.dim_px.clip(lower=1))
        ax2.plot(d.t_s, cam / cam.max() * 4.1, color="#7f8c8d", lw=1.0, alpha=0.9, label="camera lit px (log, scaled)")
        ax2.set_ylim(0, 4.4)
        ax2.set_ylabel("RAD", color="#c0392b")
        if s["rad_decay_onset_s"] is not None:
            ax.axvline(s["rad_decay_onset_s"], color="#c0392b", ls="--", lw=1)
        if s["camera_extinction_s"] is not None:
            ax.axvline(s["camera_extinction_s"], color="#7f8c8d", ls=":", lw=1.4)
        ax.set_title(f'{uid}   radiometer fall {s["rad_decay_onset_s"]} s (dashed), '
                     f'camera end {s["camera_extinction_s"]} s (dotted)', fontsize=9, loc="left")
        if ax is np.atleast_1d(axes)[0]:
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, ncol=3, loc="lower left")
    np.atleast_1d(axes)[-1].set_xlabel("video time (s)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=130)


if __name__ == "__main__":
    main()
