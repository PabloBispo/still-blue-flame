"""Reference tables that are not tests: instrument calibrations."""
from __future__ import annotations

import datetime as dt
import re
import warnings
from pathlib import Path

import pandas as pd

FAN_FILE = "files/PSI-25/Reference Data/PSI-25_Reference Data_Fan calibration data.xlsx"


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def bass2_fan_calibration(raw_dir: Path) -> pd.DataFrame | None:
    """BASS-II fan calibration: pot setting and fan display versus duct air speed, per flow restrictor and date.

    Layout: one block of columns per calibration date (Pot, Fan Display, Air Display), three stacked sections
    (no restrictor, 1 restrictor, 2 restrictors). The section label is written once, in the first block.
    """
    path = raw_dir / FAN_FILE
    if not path.exists():
        return None
    import openpyxl
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ws = openpyxl.load_workbook(path, data_only=True).worksheets[0]
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    blocks = []
    for i, row in enumerate(rows[:6]):
        for c, v in enumerate(row):
            if isinstance(v, str) and v.strip().upper() == "DATE":
                d = row[c + 1] if c + 1 < len(row) else None
                gmt = rows[i + 1][c + 1] if i + 1 < len(rows) and c + 1 < len(rows[i + 1]) else None
                blocks.append((c, d.date().isoformat() if isinstance(d, dt.datetime) else (str(d) if d else None), str(gmt) if gmt else None))
    out, restrictor = [], None
    for i, row in enumerate(rows):
        text = " ".join(str(v) for v in row[:6] if isinstance(v, str)).lower()
        m = re.search(r"(no|\d)\s*res\w*ictor", text)
        if m:
            restrictor = 0 if m.group(1) == "no" else int(m.group(1))
        for c, date, gmt in blocks:
            if c + 2 >= len(row):
                continue
            pot, fan, air = _num(row[c]), _num(row[c + 1]), _num(row[c + 2])
            if pot is None or fan is None or restrictor is None:
                continue
            out.append({"calibration_date": date, "calibration_label": gmt, "flow_restrictor": restrictor, "pot_setting": pot,
                        "fan_display": fan, "air_velocity_cm_s": air, "source_file": FAN_FILE, "source_row": i + 1})
    return pd.DataFrame(out)
