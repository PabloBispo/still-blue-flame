"""Bronze layer: read source tables as text, exactly as they are."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import re
import warnings
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm_header(h) -> str:
    return re.sub(r"\s+", " ", str(h if h is not None else "")).strip()


def cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, dt.datetime):
        return v.date().isoformat() if v.time() == dt.time(0) else v.isoformat(sep=" ")
    if isinstance(v, (dt.date, dt.time)):
        return v.isoformat()
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else repr(v)
    return str(v).replace("\xa0", " ").strip()


def _rows_csv(path: Path):
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as fh:
        for i, row in enumerate(csv.reader(fh), start=1):
            yield i, [cell(c) for c in row]


def _rows_xlsx(path: Path, sheet: str):
    import openpyxl
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wb = openpyxl.load_workbook(path, data_only=True)
    for i, row in enumerate(wb[sheet].iter_rows(values_only=True), start=1):
        yield i, [cell(c) for c in row]


def read_table(raw_dir: Path, src: dict) -> tuple[list[dict], dict]:
    """Return (rows, info). Each row: {'_sheet','_row', header: text...}. Blank rows are counted, not kept."""
    path = raw_dir / src["file"]
    sheets = src.get("sheets") or [src.get("sheet")]
    header_row = src.get("header_row", 1)
    usecols = src.get("usecols")
    rows, n_blank, n_total = [], 0, 0
    for sheet in sheets:
        it = _rows_xlsx(path, sheet) if path.suffix.lower() == ".xlsx" else _rows_csv(path)
        headers = None
        for i, cells in it:
            if usecols is not None:
                cells = [cells[j] if j < len(cells) else "" for j in usecols]
            if i < header_row:
                continue
            if i == header_row:
                aliases = src.get("header_aliases", {})
                headers = [aliases.get(norm_header(c), norm_header(c)) for c in cells]
                while headers and not headers[-1]:
                    headers.pop()
                continue
            n_total += 1
            if not any(cells[: len(headers)]):
                n_blank += 1
                continue
            row = {"_sheet": sheet, "_row": i}
            for h, c in zip(headers, cells):
                if h:
                    row[h] = c
            for h in headers[len(cells):]:
                row[h] = ""
            rows.append(row)
    info = {"file": src["file"], "sha256": sha256(path), "rows_total": n_total, "rows_blank": n_blank,
            "headers": [h for h in headers if h]}
    return rows, info


def join_sheets(raw_dir: Path, src: dict, rows: list[dict]) -> list[dict]:
    """Left-join extra sheets of the same workbook on a key column (CFI summary workbook)."""
    for j in src.get("join", []):
        extra, _ = read_table(raw_dir, {"file": src["file"], "sheet": j["sheet"], "header_row": j.get("header_row", 1)})
        index = {r[j["key"]]: r for r in extra if r.get(j["key"])}
        for r in rows:
            other = index.get(r.get(j["key"]), {})
            for k, v in other.items():
                if k.startswith("_") or k == j["key"]:
                    continue
                r[f"{j['sheet']}::{k}"] = v
            for k in extra[0] if extra else []:
                if not k.startswith("_") and k != j["key"]:
                    r.setdefault(f"{j['sheet']}::{k}", "")
    return rows


_TESTPOINT = re.compile(r"(\d{5}[A-Z]{1,2}\d{1,2}[a-z]?)")


def read_kv_dir(raw_dir: Path, src: dict) -> tuple[list[dict], dict]:
    """ACME per-test log files -> one row per test point.

    For each test point id the directory holds '<prefix>as run<ID>.txt' (key:value settings actually run),
    '<prefix><ID>_TP.txt' (script timeline: 'Label: timestamp', '***' when the step did not happen) and
    '<prefix><ID>_Comments.txt'. Headers are 'asrun::<key>', 'tp::<label>', 'comments'.
    """
    d = raw_dir / src["dir"]
    prefix = src["prefix"]
    tests: dict[str, dict] = {}
    hashes = []
    for path in sorted(d.glob(prefix + "*.txt")):
        name = path.name[len(prefix):]
        kind = ("asrun" if name.startswith("as run") else "tp" if name.endswith("_TP.txt")
                else "comments" if name.endswith("_Comments.txt") else None)
        m = _TESTPOINT.search(name)
        if kind is None or not m or "Camera" in name:
            continue
        data = path.read_bytes()
        hashes.append(path.name + hashlib.sha256(data).hexdigest())
        text = data.decode("utf-8", errors="replace")
        row = tests.setdefault(m.group(1), {"testpoint": m.group(1)})
        row.setdefault("source_files", [])
        row["source_files"].append(path.name)
        if kind == "asrun":
            for part in re.split(r",\s*\n|\n", text):
                if ":" in part:
                    k, v = part.split(":", 1)
                    row[f"asrun::{k.strip()}"] = v.strip().rstrip(",")
        elif kind == "tp":
            for line in text.splitlines():
                if line.startswith(("Camera", "=====")):
                    break
                if ": " in line or line.rstrip().endswith(":"):
                    k, _, v = line.partition(":")
                    k, v = k.strip(), v.strip()
                    if k and k != "Test Point":
                        row[f"tp::{k}"] = "" if v == "***" else v
        else:
            row["comments"] = norm_header(text)
    headers = sorted({k for r in tests.values() for k in r})
    rows = []
    for i, (tid, r) in enumerate(sorted(tests.items()), start=1):
        r["source_files"] = "; ".join(r["source_files"])
        rows.append({"_sheet": None, "_row": i, **{h: r.get(h, "") for h in headers}})
    info = {"file": src["dir"], "sha256": hashlib.sha256("".join(sorted(hashes)).encode()).hexdigest(),
            "rows_total": len(rows), "rows_blank": 0, "headers": headers}
    return rows, info


def read(raw_dir: Path, src: dict) -> tuple[list[dict], dict]:
    if src.get("format") == "kv_dir":
        return read_kv_dir(raw_dir, src)
    rows, info = read_table(raw_dir, src)
    rows = join_sheets(raw_dir, src, rows)
    if src.get("join") and rows:
        info["headers"] = [k for k in rows[0] if not k.startswith("_")]
    return rows, info
