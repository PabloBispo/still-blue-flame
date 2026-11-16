"""Download selected files from NASA PSI into the raw directory. Standard library only.

    python3 -m flame.etl.fetch_psi --raw ../flame-freefall/data/raw --psi PSI-117 --category "Analyzed Data"
    python3 -m flame.etl.fetch_psi --raw ... --psi PSI-23 --category "Raw Data" --ext .txt --max-kb 50

Selection comes from raw/inventory.csv. Files land in raw/files/<PSI>/<category>/<file_name>, the layout the
ETL already reads. Idempotent: a file that exists with the inventory size is skipped. Every download is
appended to raw/MANIFEST.sha256 and to raw/fetch_log.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://psi.nasa.gov/geode-py/ws/studies/{psi}/download?file={file}&version={version}&redirect=false"
UA = "still-blue-flame/1.0 (NASA Space Apps 2026; research use of public PSI data)"


def get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_one(row: dict, raw: Path) -> dict:
    dest = raw / "files" / row["psi_id"] / row["category"] / row["file_name"]
    size = int(float(row["size_bytes"] or 0))
    out = {"psi_id": row["psi_id"], "category": row["category"], "file_name": row["file_name"], "bytes": size, "status": "", "sha256": ""}
    if dest.exists() and dest.stat().st_size == size:
        out["status"] = "exists"
        return out
    url = BASE.format(psi=row["psi_id"], file=urllib.parse.quote(row["file_name"]), version=row["version"] or 1)
    for attempt in range(3):
        try:
            signed = get(url).decode().strip().strip('"')
            if not signed.startswith("http"):
                raise ValueError(f"unexpected answer: {signed[:80]}")
            data = get(signed, timeout=180)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            out.update(status="downloaded", bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            return out
        except Exception as exc:  # noqa: BLE001
            out["status"] = f"error: {type(exc).__name__}: {str(exc)[:120]}"
            time.sleep(1.5 * (attempt + 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--psi", required=True)
    ap.add_argument("--category")
    ap.add_argument("--ext", help="comma-separated, e.g. .txt,.csv")
    ap.add_argument("--match", help="regex on file_name")
    ap.add_argument("--max-kb", type=float)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    exts = {e.strip().lower() for e in a.ext.split(",")} if a.ext else None
    with open(a.raw / "inventory.csv", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["psi_id"] == a.psi
                and (not a.category or r["category"] == a.category)
                and (not exts or r["ext"].lower() in exts)
                and (not a.match or re.search(a.match, r["file_name"]))
                and (a.max_kb is None or float(r["size_bytes"] or 0) <= a.max_kb * 1000)
                and r["restricted"] != "True"]
    total = sum(float(r["size_bytes"] or 0) for r in rows)
    print(f"selected {len(rows)} files, {total / 1e6:.1f} MB", flush=True)
    if a.dry_run:
        return
    done = []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(lambda r: fetch_one(r, a.raw), rows), 1):
            done.append(res)
            if i % 50 == 0 or i == len(rows):
                ok = sum(d["status"] in ("downloaded", "exists") for d in done)
                print(f"{i}/{len(rows)} ok={ok}", flush=True)
    log = a.raw / "fetch_log.csv"
    new = not log.exists()
    with open(log, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(done[0]) if done else ["status"])
        if new:
            w.writeheader()
        w.writerows(done)
    with open(a.raw / "MANIFEST.sha256", "a") as fh:
        for d in done:
            if d["status"] == "downloaded":
                fh.write(f"{d['sha256']}  files/{d['psi_id']}/{d['category']}/{d['file_name']}\n")
    errors = [d for d in done if d["status"].startswith("error")]
    print(f"downloaded={sum(d['status'] == 'downloaded' for d in done)} exists={sum(d['status'] == 'exists' for d in done)} errors={len(errors)}")
    for d in errors[:10]:
        print("  ", d["file_name"], d["status"])
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
