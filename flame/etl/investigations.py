"""Investigation cards built from PSI metadata, the file inventory and the gold tables."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .gold import gravity_from_metadata, varied_fixed

VIDEO_EXT = {".mp4", ".ts", ".avi", ".mov", ".mpg", ".mpeg", ".m4v", ".wmv"}


def load_metadata(raw_dir: Path) -> dict:
    out = {}
    for f in sorted((raw_dir / "investigations").glob("PSI-*.json")):
        d = json.loads(f.read_text())
        out[d["accession"]] = d
    return out


def combustion_ids(raw_dir: Path) -> list[str]:
    with open(raw_dir / "inventory.csv", newline="") as fh:
        ids = {r["psi_id"] for r in csv.DictReader(fh)}
    return sorted(ids, key=lambda s: int(s.split("-")[1]))


def inventory_stats(raw_dir: Path) -> dict:
    stats = defaultdict(lambda: {"n_files": 0, "bytes": 0, "n_videos": 0, "video_bytes": 0, "by_category": Counter()})
    with open(raw_dir / "inventory.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            s = stats[r["psi_id"]]
            size = int(float(r["size_bytes"] or 0))
            s["n_files"] += 1
            s["bytes"] += size
            s["by_category"][r["category"]] += 1
            if r["ext"].lower() in VIDEO_EXT:
                s["n_videos"] += 1
                s["video_bytes"] += size
    return stats


def _clean(s):
    return re.sub(r"\s+", " ", s or "").strip() or None


def build(raw_dir: Path, meta: dict, curated: dict, tests: pd.DataFrame, camp: pd.DataFrame, meas: pd.DataFrame,
          table_stats: list[dict]) -> list[dict]:
    inv = inventory_stats(raw_dir)
    cards = []
    allrec = pd.concat([tests, camp], ignore_index=True)
    for psi in combustion_ids(raw_dir):
        d = meta[psi]
        g, level, gsrc = gravity_from_metadata(d)
        t = tests[tests.psi_id == psi]
        r = allrec[allrec.psi_id == psi]
        m = meas[meas.record_uid.isin(r.record_uid)]
        cur = curated["investigations"].get(psi, {})
        s = inv[psi]
        measured = (m.groupby(["category", "variable"]).size().reset_index(name="n")
                    .sort_values(["category", "variable"]).to_dict("records"))
        cards.append({
            "psi_id": psi,
            "acronym": _clean(d.get("investigationAcronym")),
            "title": _clean(d.get("title")),
            "doi": _clean(d.get("doi")),
            "url": f"https://psi.nasa.gov/physci/repo/investigation/{psi}",
            "research_area": _clean(d.get("researchArea")), "sub_research_area": _clean(d.get("subResearchArea")),
            "objective": _clean(d.get("objective")), "approach": _clean(d.get("approach")),
            "hypothesis": _clean(d.get("hypothesis")), "research_impacts": _clean(d.get("researchImpacts")),
            "hardware": _clean(d.get("experimentHardware")), "platform": _clean(d.get("flightProgram")),
            "gravity": {"gravity_g": g, "gravity_level": level, "source": gsrc},
            "start_date": _clean(d.get("investigationStartDate")), "end_date": _clean(d.get("investigationEndDate")),
            "managing_center": _clean(d.get("managingNasaCenter")),
            "team": [{"name": _clean(f"{c.get('firstName', '')} {c.get('lastName', '')}"),
                      "roles": [x.get("annotationValue") if isinstance(x, dict) else x for x in c.get("roles", [])],
                      "affiliation": _clean(c.get("affiliation"))} for c in d.get("contacts", [])],
            "publications": [{"title": _clean(p.get("title")), "authors": _clean(p.get("authorList")),
                              "doi": _clean(p.get("doi"))} for p in d.get("publications", [])],
            "pillar": {"primary": cur.get("pillar"), "secondary": cur.get("pillar_secondary"),
                       "rationale": cur.get("rationale"), "basis": curated["basis"]},
            "files": {"n_files": s["n_files"], "total_gb": round(s["bytes"] / 1e9, 2), "n_videos": s["n_videos"],
                      "video_gb": round(s["video_bytes"] / 1e9, 2), "by_category": dict(sorted(s["by_category"].items()))},
            "dataset": {
                "tables": [x for x in table_stats if x["psi_id"] == psi],
                "n_tests": int(len(t)), "n_campaign_records": int(len(r) - len(t)),
                "n_tests_with_outcome_label": int(t.has_outcome_label.sum()) if len(t) else 0,
                "fuels": sorted(x for x in r.fuel.dropna().unique()),
                "fuel_phases": sorted(x for x in r.fuel_phase.dropna().unique()),
                "geometries": sorted(x for x in r.geometry.dropna().unique()),
                "conditions": varied_fixed(t) if len(t) else None,
                "measured_variables": measured,
                "has_experimental_table": bool(len(r)),
            },
        })
    return cards
