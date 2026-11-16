"""Build site/ask/facts.json: the only context the /ask/ SI agent sees.
Run from the repo root: python3 scripts/build_ask_facts.py"""
import json, math
from pathlib import Path
import pandas as pd

D = Path("data/processed")
OUT = Path("site/ask/facts.json")

def load(p):
    return json.loads((D / p).read_text())

def clean(o):
    if isinstance(o, float):
        return None if math.isnan(o) else round(o, 4)
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o

def cut(s, n=420):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " [...]"

inv = []
for i in load("investigations.json"):
    ds = i.get("dataset") or {}
    inv.append({
        "psi_id": i["psi_id"], "acronym": i.get("acronym"), "title": i.get("title"),
        "doi": i.get("doi"), "url": i.get("url"),
        "sub_research_area": i.get("sub_research_area"),
        "objective": cut(i.get("objective")), "approach": cut(i.get("approach"), 300),
        "platform": i.get("platform"), "gravity": i.get("gravity"),
        "start_date": i.get("start_date"), "end_date": i.get("end_date"),
        "pillar": i.get("pillar"),
        "n_publications": len(i.get("publications") or []),
        "files": {k: v for k, v in (i.get("files") or {}).items() if k != "by_category"},
        "dataset": {k: ([{"table_id": t.get("table_id"), "grain": t.get("grain"), "rows_kept": t.get("rows_kept")} for t in v]
                        if k == "tables" else v)
                    for k, v in ds.items() if not isinstance(v, (dict, list)) or k == "tables"},
        "source_file": "data/processed/investigations.json",
    })

env = pd.read_csv(D / "marts/condition_envelope.csv")
envelope = clean(json.loads(env.to_json(orient="records")))

v = pd.read_parquet(D / "marts/bass2_video.parquet")
psi, doi = str(v.psi_id.iloc[0]), str(v.doi.iloc[0])
kinds = {k: int(n) for k, n in v.end_of_test_kind.value_counts().items()}
sel = v[(v.video_accepted == True) & (v.end_of_test_kind == "fall_at_steady_fan")]
speed = []
for fuel, g in sel.groupby("fuel"):
    x = g.flow_at_rad_decay_cm_s.dropna()
    if len(x):
        speed.append({"fuel": str(fuel), "n": int(len(x)),
                      "median_flow_at_radiometer_decay_cm_s": round(float(x.median()), 2),
                      "min_cm_s": round(float(x.min()), 2), "max_cm_s": round(float(x.max()), 2)})
video = {
    "psi_id": psi, "doi": doi, "investigation": "BASS-II",
    "source_file": "data/processed/marts/bass2_video.parquet",
    "n_tests_in_table": int(len(v)), "n_video_accepted": int((v.video_accepted == True).sum()),
    "end_of_test_kind_counts": kinds,
    "end_of_test_kind_meaning": {
        "fall_at_steady_fan": "radiometer signal fell while the fan setting was steady",
        "fall_after_fan_step_up": "radiometer signal fell after the fan was stepped up",
        "video_ends_before_fall": "recording ends before any fall of the radiometer signal",
        "no_fall_detected": "no fall of the radiometer signal detected",
    },
    "flow_speed_when_signal_fell": {
        "population": "video_accepted = true and end_of_test_kind = fall_at_steady_fan, flow_at_rad_decay_cm_s not null",
        "n_total": int(sel.flow_at_rad_decay_cm_s.notna().sum()),
        "by_fuel": speed,
        "caveat": "Descriptive medians of the flow speed when the radiometer signal started to fall. Not a quenching or flammability boundary.",
    },
}

eda = load("eda/eda.json")
facts = {
    "about": "Still Blue unified dataset of NASA PSI microgravity combustion tests. Every number here was computed by code from data/processed/.",
    "rules_of_the_data": {
        "partial_gravity": "PSI holds 0 partial-gravity combustion tests in this dataset. All tests are microgravity.",
        "phases": "Gas, liquid and solid tests are never summed or compared as one population.",
        "gas_burners": "CFI-G and Flame Design are gas flames on burners, not habitat materials.",
        "log_labels": "Outcome labels derived from logs are 'not sustained', a Still Blue rule, not the researcher's label.",
    },
    "investigations": inv,
    "insights": {"source_file": "data/processed/marts/insights.json", **load("marts/insights.json")},
    "coverage": {"source_file": "data/processed/analysis/coverage.json", **load("analysis/coverage.json")},
    "bass2_o2": {"source_file": "data/processed/analysis/bass2_o2.json", "psi_id": psi, "doi": doi, **load("analysis/bass2_o2.json")},
    "condition_envelope": {"source_file": "data/processed/marts/condition_envelope.parquet",
                           "note": "One row per investigation table, fuel and oxidizer location. o2 as mole fraction, pressure in kPa. n_tests is the n.",
                           "rows": envelope},
    "bass2_video": video,
    "eda": {"source_file": "data/processed/eda/eda.json",
            **{k: eda[k] for k in ("atmospheres", "cfi", "others", "logs", "quality") if k in eda}},
}
facts = clean(facts)
OUT.write_text(json.dumps(facts, ensure_ascii=False, separators=(",", ":")))
kb = OUT.stat().st_size / 1024
print(f"{OUT}: {kb:.1f} KB, {len(inv)} investigations, {len(envelope)} envelope rows")
assert kb < 150, "facts.json above 150 KB"
