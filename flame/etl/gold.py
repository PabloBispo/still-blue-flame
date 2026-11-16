"""Gold layer: conform to vocabularies, attach metadata, derive, check, build analytic marts."""
from __future__ import annotations

import json
import re

import pandas as pd

from . import parsers as P
from .contract import CORE_COLUMNS, FIELD_BY_NAME, PIPELINE_VERSION

TARGET_ATMOSPHERES = {  # name: (o2 mole fraction, kPa, source)
    "ISS / sea level": (0.21, 101.3, "ISS cabin, 14.7 psia"),
    "Challenge statement": (0.30, 70.3, "Flame in Freefall statement, 30% O2 at 10.2 psia"),
    "Exploration atmosphere": (0.34, 56.5, "NASA Exploration Atmosphere, 34% O2 at 8.2 psia"),
}


# ---------------------------------------------------------------- vocabulary
def normalize_fuel(raw: str | None, vocab: dict) -> dict:
    out = {"fuel": None, "phase": None, "cls": None, "flags": [], "pct": None, "geometry": None,
           "thickness_mm": None, "diameter_mm": None}
    if not raw:
        return out
    for e in vocab["fuels"]:
        if re.search(e["pattern"], raw, re.I):
            out.update(fuel=e["fuel"], phase=e["phase"], cls=e["class"])
            if e.get("implied"):
                out["flags"].append("material_implied")
            if e.get("artifact"):
                out["flags"].append("autofill_artifact")
            break
    else:
        out["flags"].append("fuel_unmapped")
    for e in vocab["geometry"]:
        if re.search(e["pattern"], raw, re.I):
            out["geometry"] = e["geometry"]
            break
    pcts = re.findall(rf"({P.NUM})\s*%", raw)
    if len(pcts) == 1 and out["cls"] != "blend":
        out["pct"] = float(pcts[0]) / 100
    if out["phase"] == "solid":
        m = re.search(rf"({P.NUM})\s*micron", raw, re.I)
        if m:
            out["thickness_mm"] = float(m.group(1)) / 1000
        else:
            m = re.search(rf"({P.NUM})\s*mm\b", raw, re.I)
            if m:
                out["thickness_mm"] = float(m.group(1))
        if out["geometry"] in ("slab", "thin_film") and out["thickness_mm"] is not None:
            out["geometry"] = "thin_film" if out["thickness_mm"] < 1 else "slab"
        if out["geometry"] == "rod":
            inch = P.inches(raw)
            if inch is None:
                m = re.match(rf'\s*({P.NUM})\s*"', raw)
                inch = float(m.group(1)) if m else None
            if inch:
                out["diameter_mm"] = round(inch * 25.4, 3)
        if out["geometry"] == "sphere":
            m = re.search(rf"({P.NUM})\s*cm", raw, re.I)
            if m:
                out["diameter_mm"] = float(m.group(1)) * 10
    return out


def gravity_from_metadata(meta: dict) -> tuple[float | None, str | None, str]:
    program = (meta.get("flightProgram") or "").strip()
    src = f"PSI metadata flightProgram = '{program}'"
    if re.search(r"ISS|Space Station|Cygnus", program, re.I):
        return 0.0, "microgravity", src
    return None, None, src + " (no flight platform; gravity not assigned)"


# ---------------------------------------------------------------- conform
def _clock(s: str) -> int | None:
    m = re.match(r"^(\d{1,2})[:.](\d{2})[:.](\d{2})$", s or "")
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) if m else None


def conform(records: list[dict], measurements: list[dict], provenance: list[dict], meta: dict, fuels: dict):
    times: dict[str, dict[str, str]] = {}
    for m in measurements:
        if m["variable"] in ("ignition_time_gmt", "flame_off_time_gmt"):
            times.setdefault(m["record_uid"], {})[m["variable"]] = m["value_text"]
    out = []
    for r in records:
        flags = set(r.get("flags", []))
        md = meta.get(r["psi_id"], {})
        r["investigation"] = md.get("investigationAcronym") or None
        r["doi"] = md.get("doi") or None
        r["platform"] = md.get("flightProgram") or None
        r["gravity_g"], r["gravity_level"], r["gravity_source"] = gravity_from_metadata(md)

        nf = normalize_fuel(r.get("fuel_raw"), fuels)
        r["fuel"], r["fuel_class"] = nf["fuel"], nf["cls"]
        r["fuel_phase"] = r.get("fuel_phase") or nf["phase"]
        r["geometry"] = r.get("geometry") or nf["geometry"]
        flags.update(nf["flags"])
        size_from_label = False
        if r.get("fuel_mole_frac") is None and nf["pct"] is not None and r["fuel_phase"] == "gas":
            r["fuel_mole_frac"] = nf["pct"]
            flags.add("fuel_frac_from_label")
            provenance.append(_prov(r, "fuel_mole_frac", "(fuel label)", r["fuel_raw"], "%", "label_percent"))
        for field, key in (("thickness_mm", "thickness_mm"), ("char_diameter_mm", "diameter_mm")):
            if r.get(field) is None and nf[key] is not None:
                r[field] = nf[key]
                size_from_label = True
                provenance.append(_prov(r, field, "(material label)", r["fuel_raw"], None, "label_size"))
        if size_from_label:
            flags.add("size_from_label")

        o2, o2f, p = r.get("o2_mole_frac"), r.get("o2_mole_frac_final"), r.get("pressure_kpa")
        if o2 is not None and p is not None and r.get("oxidizer_location") == "ambient":
            r["ppo2_kpa"] = round(o2 * p, 4)
        inert = [(r.get(k), n) for k, n in (("n2_mole_frac", "N2"), ("co2_mole_frac", "CO2"), ("he_mole_frac", "He"))]
        if any(v is not None for v, _ in inert):
            present = sorted(((v, n) for v, n in inert if v is not None and v > 0.01), reverse=True)
            r["diluent"] = "+".join(n for _, n in present) or None
            total = (o2 or 0) + sum(v or 0 for v, _ in inert)
            if o2 is not None and abs(total - 1) > 0.02:
                flags.add("composition_sum_mismatch")
        if o2 is not None and o2f is not None:
            if o2 - o2f > 0.05:
                flags.add("suspect_o2_drop")
            if o2f - o2 > 0.005:
                flags.add("o2_final_above_initial")
        if r.get("burn_time_s") is None and r["record_uid"] in times:
            t0, t1 = (_clock(times[r["record_uid"]].get(k)) for k in ("ignition_time_gmt", "flame_off_time_gmt"))
            if t0 is not None and t1 is not None:
                r["burn_time_s"] = float((t1 - t0) % 86400)
                flags.add("burn_time_from_clock")
                provenance.append(_prov(r, "burn_time_s", "GMT Flame Off - GMT ignition",
                                        f"{times[r['record_uid']]['flame_off_time_gmt']} - {times[r['record_uid']]['ignition_time_gmt']}",
                                        "s", "clock_difference"))
        if r.get("ignited") is False and r.get("outcome") is None:
            r["outcome"] = "no_ignition"
        r["has_outcome_label"] = r.get("ignited") is not None or r.get("outcome") is not None
        r["flags"] = sorted(flags)
        r["pipeline_version"] = PIPELINE_VERSION
        out.append({c: r.get(c) for c in CORE_COLUMNS})
    return out


def _prov(r, field, col, raw, unit_raw, parser):
    return {"record_uid": r["record_uid"], "field": field, "source_column": col, "raw_value": raw, "unit_raw": unit_raw,
            "unit": FIELD_BY_NAME[field].unit, "parser": parser, "value_lo": None, "value_hi": None,
            "qualifier": "eq", "flags": ["derived"]}


def frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records, columns=CORE_COLUMNS)
    for f in FIELD_BY_NAME.values():
        if f.dtype == "float":
            df[f.name] = pd.to_numeric(df[f.name], errors="coerce").astype("float64")
        elif f.dtype == "bool":
            df[f.name] = df[f.name].astype("boolean")
        elif f.dtype == "int":
            df[f.name] = df[f.name].astype("Int64")
        elif f.dtype == "string":
            df[f.name] = df[f.name].astype("string")
    return df.sort_values("record_uid", kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------- marts
CONDITIONS = ["fuel", "fuel_mole_frac", "o2_mole_frac", "pressure_kpa", "flow_velocity_cm_s", "flow_direction",
              "diluent", "thickness_mm", "char_diameter_mm", "gravity_g"]


def varied_fixed(df: pd.DataFrame) -> dict:
    out = {}
    for c in CONDITIONS:
        s = df[c].dropna()
        if s.empty:
            out[c] = {"status": "not_reported"}
        elif s.nunique() == 1:
            out[c] = {"status": "fixed", "value": _py(s.iloc[0]), "n": int(len(s))}
        elif pd.api.types.is_numeric_dtype(s):
            out[c] = {"status": "varied", "min": round(float(s.min()), 4), "max": round(float(s.max()), 4),
                      "n_distinct": int(s.nunique()), "n": int(len(s))}
        else:
            out[c] = {"status": "varied", "values": sorted(map(str, s.unique())), "n": int(len(s))}
    return out


def _py(v):
    return v.item() if hasattr(v, "item") else v


def envelope(tests: pd.DataFrame) -> pd.DataFrame:
    g = tests.groupby(["psi_id", "investigation", "table_id", "fuel_phase", "fuel", "oxidizer_location"], dropna=False)
    agg = g.agg(n_tests=("record_uid", "size"), n_labeled=("has_outcome_label", "sum"),
                o2_min=("o2_mole_frac", "min"), o2_max=("o2_mole_frac", "max"),
                pressure_kpa_min=("pressure_kpa", "min"), pressure_kpa_max=("pressure_kpa", "max"),
                ppo2_kpa_min=("ppo2_kpa", "min"), ppo2_kpa_max=("ppo2_kpa", "max"),
                flow_cm_s_min=("flow_velocity_cm_s", "min"), flow_cm_s_max=("flow_velocity_cm_s", "max")).reset_index()
    agg["n_labeled"] = agg["n_labeled"].astype("int64")
    return agg


def pivot(tests: pd.DataFrame, meas: pd.DataFrame, table_ids: list[str], variables: list[str], cols: list[str]) -> pd.DataFrame:
    """Model-ready wide table: core columns + scalar measurement variables as columns."""
    base = tests[tests.table_id.isin(table_ids)][cols].copy()
    m = meas[meas.variable.isin(variables) & (meas.seq_index == 0) & meas.record_uid.isin(base.record_uid)]
    wide = m.pivot_table(index="record_uid", columns="variable", values="value_num", aggfunc="first")
    wide = wide.reindex(columns=variables)
    return base.merge(wide, left_on="record_uid", right_index=True, how="left").reset_index(drop=True)


def insights(tests: pd.DataFrame, camp: pd.DataFrame, meas: pd.DataFrame) -> dict:
    amb = tests[tests.oxidizer_location == "ambient"]
    by_phase = {}
    for phase, d in amb.groupby("fuel_phase"):
        o2 = d.o2_mole_frac.dropna()
        pr = d.pressure_kpa.dropna()
        both = d.dropna(subset=["o2_mole_frac", "pressure_kpa"])
        gaps = {}
        for name, (x, p, src) in TARGET_ATMOSPHERES.items():
            entry = {"target_o2": x, "target_kpa": p, "source": src,
                     "n_tests_o2_at_or_above_target": int((o2 >= x - 1e-9).sum()),
                     "nearest_o2_gap_points": round(float((o2 - x).abs().min()) * 100, 2) if len(o2) else None}
            if len(both):
                dist = ((both.o2_mole_frac - x) / 0.05) ** 2 + ((both.pressure_kpa - p) / 10) ** 2
                near = both.loc[dist.idxmin()]
                entry["nearest_test_o2_and_pressure"] = {
                    "record_uid": near.record_uid, "o2_mole_frac": round(float(near.o2_mole_frac), 4),
                    "pressure_kpa": round(float(near.pressure_kpa), 2),
                    "delta_o2_points": round(float(near.o2_mole_frac - x) * 100, 2),
                    "delta_kpa": round(float(near.pressure_kpa - p), 2),
                    "metric": "scaled distance, 5 O2 points = 10 kPa"}
            else:
                entry["nearest_test_o2_and_pressure"] = None
            gaps[name] = entry
        by_phase[phase] = {
            "n_tests": int(len(d)), "n_with_o2": int(len(o2)), "n_with_pressure": int(len(pr)),
            "o2_min": round(float(o2.min()), 4) if len(o2) else None, "o2_max": round(float(o2.max()), 4) if len(o2) else None,
            "pressure_kpa_min": round(float(pr.min()), 2) if len(pr) else None,
            "pressure_kpa_max": round(float(pr.max()), 2) if len(pr) else None,
            "target_atmospheres": gaps}
    labels = (tests[tests.has_outcome_label].groupby("table_id")
              .agg(n=("record_uid", "size"), ignited_true=("ignited", lambda s: int((s == True).sum())),  # noqa: E712
                   ignited_false=("ignited", lambda s: int((s == False).sum()))).to_dict("index"))  # noqa: E712
    for t, d in labels.items():
        d["outcomes"] = {k: int(v) for k, v in tests[tests.table_id == t].outcome.value_counts().items()}
    flag_counts = pd.Series([f.split(":")[0] for fl in tests["flags"] for f in fl]).value_counts()
    return {
        "note": "Descriptive facts computed from the gold tables. Ambient-oxidizer tests only for atmosphere statistics.",
        "n_tests": int(len(tests)), "n_campaign_records": int(len(camp)), "n_measurements": int(len(meas)),
        "n_investigations_with_tests": int(tests.psi_id.nunique()),
        "tests_by_fuel_phase": {k: int(v) for k, v in tests.fuel_phase.value_counts().items()},
        "atmosphere_coverage_by_fuel_phase": by_phase,
        "labels_by_table": labels,
        "tests_without_any_outcome_label": int((~tests.has_outcome_label).sum()),
        "gravity": {"levels_present": sorted(x for x in tests.gravity_level.dropna().unique()),
                    "n_partial_gravity_tests": int(((tests.gravity_g > 0) & (tests.gravity_g < 1)).sum()),
                    "basis": "gravity_g comes from investigation metadata, not from the tables"},
        "record_flags": {k: int(v) for k, v in flag_counts.items()},
    }


def dumps(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False, default=_py) + "\n"
