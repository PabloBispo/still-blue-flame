"""Exploratory data analysis of the gold dataset, plus a data-utilisation audit.

    uv run python -m flame.analysis.eda --data data/processed --raw ../flame-freefall/data/raw

Writes data/processed/eda/eda.json (every number and chart series) and docs/eda.md.
Descriptive only: counts, ranges, rates with n. Hypothesis tests belong to the analysis block.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HARMONIZE = Path(__file__).resolve().parent.parent / "harmonize"
TARGETS = {"ISS": (0.21, 101.3), "Statement 30% / 10.2 psia": (0.30, 70.3), "Exploration 34% / 8.2 psia": (0.34, 56.5)}
KEY = ["fuel", "fuel_mole_frac", "geometry", "thickness_mm", "char_diameter_mm", "o2_mole_frac", "o2_mole_frac_final",
       "n2_mole_frac", "co2_mole_frac", "he_mole_frac", "pressure_kpa", "ppo2_kpa", "flow_velocity_cm_s", "flow_direction",
       "ignited", "outcome", "burn_time_s", "test_date", "campaign_day", "pi", "notes"]
OUTCOME_WORDS = {
    "extinction": r"extinguish|extinct|went out|goes out|winked out|blew out|blow ?out|blow ?off|quench|\bout at\b|flame out",
    "no ignition": r"no ignition|did not ignite|didn'?t ignite|failed to ignite|not ignite|no ign\b",
    "burned out / consumed": r"burn(ed|t)? out|burned completely|consumed|burn(ed)? to (the )?end|all the way",
    "sustained / spread": r"spread|sustain|steady|stable flame|nice (burn|flame)",
    "burn time stated": r"\b\d+\s?(s|sec|seconds|min|minutes)\b",
    "suppression (N2)": r"\bN2\b|GN2|nitrogen",
}


def r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def rate_by_bin(df, col, bins, label, positive):
    out = []
    cut = pd.cut(df[col], bins=bins, include_lowest=True)
    for iv, d in df.groupby(cut, observed=True):
        y = d[label].dropna()
        if len(y):
            out.append({"bin": f"{iv.left:g}-{iv.right:g}", "mid": r((iv.left + iv.right) / 2), "n": int(len(y)),
                        "k": int((y == positive).sum()), "rate": r((y == positive).mean(), 3)})
    return out


def structure(t, c, m):
    tables = []
    for tid, d in pd.concat([t, c]).groupby("table_id"):
        tables.append({"table_id": tid, "psi_id": d.psi_id.iloc[0], "investigation": d.investigation.iloc[0],
                       "grain": d.record_grain.iloc[0], "rows": int(len(d)),
                       "measurements": int(m.record_uid.isin(d.record_uid).sum()),
                       "fill_pct": {k: r(100 * d[k].notna().mean(), 1) for k in KEY}})
    return {"n_tests": int(len(t)), "n_campaign": int(len(c)), "n_measurements": int(len(m)),
            "n_core_columns": int(t.shape[1]), "n_variables_used": int(m.variable.nunique()),
            "measurements_numeric_pct": r(100 * m.value_num.notna().mean(), 1),
            "tests_by_phase": {k: int(v) for k, v in t.fuel_phase.value_counts().items()},
            "tests_by_fuel": {k: int(v) for k, v in t.fuel.value_counts().items()},
            "tests_by_geometry": {("not stated" if pd.isna(k) else k): int(v) for k, v in t.geometry.value_counts(dropna=False).items()},
            "tests_by_year": {k: int(v) for k, v in t.test_date.str[:4].value_counts().sort_index().items()},
            "tests_with_date": int(t.test_date.notna().sum()),
            "tables": tables}


def utilisation(data: Path, raw: Path | None, t, c, m):
    """Is anything dropped that carries information?"""
    cells = pd.read_parquet(data / "bronze/cells.parquet")
    maps = {p.stem: yaml.safe_load(p.read_text()) for p in sorted((HARMONIZE / "maps").glob("*.yaml"))}
    kept_rows = pd.concat([t, c])[["table_id", "source_sheet", "source_row"]].assign(kept=True)
    cells = cells.merge(kept_rows, how="left", left_on=["table_id", "source_sheet", "source_row"],
                        right_on=["table_id", "source_sheet", "source_row"])
    cells["kept"] = cells.kept.eq(True)
    ignored = {(tid, col): tg["ignore"] for tid, spec in maps.items() for col, tg in spec["columns"].items()
               if isinstance(tg, dict) and "ignore" in tg}
    listed = {(tid, col) for tid, spec in maps.items() for col in spec["columns"]}
    unlisted = {tid: spec["unlisted"]["ignore"] for tid, spec in maps.items() if spec.get("unlisted")}
    cells["ignore_reason"] = [ignored.get((a, b)) or (unlisted.get(a) if (a, b) not in listed else None)
                              for a, b in zip(cells.table_id, cells.column)]
    cells["status"] = np.where(~cells.kept, "row dropped", np.where(cells.ignore_reason.notna(), "column ignored", "loaded"))
    by_status = cells.status.value_counts()
    ign = (cells[cells.status == "column ignored"].groupby(["table_id", "column", "ignore_reason"]).size()
           .reset_index(name="cells").sort_values("cells", ascending=False))
    ign["kind"] = np.where(ign.ignore_reason.str.contains("configuration|files merged", case=False), "script configuration (ACME logs)",
                  np.where(ign.ignore_reason.str.contains("camera", case=False), "camera setting",
                           np.where(ign.ignore_reason.str.contains("duplicate|redundant|constant|label|index", case=False),
                                    "redundant with a loaded column", "other")))
    # numeric columns whose cells did not parse to a number
    prov = pd.read_parquet(data / "provenance.parquet")
    bad = prov[prov["flags"].map(lambda f: any(x in f for x in ("non_numeric", "ramped")) or any(x.startswith("out_of_range") for x in f))]
    lost_numeric = bad.groupby(["field", "raw_value"]).size().reset_index(name="n").sort_values("n", ascending=False)
    # free text that carries structured information
    notes = {}
    for tid, d in t.groupby("table_id"):
        s = d.notes.dropna()
        if len(s) < 5:
            continue
        notes[tid] = {"n_tests": int(len(d)), "n_with_notes": int(len(s)), "median_chars": int(s.str.len().median()),
                      "keyword_tests": {k: int(s.str.contains(p, case=False, regex=True).sum()) for k, p in OUTCOME_WORDS.items()}}
    # extinction marked inside the flow / fan sequences of BASS and BASS-II
    solid = t[t.table_id.isin(["bass", "bass2"])]
    rawj = solid.raw_json.map(json.loads)
    air = rawj.map(lambda d: d.get("Air display", ""))
    fan = rawj.map(lambda d: d.get("Fan display", ""))
    has_ext = air.str.contains(r"ext", case=False) | fan.str.contains(r"ext", case=False)
    seq = m[m.variable == "flow_velocity_seq"].groupby("record_uid").agg(steps=("value_num", "count"), last=("value_num", "last"),
                                                                           lowest=("value_num", "min"))
    seq = seq.reindex(solid.record_uid)
    ext_points = {"tests": int(len(solid)), "with_flow_sequence_2plus": int((seq.steps >= 2).sum()),
                  "sequence_ends_at_zero_or_below_1": int((seq.lowest < 1).sum()),
                  "cell_says_ext": int(has_ext.sum()),
                  "cell_says_ext_by_table": {k: int(v) for k, v in has_ext.groupby(solid.table_id.values).sum().items()},
                  "examples": sorted(set(air[air.str.contains("ext", case=False)]))[:6]}
    out = {"cells_total": int(len(cells)), "cells_by_status": {k: int(v) for k, v in by_status.items()},
           "cells_loaded_pct": r(100 * (cells.status == "loaded").mean(), 2),
           "ignored_by_kind": {k: int(v) for k, v in ign.groupby("kind").cells.sum().items()},
           "ignored_columns": ign.to_dict("records"),
           "numeric_cells_not_parsed": lost_numeric.head(25).to_dict("records"),
           "numeric_cells_not_parsed_total": int(lost_numeric.n.sum()),
           "notes": notes, "extinction_in_sequences": ext_points}
    if raw is not None and (raw / "inventory.csv").exists():
        inv = pd.read_csv(raw / "inventory.csv")
        inv["ext"] = inv.ext.str.lower()
        inv["downloaded"] = inv.local_path.notna()
        if (raw / "fetch_log.csv").exists():
            got = pd.read_csv(raw / "fetch_log.csv")
            got = set(zip(got.psi_id, got.file_name))
            inv["downloaded"] |= np.array([(a, b) in got for a, b in zip(inv.psi_id, inv.file_name)])
        tab = inv[inv.ext.isin([".csv", ".xlsx", ".xls", ".txt", ".dat", ".asc"]) & ~inv.downloaded & (inv.category != "metadata")]
        g = (tab.groupby(["psi_id", "acronym", "category"], dropna=False)
             .agg(files=("file_name", "size"), mb=("size_bytes", lambda s: round(s.sum() / 1e6, 1)),
                  median_kb=("size_bytes", lambda s: round(s.median() / 1e3, 1))).reset_index().sort_values("files", ascending=False))
        g["acronym"] = g.acronym.fillna("")
        out["inventory"] = {"files_total": int(len(inv)), "files_downloaded": int(inv.downloaded.sum()),
                            "tabular_not_downloaded_files": int(len(tab)), "tabular_not_downloaded_gb": r(tab.size_bytes.sum() / 1e9, 2),
                            "by_investigation": g.to_dict("records")}
    return out


def atmospheres(t):
    amb = t[(t.oxidizer_location == "ambient") & t.o2_mole_frac.notna()]
    pts = amb.dropna(subset=["pressure_kpa"])
    grid = (pts.assign(o2=pts.o2_mole_frac.round(3), p=pts.pressure_kpa.round(0))
            .groupby(["fuel_phase", "o2", "p"]).size().reset_index(name="n"))
    hist = {}
    edges = np.arange(0.10, 0.4201, 0.02)
    for ph, d in amb.groupby("fuel_phase"):
        h, _ = np.histogram(d.o2_mole_frac.clip(upper=0.4199), bins=edges)
        hist[ph] = [int(x) for x in h]
    pp = {}
    for ph, d in pts.groupby("fuel_phase"):
        q = d.ppo2_kpa.quantile([0, 0.25, 0.5, 0.75, 1])
        pp[ph] = {"n": int(len(d)), "min": r(q.iloc[0], 2), "q25": r(q.iloc[1], 2), "median": r(q.iloc[2], 2),
                  "q75": r(q.iloc[3], 2), "max": r(q.iloc[4], 2),
                  "n_within_2kpa_of_19_to_21": int(d.ppo2_kpa.between(17.2, 23.3).sum())}
    return {"n_ambient_with_o2": int(len(amb)), "n_ambient_with_o2_and_pressure": int(len(pts)),
            "o2_hist_edges": [r(x, 2) for x in edges], "o2_hist": hist,
            "o2_pressure_points": grid.to_dict("records"),
            "targets": [{"name": k, "o2": v[0], "kpa": v[1], "ppo2_kpa": r(v[0] * v[1], 2)} for k, v in TARGETS.items()],
            "ppo2_by_phase": pp,
            "n_tests_o2_ge_30": {ph: int((d.o2_mole_frac >= 0.2999).sum()) for ph, d in amb.groupby("fuel_phase")},
            "n_tests_pressure_le_70": {ph: int((d.pressure_kpa <= 70.3).sum()) for ph, d in pts.groupby("fuel_phase")},
            "n_tests_o2_ge_30_and_p_le_70": {ph: int(((d.o2_mole_frac >= 0.2999) & (d.pressure_kpa <= 70.3)).sum())
                                             for ph, d in pts.groupby("fuel_phase")}}


def cfig(data: Path):
    d = pd.read_parquet(data / "marts/model_cfig.parquet")
    out = {}
    for ft, x in d.groupby("flame_type"):
        ign = x[x.ignited == True]  # noqa: E712
        bins = [0.1, 0.16, 0.2, 0.22, 0.26, 0.3, 0.34, 0.38, 0.42] if ft == "normal" else [0.1, 0.14, 0.2, 0.25, 0.35, 0.5, 0.7, 0.9]
        out[ft] = {
            "n": int(len(x)), "o2_range": [r(x.o2_mole_frac.min()), r(x.o2_mole_frac.max())],
            "pressure_levels_kpa": {str(k): int(v) for k, v in x.pressure_kpa.round(0).value_counts().sort_index().items()},
            "ignited": {str(k): int(v) for k, v in x.ignited.value_counts(dropna=False).items()},
            "outcome": {str(k): int(v) for k, v in x.outcome.value_counts(dropna=False).items()},
            "ignition_rate_by_o2": rate_by_bin(x, "o2_mole_frac", bins, "ignited", True),
            "self_ext_rate_by_o2": rate_by_bin(ign, "o2_mole_frac", bins, "outcome", "self_extinguished"),
            "self_ext_rate_by_pressure": [{"kpa": r(k, 0), "n": int(len(g)), "rate": r((g.outcome == "self_extinguished").mean(), 3)}
                                          for k, g in ign.groupby(ign.pressure_kpa.round(0))],
            "self_ext_rate_by_fuel": [{"fuel": k, "n": int(len(g)), "rate": r((g.outcome == "self_extinguished").mean(), 3)}
                                      for k, g in ign.groupby("fuel")],
            "burn_time_s": {"median_self_ext": r(ign[ign.outcome == "self_extinguished"].burn_time_s.median(), 1),
                            "median_flow_term": r(ign[ign.outcome == "flow_terminated"].burn_time_s.median(), 1)},
            "points": ign.assign(se=(ign.outcome == "self_extinguished").astype(int))[["o2_mole_frac", "pressure_kpa", "fuel", "se", "burn_time_s"]]
            .round(4).to_dict("records"),
            "spearman_with_self_ext": {k: r(ign[[k]].assign(se=(ign.outcome == "self_extinguished").astype(float)).corr(method="spearman").iloc[0, 1], 3)
                                       for k in ["o2_mole_frac", "pressure_kpa", "fuel_mole_frac", "hc_fuel_mass_flow",
                                                 "adiabatic_flame_temp", "stoich_mixture_fraction", "peak_burner_temp"]
                                       if ign[k].notna().sum() > 20},
        }
    return out


def flex(data: Path):
    d = pd.read_parquet(data / "marts/model_flex.parquet")
    d["ratio"] = d.extinction_diameter / d.droplet_d0
    rows = []
    for (fuel, dil), g in d.groupby(["fuel", "diluent"], dropna=False):
        e = g.dropna(subset=["extinction_diameter"])
        rows.append({"fuel": fuel, "diluent": dil if isinstance(dil, str) else "none reported", "n": int(len(g)),
                     "n_with_extinction_diameter": int(len(e)), "o2_min": r(g.o2_mole_frac.min()), "o2_max": r(g.o2_mole_frac.max()),
                     "median_ext_diameter_mm": r(e.extinction_diameter.median(), 3), "median_ext_over_d0": r(e.ratio.median(), 3),
                     "median_burning_rate": r(g.burning_rate_constant.median(), 3),
                     "outcomes": {k: int(v) for k, v in g.outcome.value_counts().items()}})
    pts = d.dropna(subset=["extinction_diameter"])[["fuel", "diluent", "o2_mole_frac", "co2_mole_frac", "he_mole_frac", "pressure_kpa",
                                                    "droplet_d0", "extinction_diameter", "ratio"]].round(4)
    return {"n": int(len(d)), "by_fuel_diluent": rows, "points": pts.fillna("").to_dict("records"),
            "outcome_by_fuel": {f: {k: int(v) for k, v in g.outcome.value_counts().items()} for f, g in d.groupby("fuel")},
            "pressure_kpa": {"min": r(d.pressure_kpa.min(), 1), "median": r(d.pressure_kpa.median(), 1), "max": r(d.pressure_kpa.max(), 1)},
            "o2_levels": {str(k): int(v) for k, v in d.o2_mole_frac.round(2).value_counts().sort_index().items()}}


def cfi(t, m):
    d = t[t.table_id == "cfi"].set_index("record_uid")
    w = m[m.record_uid.isin(d.index) & m.variable.isin(["extinction_diameter", "extinction_diameter_cool", "burning_rate_constant",
                                                        "burning_rate_constant_cool", "droplet_d0", "hot_flame_burn_time", "reignitions"])]
    w = w.pivot_table(index="record_uid", columns="variable", values="value_num", aggfunc="first")
    d = d.join(w)
    st = m[(m.variable == "analysis_status") & m.record_uid.isin(d.index)].value_text.value_counts()
    rows = []
    for (fuel, p), g in d.groupby(["fuel", d.pressure_kpa.round(0)]):
        rows.append({"fuel": fuel, "kpa": r(p, 0), "n": int(len(g)), "n_hot_ext": int(g.extinction_diameter.notna().sum()),
                     "n_cool_ext": int(g.extinction_diameter_cool.notna().sum()),
                     "median_k_hot": r(g.burning_rate_constant.median(), 3), "median_k_cool": r(g.burning_rate_constant_cool.median(), 3),
                     "median_cool_ext_mm": r(g.extinction_diameter_cool.median(), 3)})
    return {"n": int(len(d)), "analysis_status": {k: int(v) for k, v in st.items()},
            "n_with_results": int(d.extinction_diameter.notna().sum()), "n_with_cool_flame_extinction": int(d.extinction_diameter_cool.notna().sum()),
            "by_fuel_pressure": rows,
            "diluent": {str(k): int(v) for k, v in d.diluent.value_counts(dropna=False).items()},
            "o2_levels": {str(k): int(v) for k, v in d.o2_mole_frac.value_counts().sort_index().items()}}


def solids(data: Path, t, m):
    s = t[(t.fuel_phase == "solid") & t.table_id.isin(["bass", "bass2"])]
    seq = m[m.variable == "flow_velocity_seq"].groupby("record_uid").agg(lowest=("value_num", "min"), steps=("value_num", "count"))
    s = s.join(seq, on="record_uid")
    pts = s.dropna(subset=["o2_mole_frac", "flow_velocity_cm_s"])[["record_uid", "table_id", "fuel", "geometry", "o2_mole_frac",
                                                                   "flow_velocity_cm_s", "lowest", "steps"]].round(4)
    by_mat = [{"fuel": f, "geometry": g if isinstance(g, str) else "not stated", "n": int(len(d)), "o2_min": r(d.o2_mole_frac.min(), 3),
               "o2_max": r(d.o2_mole_frac.max(), 3), "flow_min": r(d.flow_velocity_cm_s.min(), 1), "flow_max": r(d.flow_velocity_cm_s.max(), 1)}
              for (f, g), d in s.groupby(["fuel", "geometry"], dropna=False)]
    b2 = pd.read_parquet(data / "marts/bass2_gas.parquet")
    u = b2[b2.usable_gas_balance]
    gas = {"n": int(len(b2)), "n_usable": int(len(u)), "n_o2_consumed_gt_0p1pt": int((u.o2_consumed > 0.001).sum()),
           "median_o2_consumed_pts": r(100 * u.o2_consumed.median(), 2), "max_o2_consumed_pts": r(100 * u.o2_consumed.max(), 2),
           "n_co_over_range_or_gt500": int((u.co_ppm_final >= 500).sum()),
           "spearman_o2_initial_vs_consumed": r(u[["o2_mole_frac", "o2_consumed"]].corr(method="spearman").iloc[0, 1], 3),
           "spearman_o2_consumed_vs_co2_produced": r(u[["o2_consumed", "co2_produced"]].corr(method="spearman").iloc[0, 1], 3),
           "by_geometry": [{"geometry": g if isinstance(g, str) else "not stated", "n": int(len(d)), "median_o2_consumed_pts": r(100 * d.o2_consumed.median(), 2),
                            "median_co2_produced_pts": r(100 * d.co2_produced.median(), 2), "median_co_final_ppm": r(d.co_ppm_final.median(), 0)}
                           for g, d in u.groupby("geometry", dropna=False)],
           "points": u[["test_label", "pi", "geometry", "o2_mole_frac", "o2_consumed", "co2_produced", "co_ppm_final"]].round(5).fillna("").to_dict("records")}
    saf = t[t.table_id == "saffire2"][["test_label", "fuel", "flow_direction", "o2_mole_frac", "burn_time_s", "record_uid"]]
    sm = m[m.record_uid.isin(saf.record_uid) & m.variable.isin(["burn_length_ug", "burn_length_1g", "spread_rate_ug", "spread_rate_1g"])]
    txt = sm.pivot_table(index="record_uid", columns="variable", values="value_text", aggfunc="first")
    saffire = saf.join(txt, on="record_uid").drop(columns="record_uid").fillna("").to_dict("records")
    return {"n_bass": int((s.table_id == "bass").sum()), "n_bass2": int((s.table_id == "bass2").sum()),
            "o2_flow_points": pts.fillna("").to_dict("records"), "by_material": by_mat, "bass2_gas": gas, "saffire2_pairs": saffire,
            "o2_max_solids": r(t[t.fuel_phase == "solid"].o2_mole_frac.max(), 4)}


def others(t, c, m):
    sp = t[t.table_id == "spice"].join(m[m.variable == "fuel_mass_flow"].set_index("record_uid").value_num.rename("fuel_mg_s"), on="record_uid")
    spice = [{"fuel": f, "fuel_mole_frac": r(x) if x == x else None, "nozzle_mm": r(nz), "n": int(len(g)),
              "air_cm_s": [r(g.flow_velocity_cm_s.min(), 1), r(g.flow_velocity_cm_s.max(), 1)],
              "fuel_mg_s": [r(g.fuel_mg_s.min(), 2), r(g.fuel_mg_s.max(), 2)],
              "spearman_air_vs_fuel": r(g[["flow_velocity_cm_s", "fuel_mg_s"]].corr(method="spearman").iloc[0, 1], 2)}
             for (f, x, nz), g in sp.groupby(["fuel", sp.fuel_mole_frac.fillna(1.0), "char_diameter_mm"])]
    sr = t[t.table_id == "same_r"].join(m[m.variable == "heater_resistance_ratio"].set_index("record_uid").value_num.rename("rt_r0"), on="record_uid")
    same = [{"material": f, "n": int(len(g)), "rt_r0_median": r(g.rt_r0.median(), 3), "rt_r0_min": r(g.rt_r0.min(), 3), "rt_r0_max": r(g.rt_r0.max(), 3),
             "flows_cm_s": sorted(set(g.flow_velocity_cm_s.dropna()))} for f, g in sr.groupby("fuel")]
    nf = m[m.variable == "n_flames_ignited"].merge(c[["record_uid", "investigation", "fuel"]], on="record_uid")
    camp = [{"investigation": k, "test_days": int(len(g)), "flames_ignited": int(g.value_num.sum())} for k, g in nf.groupby("investigation")]
    o2s = m[m.variable == "o2_setpoints"].merge(c[["record_uid", "investigation"]], on="record_uid")
    return {"spice": spice, "same_r": same, "campaign_flames": camp,
            "campaign_o2_setpoints": {k: sorted(set(round(x, 3) for x in g.value_num.dropna())) for k, g in o2s.groupby("investigation")}}


def logs(data: Path, t):
    """Outcomes recovered from the ACME script logs."""
    out = {}
    q = json.loads((data / "quality.json").read_text())
    if "log_rule_validation_cfig" in q:
        out["rule_validation_cfig"] = q["log_rule_validation_cfig"]
    p = data / "marts/model_flame_design.parquet"
    if p.exists():
        f = pd.read_parquet(p)
        f["o2_band"] = np.where(f.o2_mole_frac.isna(), "O2 not reported", np.where(f.o2_mole_frac < 0.30, "below 30%", "30% or more"))
        out["flame_design"] = {"n_tests": int(len(f)), "n_with_log_outcome": int(f.log_outcome.notna().sum()),
                               "by_flame_type_and_o2": [{"flame_type": ft, "o2_band": b, "n": int(len(g)),
                                                         **{k: int(v) for k, v in g.log_outcome.value_counts().items()}}
                                                        for (ft, b), g in f[f.log_outcome.notna()].groupby(["flame_type", "o2_band"])]}
    s = t[t.table_id == "acme_sflame_logs"]
    if len(s):
        out["sflame"] = {"n_tests": int(len(s)), "n_test_days": int(s.campaign_day.nunique()),
                         "outcome": {str(k): int(v) for k, v in s.outcome.value_counts(dropna=False).items()},
                         "by_fuel": [{"fuel": k, "n": int(len(g)), **{str(a): int(b) for a, b in g.outcome.value_counts().items()}} for k, g in s.groupby("fuel")]}
    p = data / "marts/bass2_extinction_points.parquet"
    if p.exists():
        e = pd.read_parquet(p)
        v = e.air_at_extinction_cm_s.notna() | e.get("velocity_from_fan_cm_s", pd.Series(index=e.index, dtype=float)).notna()
        out["bass2_extinction_points"] = {"rows": int(len(e)), "tests": int(e.record_uid.nunique()),
                                          "by_source": {k: int(n) for k, n in e.source.value_counts().items()},
                                          "tests_with_a_velocity": int(e[v].record_uid.nunique())}
    return out


def quality(t):
    cond = ["table_id", "fuel", "fuel_mole_frac", "o2_mole_frac", "pressure_kpa", "flow_velocity_cm_s"]
    dup = t[t.table_id.isin(["cfig_normal", "cfig_inverse", "flex", "cfi"])]
    rep = dup.groupby(cond, dropna=False).size()
    fl = pd.Series([f.split(":")[0] for x in t["flags"] for f in x]).value_counts()
    return {"records_with_any_flag": int(t["flags"].map(len).gt(0).sum()),
            "records_with_suspect_flag": int(t["flags"].map(lambda f: any(x.startswith(("suspect", "out_of_range", "composition", "o2_final")) for x in f)).sum()),
            "flags": {k: int(v) for k, v in fl.items()},
            "repeated_conditions": {"groups": int(len(rep)), "tests": int(rep.sum()), "groups_with_replicates": int((rep > 1).sum()),
                                    "max_replicates": int(rep.max())}}


def run(data: Path, raw: Path | None, docs: Path) -> dict:
    t = pd.read_parquet(data / "tests.parquet")
    c = pd.read_parquet(data / "campaigns.parquet")
    m = pd.read_parquet(data / "measurements.parquet")
    eda = {"structure": structure(t, c, m), "utilisation": utilisation(data, raw, t, c, m), "atmospheres": atmospheres(t),
           "cfig": cfig(data), "flex": flex(data), "cfi": cfi(t, m), "solids": solids(data, t, m), "others": others(t, c, m),
           "logs": logs(data, t), "quality": quality(t)}
    (data / "eda").mkdir(exist_ok=True)
    (data / "eda/eda.json").write_text(json.dumps(eda, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    return eda


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=Path("data/processed"), type=Path)
    ap.add_argument("--raw", default=None, type=Path)
    ap.add_argument("--docs", default=Path("docs"), type=Path)
    a = ap.parse_args()
    eda = run(a.data, a.raw, a.docs)
    print(json.dumps({k: (list(v) if isinstance(v, dict) else v) for k, v in eda.items()}, indent=1)[:1500])


if __name__ == "__main__":
    main()
