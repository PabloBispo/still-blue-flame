"""Generated documentation: data dictionary and data-quality report. Rendered from the contract, the maps and the data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .contract import (FIELDS, FLAG_GLOSSARY, GRAINS, MEASUREMENT_COLUMNS, PIPELINE_VERSION,
                       PROVENANCE_COLUMNS)
from .engine import _targets

KEY_FIELDS = ["fuel", "o2_mole_frac", "pressure_kpa", "flow_velocity_cm_s", "gravity_g", "doi", "source_row"]
T1_FIELDS = ["fuel", "o2_mole_frac", "pressure_kpa"]
SOURCE_OF = {"fuel": "fuel_raw"}


def crosswalk(maps: list[dict]) -> pd.DataFrame:
    rows = []
    for spec in maps:
        for k, v in spec.get("constants", {}).items():
            rows.append({"target_kind": "field", "target": k, "table_id": spec["table_id"], "psi_id": spec["psi_id"],
                         "source_column": "(constant)", "unit_raw": None, "parser": f"constant = {v}", "note": None})
        for col, tg in spec["columns"].items():
            for t in _targets(tg):
                if "ignore" in t:
                    rows.append({"target_kind": "ignored", "target": None, "table_id": spec["table_id"], "psi_id": spec["psi_id"],
                                 "source_column": col, "unit_raw": None, "parser": None, "note": t["ignore"]})
                    continue
                for kind in ("field", "variable"):
                    if kind in t:
                        rows.append({"target_kind": kind, "target": t[kind], "table_id": spec["table_id"],
                                     "psi_id": spec["psi_id"], "source_column": col, "unit_raw": t.get("unit"),
                                     "parser": t.get("parser", "text"), "note": ", ".join(t.get("flags", [])) or None})
    return pd.DataFrame(rows)


def _has_source(spec: dict, field: str) -> bool:
    f = SOURCE_OF.get(field, field)
    if f in spec.get("constants", {}):
        return True
    return any(t.get("field") == f for tg in spec["columns"].values() for t in _targets(tg))


def quality(maps, stats, tests, camp, meas, rejects, raw_dir: Path) -> dict:
    by_id = {m["table_id"]: m for m in maps}
    psi_tables = sorted(str(p.relative_to(raw_dir)) for p in (raw_dir / "files").glob("*/Experimental table/*.csv"))
    used = {m["source"].get("file") for m in maps}
    superseded = {s["file"]: m["table_id"] for m in maps for s in m.get("supersedes", [])}
    accounted = [{"file": f, "status": "loaded" if f in used else ("superseded" if f in superseded else "MISSING"),
                  "by": superseded.get(f)} for f in psi_tables]
    per_table = []
    cells_ok = cells_total = 0
    for st in stats:
        spec = by_id[st["table_id"]]
        df = tests[tests.table_id == st["table_id"]] if st["grain"] == "test" else camp[camp.table_id == st["table_id"]]
        row = dict(st)
        row["fill"] = {}
        for f in KEY_FIELDS:
            n = int(df[f].notna().sum())
            in_source = True if f in ("gravity_g", "doi", "source_row") else _has_source(spec, f)
            row["fill"][f] = {"n": n, "pct": round(100 * n / len(df), 1) if len(df) else None, "in_source": in_source}
            if st["grain"] == "test" and f in T1_FIELDS and in_source:
                cells_ok += n
                cells_total += len(df)
        row["n_flagged_records"] = int(df["flags"].map(len).gt(0).sum())
        per_table.append(row)
    n = len(tests)
    strict = int(tests[T1_FIELDS].notna().all(axis=1).sum())
    t1 = {
        "definition": "Test-grain rows only. Fields: fuel, o2_mole_frac, pressure_kpa. Source lineage (file, row, sha256) is 100% by construction.",
        "n_tests": n,
        "per_field_pct_all_rows": {f: round(100 * tests[f].notna().mean(), 1) for f in T1_FIELDS},
        "strict_all_three_pct": round(100 * strict / n, 1), "strict_all_three_n": strict,
        "source_aware_pct": round(100 * cells_ok / cells_total, 1),
        "source_aware_definition": "Share of cells filled, counting only tables whose source actually carries the field. "
                                   "Measures the harmonization; the strict number measures what PSI published.",
        "source_aware_cells": [cells_ok, cells_total],
    }
    flags = pd.Series([f.split(":")[0] for fl in pd.concat([tests["flags"], camp["flags"]]) for f in fl]).value_counts()
    return {
        "pipeline_version": PIPELINE_VERSION, "t1": t1,
        "totals": {"source_rows": sum(s["rows_total"] for s in stats), "blank_rows": sum(s["rows_blank"] for s in stats),
                   "dropped_rows": sum(sum(s["rows_dropped"].values()) for s in stats),
                   "records": len(tests) + len(camp), "tests": len(tests), "campaign_records": len(camp),
                   "measurements": len(meas), "rejects": len(rejects),
                   "source_columns": sum(s["n_source_columns"] for s in stats),
                   "source_columns_ignored": sum(s["n_columns_ignored"] for s in stats)},
        "psi_experimental_tables": accounted,
        "tables": per_table,
        "record_flags": {k: int(v) for k, v in flags.items()},
        "rejects": rejects.to_dict("records"),
        "checks": {
            "record_uid_unique": bool(pd.concat([tests.record_uid, camp.record_uid]).is_unique),
            "all_psi_tables_accounted": all(a["status"] != "MISSING" for a in accounted),
            "every_source_column_mapped_or_ignored": True,  # enforced by engine.validate_map
            "measurements_reference_existing_records": bool(meas.record_uid.isin(pd.concat([tests.record_uid, camp.record_uid])).all()),
            "gravity_only_from_metadata": bool(tests.gravity_source.str.startswith("PSI metadata").all()),
        },
    }


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    esc = lambda v: "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(esc(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def dictionary_csv(maps, variables, tests) -> pd.DataFrame:
    cw = crosswalk(maps)
    rows = []
    for f in FIELDS:
        src = cw[(cw.target_kind == "field") & (cw.target == SOURCE_OF.get(f.name, f.name))]
        rows.append({"table": "tests / campaigns", "name": f.name, "type": f.dtype, "unit": f.unit, "origin": f.origin,
                     "valid_range": f"[{f.lo}, {f.hi}]" if f.lo is not None else None,
                     "allowed_values": " | ".join(f.enum) if f.enum else None,
                     "fill_pct_tests": round(100 * tests[f.name].notna().mean(), 1),
                     "n_source_tables": int(src.table_id.nunique()), "description": f.description})
    for name, v in variables.items():
        src = cw[(cw.target_kind == "variable") & (cw.target == name)]
        rows.append({"table": "measurements", "name": name, "type": "float / text", "unit": v.get("unit"), "origin": "source",
                     "valid_range": None, "allowed_values": None, "fill_pct_tests": None,
                     "n_source_tables": int(src.table_id.nunique()), "description": f"[{v['category']}] {v['description']}"})
    return pd.DataFrame(rows)


def dictionary_md(maps, variables, tests, camp, meas) -> str:
    cw = crosswalk(maps)
    out = [f"# Data dictionary: Flame in Freefall unified combustion dataset\n",
           f"Generated by `flame.etl.pipeline` v{PIPELINE_VERSION} from the data contract (`flame/etl/contract.py`), "
           "the curated maps (`flame/harmonize/maps/`) and the vocabularies (`flame/harmonize/vocab/`). Do not edit by hand.\n",
           "## 1. Files\n",
           _md_table(pd.DataFrame([
               ("tests.parquet / .csv", len(tests), "One row per test (burn / run). The analytic table."),
               ("campaigns.parquet / .csv", len(camp), "Rows that are not single tests: test days, planned matrices, sessions, data products. Same columns as tests."),
               ("measurements.parquet / .csv", len(meas), "Long table: every other reported quantity, one value per row, keyed by record_uid."),
               ("provenance.parquet", "", "For each core value: source column, raw cell text, raw unit, parser, flags."),
               ("investigations.json", 24, "One card per PSI combustion investigation: metadata, publications, files, what varied and what was fixed."),
               ("marts/*.parquet, insights.json", "", "Model-ready and analytic views derived from the three tables above."),
               ("rejects.csv", "", "Values that failed the contract range check. Never silently corrected."),
               ("bronze/cells.parquet", "", "Every non-empty source cell as text, with file hash and row number."),
           ], columns=["file", "rows", "content"])), "",
           "Join key everywhere: `record_uid`.\n",
           "## 2. Gravity: where it comes from\n",
           "> `gravity_g` is **not** read from any experimental table. No PSI combustion table has a gravity column. "
           "The value is assigned from the investigation metadata record (`flightProgram`): investigations flown on the "
           "ISS or on Cygnus get `gravity_g = 0.0` and `gravity_level = microgravity`. It is a nominal label, not a measured "
           "acceleration. `gravity_source` stores the literal metadata text used. Investigations without a flight platform "
           "get null. The 1 g reference results of Saffire-II are separate measurement variables (`burn_length_1g`, "
           "`spread_rate_1g`), not rows with `gravity_g = 1`. There is no partial-gravity record in the dataset.\n",
           "## 3. Record grains\n",
           _md_table(pd.DataFrame([(g, int((pd.concat([tests, camp]).record_grain == g).sum())) for g in GRAINS],
                                  columns=["record_grain", "rows"])), "",
           "## 4. Core columns (tests.parquet, campaigns.parquet)\n",
           "Origin: `source` = read from a table column; `constant` = fixed per table in its map; `metadata` = PSI "
           "investigation record; `vocabulary` = normalised by a curated vocabulary; `derived` = computed; `system` = lineage.\n"]
    d = dictionary_csv(maps, variables, tests)
    core = d[d.table != "measurements"][["name", "type", "unit", "origin", "valid_range", "fill_pct_tests", "description", "allowed_values"]]
    core = core.rename(columns={"fill_pct_tests": "filled % (tests)"})
    out += [_md_table(core), "", "## 5. measurements.parquet columns\n",
            _md_table(pd.DataFrame(MEASUREMENT_COLUMNS.items(), columns=["column", "description"])), "",
            "## 6. provenance.parquet columns\n",
            _md_table(pd.DataFrame(PROVENANCE_COLUMNS.items(), columns=["column", "description"])), "",
            "## 7. Semantic crosswalk: which source columns feed each core column\n",
            "This is the harmonization itself. Different investigations name the same physical quantity differently and in "
            "different units; each line below is one curated decision.\n"]
    fields = cw[(cw.target_kind == "field") & (cw.source_column != "(constant)")]
    for f in FIELDS:
        sub = fields[fields.target == f.name]
        if sub.empty or f.name in ("test_label", "notes"):
            continue
        out.append(f"**{f.name}** ({f.unit or 'text'})\n")
        out.append(_md_table(sub[["table_id", "psi_id", "source_column", "unit_raw", "parser", "note"]]
                             .rename(columns={"unit_raw": "source unit"})) + "\n")
    out += ["## 8. Measurement variables (controlled vocabulary)\n"]
    counts = meas.groupby("variable").agg(n=("record_uid", "size"), n_numeric=("value_num", "count")).to_dict("index")
    rows = []
    for name, v in sorted(variables.items(), key=lambda kv: (kv[1]["category"], kv[0])):
        src = cw[(cw.target_kind == "variable") & (cw.target == name)]
        rows.append((v["category"], name, v.get("unit"), counts.get(name, {}).get("n", 0), counts.get(name, {}).get("n_numeric", 0),
                     ", ".join(sorted(src.table_id.unique())), v["description"]))
    out += [_md_table(pd.DataFrame(rows, columns=["category", "variable", "unit", "values", "numeric", "tables", "description"])), "",
            "## 9. Flag glossary\n", _md_table(pd.DataFrame(FLAG_GLOSSARY.items(), columns=["flag", "meaning"])), "",
            "Other flags: `ramped` (flow ramped during the test, no single value), `potentiometer_setting` (fan value is a pot "
            "setting, different scale), `size_from_label` (thickness or diameter read from the material text), "
            "`burn_time_from_clock` (burn time = flame-off minus ignition clock time), `duplicate_label` (same test id twice "
            "in the source; uid suffixed).\n",
            "## 10. Source tables and semantic notes\n"]
    for spec in maps:
        out.append(f"### {spec['table_id']} ({spec['psi_id']}, grain: {spec['grain']})\n")
        src = spec["source"]
        where = src.get("file") or f"{src['dir']}/{src['prefix']}*.txt"
        out.append(f"{spec['title']}. Source: `{where}`"
                   + (f", sheet(s) `{src.get('sheet') or ', '.join(src.get('sheets', []))}`" if where.endswith("xlsx") else "")
                   + (f", columns {src['usecols']}" if src.get("usecols") else "") + ".\n")
        if spec.get("unlisted"):
            out.append(f"- Columns not listed in the map are not loaded: {spec['unlisted']['ignore']}")
        for s in spec.get("supersedes", []):
            out.append(f"- Supersedes `{s['file']}`: {s['reason']}")
        for n in spec.get("semantic_notes", []):
            out.append(f"- {n}")
        ign = cw[(cw.table_id == spec["table_id"]) & (cw.target_kind == "ignored")]
        if len(ign):
            out.append("- Columns not loaded: " + "; ".join(f"`{r.source_column}` ({r.note})" for r in ign.itertuples()))
        out.append("")
    return "\n".join(out) + "\n"


def quality_md(q: dict, ins: dict) -> str:
    t1, tot = q["t1"], q["totals"]
    out = ["# Data quality report\n",
           f"Generated by `flame.etl.pipeline` v{q['pipeline_version']}. Numbers are computed from the outputs, not typed.\n",
           "## T1. Harmonization coverage\n", t1["definition"] + "\n",
           _md_table(pd.DataFrame([
               ("Tests", t1["n_tests"]),
               *[(f"{f} filled, all tests", f"{v}%") for f, v in t1["per_field_pct_all_rows"].items()],
               ("All three filled (strict)", f"{t1['strict_all_three_pct']}% ({t1['strict_all_three_n']} of {t1['n_tests']})"),
               ("Filled where the source carries the field (source-aware)",
                f"{t1['source_aware_pct']}% ({t1['source_aware_cells'][0]} of {t1['source_aware_cells'][1]} cells)"),
           ], columns=["metric", "value"])), "", t1["source_aware_definition"] + "\n",
           "## Totals\n", _md_table(pd.DataFrame(tot.items(), columns=["item", "n"])), "",
           "## Checks\n", _md_table(pd.DataFrame([(k, "pass" if v else "FAIL") for k, v in q["checks"].items()], columns=["check", "result"])), "",
           "## PSI experimental tables accounted for\n",
           _md_table(pd.DataFrame(q["psi_experimental_tables"])), "",
           "## Maps skipped because their source files are not in the raw directory\n",
           _md_table(pd.DataFrame(q.get("skipped_maps", []))) if q.get("skipped_maps") else "None.", "",
           "## Per table\n"]
    rows = []
    for t in q["tables"]:
        f = t["fill"]
        cell = lambda k: (f"{f[k]['pct']}%" if f[k]["in_source"] else "not in source") if f[k]["pct"] is not None else ""
        rows.append((t["table_id"], t["psi_id"], t["grain"], t["rows_total"], t["rows_blank"],
                     sum(t["rows_dropped"].values()), t["rows_kept"], f"{t['n_source_columns'] - t['n_columns_ignored']}/{t['n_source_columns']}",
                     cell("fuel"), cell("o2_mole_frac"), cell("pressure_kpa"), cell("flow_velocity_cm_s"), t["n_flagged_records"]))
    out += [_md_table(pd.DataFrame(rows, columns=["table", "psi", "grain", "source rows", "blank", "dropped", "kept",
                                                  "columns loaded", "fuel", "o2", "pressure", "flow", "flagged records"])), "",
            "Dropped rows and why:\n"]
    for t in q["tables"]:
        for reason, n in t["rows_dropped"].items():
            out.append(f"- {t['table_id']}: {n} ({reason})")
    out += ["", "## Flags raised (records)\n", _md_table(pd.DataFrame(q["record_flags"].items(), columns=["flag", "records"])), "",
            "## Rejected values (range check)\n",
            _md_table(pd.DataFrame(q["rejects"])) if q["rejects"] else "None.", "",
            *_log_validation_md(q),
            "## Atmosphere coverage (ambient-oxidizer tests)\n"]
    rows = []
    for phase, d in ins["atmosphere_coverage_by_fuel_phase"].items():
        for name, g in d["target_atmospheres"].items():
            near = g["nearest_test_o2_and_pressure"]
            rows.append((phase, d["n_with_o2"], d["o2_min"], d["o2_max"], name, f"{g['target_o2']:.2f} / {g['target_kpa']} kPa",
                         g["n_tests_o2_at_or_above_target"], g["nearest_o2_gap_points"],
                         f"{near['record_uid']} ({near['delta_o2_points']:+} pts, {near['delta_kpa']:+} kPa)" if near else "no test with O2 and pressure"))
    out += [_md_table(pd.DataFrame(rows, columns=["fuel phase", "tests with O2", "O2 min", "O2 max", "target", "target O2 / P",
                                                  "tests at or above target O2", "nearest O2 gap (points)", "nearest test in O2 and pressure"])), ""]
    return "\n".join(out) + "\n"


def _log_validation_md(q: dict) -> list[str]:
    v = q.get("log_rule_validation_cfig")
    if not v:
        return []
    tab = pd.DataFrame(v["crosstab_pi_rows_log_cols"]).T.fillna(0).astype(int)
    tab.insert(0, "PI table outcome", tab.index)
    w = v["log_window_minus_pi_burn_time_s"]
    return ["## Script-log outcome rule, checked against the CFI-G PI tables\n",
            f"{v['n_with_log']} of {v['n_tests']} CFI-G tests have a script log; {v['n_compared']} have an outcome on both sides. "
            f"Sustained versus not sustained agrees in {v['binary_agreement_pct']}% ({v['binary_agreement_n'][0]} of {v['binary_agreement_n'][1]}). "
            f"When the log says not sustained the PI agrees in {v['log_says_not_sustained']['pi_agrees']} of {v['log_says_not_sustained']['n']}; "
            f"when the log says sustained (flow terminated or temperature abort) the PI agrees in {v['log_says_sustained']['pi_agrees']} of {v['log_says_sustained']['n']}.\n",
            _md_table(tab), "",
            f"{v['reading']} Log window minus PI burn time: median {w['median']:.0f} s (quartiles {w['q25']:.0f} to {w['q75']:.0f}, n = {w['n']}).\n"]
