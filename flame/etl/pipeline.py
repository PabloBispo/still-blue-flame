"""Run the whole ETL: raw PSI files -> bronze -> silver -> gold -> docs.

    python -m flame.etl.pipeline --raw <data/raw> --out data/processed --docs docs

Pure function of the raw directory: no network, no LLM, no randomness.
To onboard a new table, add one YAML map under flame/harmonize/maps/. No code change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from . import docs, engine, gold, investigations, marts as M, readers, reference
from .contract import MEASUREMENT_COLUMNS, PIPELINE_VERSION, PROVENANCE_COLUMNS

HARMONIZE = Path(__file__).resolve().parent.parent / "harmonize"

CFIG_VARS = ["hc_fuel_mass_flow", "o2_mass_flow", "adiabatic_flame_temp", "stoich_mixture_fraction", "peak_burner_temp"]
FLEX_VARS = ["droplet_d0", "extinction_diameter", "burning_rate_constant"]
BASS2_VARS = ["co2_mole_frac_initial", "co2_mole_frac_final", "co_ppm_initial", "co_ppm_final"]
KEYS = ["record_uid", "psi_id", "investigation", "doi", "table_id", "test_label", "campaign_day"]


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text())


def load_maps() -> list[dict]:
    return [load_yaml(p) for p in sorted((HARMONIZE / "maps").glob("*.yaml"))]


def _list_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["flags"] = df["flags"].apply(lambda x: list(x) if x is not None else [])
    return df


def _write(df: pd.DataFrame, path: Path, csv: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    if csv:
        flat = df.copy()
        for c in flat.columns:
            if flat[c].map(lambda v: isinstance(v, (list, tuple))).any():
                flat[c] = flat[c].map(lambda v: ";".join(v) if isinstance(v, (list, tuple)) else v)
        flat.to_csv(path.with_suffix(".csv"), index=False)


def run(raw_dir: Path, out_dir: Path, docs_dir: Path) -> dict:
    variables = load_yaml(HARMONIZE / "vocab" / "variables.yaml")["variables"]
    fuels = load_yaml(HARMONIZE / "vocab" / "fuels.yaml")
    curated = load_yaml(HARMONIZE / "vocab" / "investigations.yaml")
    meta = investigations.load_metadata(raw_dir)
    maps = load_maps()

    bronze, records, meas, prov, rejects, stats = [], [], [], [], [], []
    skipped = []
    for spec in list(maps):
        src_path = raw_dir / (spec["source"].get("file") or spec["source"]["dir"])
        if not src_path.exists() or (src_path.is_dir() and not any(src_path.glob(spec["source"].get("prefix", "") + "*.txt"))):
            skipped.append({"table_id": spec["table_id"], "source": str(src_path.relative_to(raw_dir)), "reason": "source not present in the raw directory"})
            maps.remove(spec)
            continue
        rows, info = readers.read(raw_dir, spec["source"])
        for r in rows:
            for k, v in r.items():
                if not k.startswith("_") and v != "":
                    bronze.append({"table_id": spec["table_id"], "source_file": info["file"], "source_sha256": info["sha256"],
                                   "source_sheet": r["_sheet"], "source_row": r["_row"], "column": k, "value": v})
        res = engine.harmonize(spec, rows, info, variables)
        records += res["records"]
        meas += res["measurements"]
        prov += res["provenance"]
        rejects += res["rejects"]
        stats.append(res["stats"])

    # inherit declared fields from a test-day summary table of the same investigation
    by_table: dict[str, dict[str, dict]] = {}
    for r in records:
        by_table.setdefault(r["table_id"], {}).setdefault(str(r.get("campaign_day")), r)
    for spec in maps:
        inh = spec.get("inherit")
        if not inh:
            continue
        donors = by_table.get(inh["table_id"], {})
        for r in records:
            if r["table_id"] != spec["table_id"]:
                continue
            donor = donors.get(str(r.get("campaign_day")))
            for f in inh["fields"]:
                if donor is not None and donor.get(f) is not None and r.get(f) is None:
                    r[f] = donor[f]
                    r["flags"] = sorted(set(r["flags"]) | {f"inherited:{f}"})
                    prov.append({"record_uid": r["record_uid"], "field": f, "source_column": f"{inh['table_id']} (test day {r.get('campaign_day')})",
                                 "raw_value": str(donor[f]), "unit_raw": None, "unit": None, "parser": "inherit", "value_lo": None,
                                 "value_hi": None, "qualifier": "eq", "flags": [f"inherited:{f}"]})

    silver = pd.DataFrame(records)
    conformed = gold.frame(gold.conform(records, meas, prov, meta, fuels))
    assert conformed.record_uid.is_unique, "record_uid must be unique"
    tests = conformed[conformed.record_grain == "test"].reset_index(drop=True)
    camp = conformed[conformed.record_grain != "test"].reset_index(drop=True)
    meas_df = pd.DataFrame(meas, columns=list(MEASUREMENT_COLUMNS)).sort_values(
        ["record_uid", "variable", "source_column", "seq_index"], kind="stable").reset_index(drop=True)
    prov_df = pd.DataFrame(prov, columns=list(PROVENANCE_COLUMNS)).sort_values(
        ["record_uid", "field", "source_column"], kind="stable").reset_index(drop=True)
    rej_df = pd.DataFrame(rejects, columns=["table_id", "source_row", "field", "raw_value", "converted", "reason"])

    # ---- write layers
    b = out_dir / "bronze"
    _write(pd.DataFrame(bronze), b / "cells.parquet", csv=False)
    s = out_dir / "silver"
    silver_out = silver.drop(columns=["raw_json"]).copy()
    for c in silver_out.columns:
        if c != "flags" and silver_out[c].dtype == object:
            silver_out[c] = silver_out[c].map(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
    _write(silver_out, s / "records.parquet", csv=False)
    g = out_dir
    _write(_list_cols(tests), g / "tests.parquet")
    _write(_list_cols(camp), g / "campaigns.parquet")
    _write(_list_cols(meas_df), g / "measurements.parquet")
    _write(_list_cols(prov_df), g / "provenance.parquet", csv=False)
    rej_df.to_csv(g / "rejects.csv", index=False)

    # ---- marts (analytics / model-ready)
    marts = g / "marts"
    env = gold.envelope(tests)
    _write(env, marts / "condition_envelope.parquet")
    cfig_cols = KEYS + ["fuel", "fuel_mole_frac", "flame_type", "oxidizer_location", "o2_mole_frac", "pressure_kpa",
                        "ppo2_kpa", "ignited", "outcome", "burn_time_s", "flags"]
    cfig = gold.pivot(tests, meas_df, ["cfig_normal", "cfig_inverse"], CFIG_VARS, cfig_cols)
    cfig["label_ignited"] = cfig.ignited
    cfig["label_self_extinguished"] = cfig.outcome.map({"self_extinguished": True, "flow_terminated": False}).astype("boolean")
    cfig["group_fuel"], cfig["group_day"] = cfig.fuel, cfig.table_id + ":" + cfig.campaign_day.fillna("")
    _write(_list_cols(cfig), marts / "model_cfig.parquet")
    flex_cols = KEYS + ["test_date", "fuel", "o2_mole_frac", "n2_mole_frac", "co2_mole_frac", "he_mole_frac", "diluent",
                        "pressure_kpa", "ppo2_kpa", "outcome", "burn_time_s", "flags"]
    flex = gold.pivot(tests, meas_df, ["flex"], FLEX_VARS, flex_cols)
    flex["label_extinction"] = flex.outcome.map({"extinction": True, "burnout": False, "disruption": pd.NA}).astype("boolean")
    _write(_list_cols(flex), marts / "model_flex.parquet")
    b2_cols = KEYS + ["pi", "test_date", "fuel", "fuel_raw", "geometry", "thickness_mm", "char_diameter_mm",
                      "o2_mole_frac", "o2_mole_frac_final", "flow_velocity_cm_s", "flags"]
    b2 = gold.pivot(tests, meas_df, ["bass2"], BASS2_VARS, b2_cols)
    b2["o2_consumed"] = (b2.o2_mole_frac - b2.o2_mole_frac_final).round(6)
    b2["co2_produced"] = (b2.co2_mole_frac_final - b2.co2_mole_frac_initial).round(6)
    b2["usable_gas_balance"] = ~b2["flags"].map(lambda f: "suspect_o2_drop" in f) & b2.o2_consumed.notna()
    _write(_list_cols(b2), marts / "bass2_gas.parquet")
    fan = reference.bass2_fan_calibration(raw_dir)
    if fan is not None:
        _write(fan, g / "reference" / "bass2_fan_calibration.parquet")
    extra_quality = {}
    if (camp.table_id == "cfig_logs").any():
        jc = M.join_logs(tests, camp, meas_df, ["cfig_normal", "cfig_inverse"], "cfig_logs")
        extra_quality["log_rule_validation_cfig"] = M.validate_log_rule(jc)
        keep = KEYS + ["flame_type", "fuel", "o2_mole_frac", "pressure_kpa", "ignited", "outcome", "burn_time_s", "log_record_uid", "log_outcome", "log_ignition_to_end_s"]
        _write(jc[keep], marts / "log_validation_cfig.parquet")
    if (camp.table_id == "flame_design_logs").any():
        jf = M.join_logs(tests, camp, meas_df, ["flame_design_normal", "flame_design_inverse"], "flame_design_logs")
        jf["label_sustained"] = jf.log_outcome.map(M.SUSTAINED).astype("boolean")
        jf["label_source"] = jf.log_outcome.notna().map({True: "script log rule (outcome_from_log)", False: None})
        keep = KEYS + ["flame_type", "oxidizer_location", "fuel", "fuel_mole_frac", "o2_mole_frac", "pressure_kpa", "ppo2_kpa", "log_record_uid",
                       "log_outcome", "log_ignition_to_end_s", "label_sustained", "label_source", "flags"]
        fd = gold.pivot(tests, meas_df, ["flame_design_normal", "flame_design_inverse"], ["adiabatic_flame_temp", "stoich_mixture_fraction", "total_mass_flow"], ["record_uid"])
        _write(_list_cols(jf[keep].merge(fd, on="record_uid", how="left")), marts / "model_flame_design.parquet")
    video = M.load_video_summary(out_dir)
    ext = M.bass2_extinction_points(tests, meas_df, fan, video)
    if len(ext):
        _write(ext, marts / "bass2_extinction_points.parquet")
    vm = M.video_measurements(video, tests, variables, MEASUREMENT_COLUMNS)
    if len(vm):
        _write(_list_cols(vm), marts / "bass2_video_measurements.parquet")
    ins = gold.insights(tests, camp, meas_df)
    (marts / "insights.json").write_text(gold.dumps(ins))

    cards = investigations.build(raw_dir, meta, curated, tests, camp, meas_df, stats)
    (g / "investigations.json").write_text(gold.dumps(cards))

    # ---- docs
    quality = docs.quality(maps, stats, tests, camp, meas_df, rej_df, raw_dir)
    quality["skipped_maps"] = skipped
    quality.update(extra_quality)
    (g / "quality.json").write_text(gold.dumps(quality))
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "data-dictionary.md").write_text(docs.dictionary_md(maps, variables, tests, camp, meas_df))
    docs.dictionary_csv(maps, variables, tests).to_csv(g / "data_dictionary.csv", index=False)
    (docs_dir / "data-quality.md").write_text(docs.quality_md(quality, ins))

    manifest = {"pipeline_version": PIPELINE_VERSION, "outputs": {}}
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json":
            manifest["outputs"][str(p.relative_to(out_dir))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (g / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"tests": len(tests), "campaigns": len(camp), "measurements": len(meas_df), "rejects": len(rej_df),
            "t1": quality["t1"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", default=Path("data/processed"), type=Path)
    ap.add_argument("--docs", default=Path("docs"), type=Path)
    a = ap.parse_args()
    print(json.dumps(run(a.raw, a.out, a.docs), indent=2))


if __name__ == "__main__":
    main()
