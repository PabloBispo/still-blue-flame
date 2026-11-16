"""Tests for the ETL. Unit tests always run; pipeline tests need the raw PSI files.

    FLAME_RAW=/path/to/flame-freefall/data/raw python -m pytest tests/test_etl.py
"""
import hashlib
import math
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

from flame.etl import engine, gold, marts, parsers as P, pipeline, readers, units as U

ROOT = Path(__file__).resolve().parent.parent
RAW = Path(os.environ.get("FLAME_RAW", ROOT.parent / "flame-freefall" / "data" / "raw"))
needs_raw = pytest.mark.skipif(not (RAW / "inventory.csv").exists(), reason="raw PSI files not available")


# ---------------------------------------------------------------- units
@pytest.mark.parametrize("value,unit,canonical,expected", [
    (21, "%", "mole fraction", 0.21), (0.21, "mole_frac", "mole fraction", 0.21), (500, "ppm", "mole fraction", 5e-4),
    (1, "atm", "kPa", 101.325), (1, "bar", "kPa", 100.0), (8.2, "psia", "kPa", 56.537), (760, "mmHg", "kPa", 101.325),
    (760, "torr", "kPa", 101.325), (10, "mm/s", "cm/s", 1.0), (1, "m/s", "cm/s", 100.0), (100, "micron", "mm", 0.1),
    (0.37, "cm", "mm", 3.7), (0.25, "in", "mm", 6.35), (3700, "ms", "s", 3.7), (20, "minutes", "s", 1200),
    (27.1, "C", "K", 300.25), (2, "mg/min", "mg/s", 2 / 60),
])
def test_convert(value, unit, canonical, expected):
    assert math.isclose(U.convert(value, unit, canonical), expected, rel_tol=1e-4)


def test_unknown_unit_raises():
    with pytest.raises(ValueError):
        U.convert(1, "furlong", "mm")
    with pytest.raises(ValueError):
        U.convert(1, "sccm", "cm/s")  # volumetric flow is never turned into a velocity


# ---------------------------------------------------------------- parsers
def test_number_plain_and_units():
    assert P.number("0.21").value == 0.21
    p = P.number("20 cm/s")
    assert (p.value, p.unit) == (20.0, "cm/s")
    assert P.number(".33 cm").value == 0.33
    assert P.number("1 atm ").unit == "atm"


def test_number_qualifiers():
    assert P.number("~ 21.5").qualifier == "approx"
    p = P.number(">500")
    assert (p.value, p.qualifier) == (500.0, "gt")
    p = P.number("21.5 - 21.7")
    assert (p.lo, p.hi, p.qualifier) == (21.5, 21.7, "range") and math.isclose(p.value, 21.6)
    p = P.number("22.1 to 22.0")
    assert (p.lo, p.hi) == (22.0, 22.1)
    assert P.number("~ 10 cm (note 3)").value == 10.0


def test_number_never_guesses():
    for txt in ("none", "n/a", "-", "?", "n/a (Nomex)"):
        assert P.number(txt).value is None
    p = P.number("Insignificant")
    assert p.value is None and "non_numeric" in p.flags
    assert P.number("5 and 24 cm").value is None
    assert "suspect_leading_zero" in P.number("037 mm").flags
    assert P.number("-0.28").value == -0.28


def test_number_tokens():
    p = P.number("OR", {"OR": {"value": 1000, "qualifier": "gt", "flag": "censored_over_range"}})
    assert (p.value, p.qualifier, p.flags) == (1000, "gt", ["censored_over_range"])


@pytest.mark.parametrize("text,expected", [
    ("89,62,50,41", [89, 62, 50, 41]), ("66 > 52 > 45", [66, 52, 45]), ("0.9   1.3   2.75", [0.9, 1.3, 2.75]),
    ("5,2,1, <1", [5, 2, 1, 1]), ("5, 2.5 (1.15, .5, .35)", [5, 2.5]), ("5, down to ext", [5, None]),
])
def test_sequence(text, expected):
    assert [p.value for p in P.sequence(text)] == expected


def test_sequence_censored_and_range():
    seq = P.sequence("10, 3-4, <1")
    assert seq[1].qualifier == "range" and seq[1].value == 3.5
    assert seq[2].qualifier == "lt"


def test_multi_and_enum_and_date():
    assert [p.value for p in P.multi("0.5; 1.0; 1.5; 2.0; or 3.0")] == [0.5, 1.0, 1.5, 2.0, 3.0]
    assert P.enum("yes, PCI-D", {"yes": True, "no": False}).value is True
    assert P.enum("-", {"SE": "self_extinguished"}).value is None
    assert P.date("3/5/2009").value == "2009-03-05"
    assert P.date("6/29/`12").value == "2012-06-29"
    assert P.date("2012-FEB-13 SLICE testing").value == "2012-02-13"
    assert "suspect_date" in P.date("2103-05-15").flags


# ---------------------------------------------------------------- engine / vocabulary
def test_map_must_account_for_every_column():
    spec = {"table_id": "t", "psi_id": "PSI-0", "grain": "test", "columns": {"A": {"field": "fuel_raw"}}}
    with pytest.raises(engine.MapError):
        engine.validate_map(spec, ["A", "B"], {})
    spec["columns"]["B"] = {"variable": "not_in_vocab"}
    with pytest.raises(engine.MapError):
        engine.validate_map(spec, ["A", "B"], {})


def test_out_of_range_is_rejected_not_fixed():
    spec = {"table_id": "t", "psi_id": "PSI-0", "grain": "test", "label": ["id"],
            "columns": {"id": {"field": "test_label"}, "o2": {"field": "o2_mole_frac", "parser": "number", "unit": "%"}}}
    rows = [{"_sheet": None, "_row": 2, "id": "a", "o2": "3400"}]
    res = engine.harmonize(spec, rows, {"file": "f", "sha256": "x", "headers": ["id", "o2"], "rows_total": 1, "rows_blank": 0}, {})
    assert res["records"][0].get("o2_mole_frac") is None
    assert res["rejects"][0]["raw_value"] == "3400"
    assert "out_of_range:o2_mole_frac" in res["records"][0]["flags"]


def test_fuel_vocabulary():
    vocab = yaml.safe_load((ROOT / "flame/harmonize/vocab/fuels.yaml").read_text())
    f = gold.normalize_fuel("40% CH9", vocab)
    assert f["fuel"] == "methane" and "autofill_artifact" in f["flags"] and f["pct"] == 0.4
    f = gold.normalize_fuel('1/4" clear rod', vocab)
    assert (f["fuel"], f["geometry"], f["diameter_mm"]) == ("PMMA", "rod", 6.35) and "material_implied" in f["flags"]
    f = gold.normalize_fuel("1 cm wide 200 micron thick pmma", vocab)
    assert (f["thickness_mm"], f["geometry"]) == (0.2, "thin_film")
    assert "fuel_unmapped" in gold.normalize_fuel("unobtainium", vocab)["flags"]


def test_gravity_comes_only_from_metadata():
    assert gold.gravity_from_metadata({"flightProgram": "International Space Station (ISS)"})[:2] == (0.0, "microgravity")
    assert gold.gravity_from_metadata({"flightProgram": "Cygnus Spacecraft"})[0] == 0.0
    assert gold.gravity_from_metadata({"flightProgram": "Not Applicable"})[0] is None


def test_kv_dir_reader(tmp_path):
    d = tmp_path / "files" / "PSI-0" / "Raw Data"
    d.mkdir(parents=True)
    (d / "PSI-0_Raw Data_as run21001A1.txt").write_text("testpointName:21001A1,\nconstantFlowTimeSec:45,\nflowTimeoutSec:120")
    (d / "PSI-0_Raw Data_21001A1_TP.txt").write_text("\nTest Point: 21001A1\n\nIgnition and retraction: 2021-01-01 10:00:00\n"
                                                    "MFCs off: 2021-01-01 10:00:33\nExtinction detected: 2021-01-01 10:00:30\nReason: EXTINCTION DETECTED\n"
                                                    "Post-flame data: ***\n\nCamera    Archived\n=======\nIPSU1 \t 10\n")
    rows, info = readers.read_kv_dir(tmp_path, {"dir": "files/PSI-0/Raw Data", "prefix": "PSI-0_Raw Data_"})
    assert len(rows) == 1 and rows[0]["testpoint"] == "21001A1"
    assert rows[0]["asrun::constantFlowTimeSec"] == "45" and rows[0]["tp::Reason"] == "EXTINCTION DETECTED"
    assert rows[0]["tp::Post-flame data"] == "" and "tp::IPSU1" not in rows[0]
    spec = {"table_id": "t", "psi_id": "PSI-0", "grain": "test", "label": ["testpoint"], "unlisted": {"ignore": "config"},
            "derive": {"variable:ignition_to_end": {"seconds_between": [["tp::Ignition and retraction", "tp::Extinction detected"]]}},
            "derive_enum": {"outcome": [{"match": {"tp::Reason": "EXTINCTION"}, "value": "not_sustained"}]},
            "columns": {"testpoint": {"field": "test_label"}}}
    variables = {"ignition_to_end": {"category": "timing", "unit": "s"}}
    res = engine.harmonize(spec, rows, info, variables)
    assert res["records"][0]["outcome"] == "not_sustained" and "outcome_from_log" in res["records"][0]["flags"]
    assert res["measurements"][0]["value_num"] == 30.0


def test_fan_to_velocity_uses_closest_calibration():
    fan = pd.DataFrame({"calibration_date": ["2014-02-18"] * 3 + ["2014-07-08"] * 3, "calibration_label": ["a"] * 3 + ["b"] * 3,
                        "flow_restrictor": [2] * 6, "pot_setting": [0, 2, 10] * 2, "fan_display": [38, 55, 122] * 2,
                        "air_velocity_cm_s": [0, 9, 31, 0, 5, 20]})
    r = marts.fan_to_velocity(fan, 55, "2014-02-19", 2)
    assert r["velocity_from_fan_cm_s"] == 9.0 and r["calibration_gap_days"] == 1
    assert marts.fan_to_velocity(fan, 55, "2014-07-01", 2)["velocity_from_fan_cm_s"] == 5.0
    assert marts.fan_to_velocity(fan, 1.0, "2014-07-08", 2)["calibration_note"] == "reading is a potentiometer setting"
    assert marts.fan_to_velocity(fan, 55, "2014-07-08", 0)["velocity_from_fan_cm_s"] is None


# ---------------------------------------------------------------- pipeline (needs raw data)
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("run1")
    pipeline.run(RAW, out / "data", out / "docs")
    return out


@needs_raw
def test_idempotent(built, tmp_path):
    pipeline.run(RAW, tmp_path / "data", tmp_path / "docs")
    for rel in ("data/tests.parquet", "data/campaigns.parquet", "data/measurements.parquet", "data/provenance.parquet",
                "data/investigations.json", "data/marts/insights.json", "data/marts/bass2_extinction_points.parquet",
                "docs/data-dictionary.md", "docs/data-quality.md"):
        a, b = (hashlib.sha256((d / rel).read_bytes()).hexdigest() for d in (built, tmp_path))
        assert a == b, rel


@needs_raw
def test_contract(built):
    t = pd.read_parquet(built / "data/tests.parquet")
    c = pd.read_parquet(built / "data/campaigns.parquet")
    m = pd.read_parquet(built / "data/measurements.parquet")
    assert pd.concat([t.record_uid, c.record_uid]).is_unique
    assert (t.record_grain == "test").all() and (c.record_grain != "test").all()
    assert m.record_uid.isin(pd.concat([t.record_uid, c.record_uid])).all()
    for col in ("o2_mole_frac", "o2_mole_frac_final", "fuel_mole_frac", "n2_mole_frac", "co2_mole_frac", "he_mole_frac"):
        assert t[col].dropna().between(0, 1).all(), col
    assert t.pressure_kpa.dropna().between(1, 1000).all()
    assert t[["source_file", "source_row", "source_sha256", "raw_json", "doi"]].notna().all().all()
    assert t.gravity_source.str.startswith("PSI metadata").all()
    assert t[t.oxidizer_location != "ambient"].ppo2_kpa.isna().all()
    assert len(pd.read_json(built / "data/investigations.json")) == 24
    logs = c[c.record_grain == "test_log"]
    assert logs.outcome.dropna().isin(["not_sustained", "flow_terminated", "aborted_temperature_limit"]).all()
    assert (t[t.table_id == "acme_sflame_logs"].burn_time_s.isna()).all()


@needs_raw
def test_golden_rows(built):
    t = pd.read_parquet(built / "data/tests.parquet").set_index("record_uid")
    m = pd.read_parquet(built / "data/measurements.parquet")
    golden = yaml.safe_load((ROOT / "tests/golden/golden_rows.yaml").read_text())
    assert len(golden) >= 14
    for g in golden:
        row = t.loc[g["uid"]]
        for k, v in g["expect"].items():
            got = row[k]
            if v is None:
                assert pd.isna(got), (g["uid"], k, got)
            elif isinstance(v, bool) or isinstance(v, str):
                assert got == v, (g["uid"], k, got)
            else:
                assert math.isclose(float(got), v, rel_tol=1e-6, abs_tol=1e-6), (g["uid"], k, got)
        for var, v in g.get("measurements", {}).items():
            vals = m[(m.record_uid == g["uid"]) & (m.variable == var)].value_num.tolist()
            assert any(math.isclose(x, v, rel_tol=1e-6) for x in vals), (g["uid"], var, vals)
        for f in g.get("flags", []):
            assert f in list(row["flags"]), (g["uid"], f)
