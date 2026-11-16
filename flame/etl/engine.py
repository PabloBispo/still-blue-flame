"""Silver layer: apply a curated YAML map to a bronze table.

The engine is dataset-agnostic. Everything specific to a source table lives in its
YAML map; everything specific to the domain lives in contract.py and the vocab files.
Strict by design: a source column that the map does not account for is an error.
"""
from __future__ import annotations

import json
import re

import datetime as dt

from . import parsers as P
from . import units as U
from .contract import FIELD_BY_NAME


class MapError(ValueError):
    pass


def _targets(spec) -> list[dict]:
    return spec if isinstance(spec, list) else [spec]


def validate_map(spec: dict, headers: list[str], variables: dict) -> None:
    cols = spec["columns"]
    missing = [] if spec.get("unlisted") else [h for h in headers if h not in cols]
    extra = [c for c in cols if c not in headers]
    if missing:
        raise MapError(f"{spec['table_id']}: source columns not accounted for: {missing}")
    if extra:
        raise MapError(f"{spec['table_id']}: map names columns absent from the source: {extra}")
    for col, tg in cols.items():
        for t in _targets(tg):
            if "ignore" in t:
                continue
            if "field" not in t and "variable" not in t:
                raise MapError(f"{spec['table_id']}: column {col!r} has no field/variable/ignore")
            if "field" in t and t["field"] not in FIELD_BY_NAME:
                raise MapError(f"{spec['table_id']}: unknown field {t['field']!r}")
            if "variable" in t and t["variable"] not in variables:
                raise MapError(f"{spec['table_id']}: variable {t['variable']!r} not in vocab/variables.yaml")


def _convert(p: P.Parsed, col_unit: str | None, canonical: str | None):
    """Return (value, lo, hi, unit_raw) in the canonical unit."""
    if not isinstance(p.value, (int, float)) or isinstance(p.value, bool) or canonical is None:
        return p.value, p.lo, p.hi, col_unit
    unit_raw = p.unit if (p.unit and U.known(p.unit, canonical)) else col_unit
    if unit_raw is None:
        return p.value, p.lo, p.hi, None
    conv = lambda x: None if x is None else round(U.convert(x, unit_raw, canonical), 9)
    return conv(p.value), conv(p.lo), conv(p.hi), unit_raw


def _parse(t: dict, text: str) -> list[P.Parsed]:
    kind = t.get("parser", "text")
    if kind == "number":
        return [P.number(text, t.get("tokens"))]
    if kind == "sequence":
        return P.sequence(text)
    if kind == "multi":
        return P.multi(text, t.get("tokens"))
    if kind == "enum":
        return [P.enum(text, t["map"])]
    if kind == "date":
        return [P.date(text)]
    if kind == "text":
        return [P.text(text)]
    raise MapError(f"unknown parser {kind!r}")


def harmonize(spec: dict, rows: list[dict], info: dict, variables: dict) -> dict:
    validate_map(spec, info["headers"], variables)
    records, measurements, provenance, rejects = [], [], [], []
    dropped: dict[str, int] = {}
    last: dict[str, str] = {}
    seen: dict[str, int] = {}
    for row in rows:
        row_flags: list[str] = []
        original = {k: v for k, v in row.items() if not k.startswith("_")}
        for col in spec.get("ffill", []):
            if row.get(col):
                last[col] = row[col]
            elif last.get(col):
                row[col] = last[col]
                row_flags.append(f"forward_filled:{col}")
        reason = None
        for rule in spec.get("keep_if", []):
            if not re.search(rule["regex"], row.get(rule["column"], ""), re.I):
                reason = rule.get("reason", f"{rule['column']} does not match")
        for rule in spec.get("drop_if", []):
            if re.search(rule["regex"], row.get(rule["column"], ""), re.I):
                reason = rule.get("reason", f"{rule['column']} matches")
        if reason:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue

        rec = {"psi_id": spec["psi_id"], "table_id": spec["table_id"], "record_grain": spec["grain"]}
        rec.update(spec.get("constants", {}))
        if spec.get("sheet_field"):
            rec[spec["sheet_field"]] = row["_sheet"]
        notes: list[str] = []
        meas_rows, prov_rows = [], []
        for col, tg in spec["columns"].items():
            text = row.get(col, "")
            if not text.strip():
                continue
            for t in _targets(tg):
                if "ignore" in t:
                    continue
                parsed = _parse(t, text)
                if not parsed:
                    continue
                if "field" in t:
                    f = FIELD_BY_NAME[t["field"]]
                    p = parsed[0] if t.get("parser") != "sequence" else next(
                        (x for x in parsed if isinstance(x.value, float)), parsed[0])
                    value, lo, hi, unit_raw = _convert(p, t.get("unit"), f.unit)
                    flags = list(p.flags) + list(t.get("flags", []))
                    if t.get("parser") == "sequence" and len(parsed) > 1:
                        flags.append("sequence_first")
                    if f.dtype == "float" and isinstance(value, (int, float)) and not isinstance(value, bool):
                        if (f.lo is not None and value < f.lo) or (f.hi is not None and value > f.hi):
                            rejects.append({"table_id": spec["table_id"], "source_row": row["_row"], "field": f.name,
                                            "raw_value": text, "converted": value, "reason": f"outside [{f.lo}, {f.hi}] {f.unit}"})
                            flags.append(f"out_of_range:{f.name}")
                            value = None
                    if f.name == "notes":
                        if value:
                            notes.append(f"{col}: {value}" if t.get("prefix") else str(value))
                    elif value is not None and rec.get(f.name) is None:
                        rec[f.name] = value
                    prov_rows.append({"field": f.name, "source_column": col, "raw_value": text, "unit_raw": unit_raw,
                                      "unit": f.unit, "parser": t.get("parser", "text"), "value_lo": lo, "value_hi": hi,
                                      "qualifier": p.qualifier, "flags": flags})
                    row_flags.extend(x for x in flags if x != "missing_token")
                if "variable" in t:
                    v = variables[t["variable"]]
                    for i, p in enumerate(parsed):
                        value, lo, hi, unit_raw = _convert(p, t.get("unit"), v.get("unit"))
                        flags = list(p.flags) + list(t.get("flags", []))
                        num = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
                        if num is not None and v.get("unit") == "mole fraction" and not 0 <= num <= 1:
                            flags.append("out_of_range")
                            rejects.append({"table_id": spec["table_id"], "source_row": row["_row"], "field": t["variable"],
                                            "raw_value": text, "converted": num, "reason": "mole fraction outside [0, 1]"})
                            num = lo = hi = None
                        meas_rows.append({"variable": t["variable"], "category": v["category"], "seq_index": i,
                                          "value_num": num, "value_lo": lo, "value_hi": hi, "qualifier": p.qualifier,
                                          "unit": v.get("unit"), "value_text": p.text, "unit_raw": unit_raw,
                                          "source_column": col, "flags": flags})

        for fname, d in spec.get("derive_text", {}).items():
            m = re.search(d["regex"], row.get(d["column"], ""))
            if m:
                rec[fname] = m.group(1)
        for fname, d in spec.get("derive", {}).items():
            secs = pair = None
            for pair in d["seconds_between"]:
                a, b = (row.get(c, "") for c in pair)
                try:
                    secs = (dt.datetime.fromisoformat(b) - dt.datetime.fromisoformat(a)).total_seconds()
                    break
                except ValueError:
                    secs = None
            if secs is None or secs < 0:
                continue
            if fname.startswith("variable:"):
                var = fname.split(":", 1)[1]
                meas_rows.append({"variable": var, "category": variables[var]["category"], "seq_index": 0, "value_num": secs,
                                  "value_lo": None, "value_hi": None, "qualifier": "eq", "unit": "s", "value_text": f"{b} - {a}",
                                  "unit_raw": "s", "source_column": " - ".join(reversed(pair)), "flags": ["derived_from_timestamps"]})
                continue
            rec[fname] = secs
            row_flags.append("derived_from_timestamps")
            prov_rows.append({"field": fname, "source_column": " - ".join(reversed(pair)), "raw_value": f"{b} - {a}",
                              "unit_raw": "s", "unit": "s", "parser": "seconds_between", "value_lo": None, "value_hi": None,
                              "qualifier": "eq", "flags": ["derived_from_timestamps"]})
        for fname, rules in spec.get("derive_enum", {}).items():
            for rule in rules:
                if all(bool(row.get(c, "")) == want for c, want in rule.get("when", {}).items()) and \
                        all(re.search(rx, row.get(c, ""), re.I) for c, rx in rule.get("match", {}).items()):
                    if rec.get(fname) is None:
                        rec[fname] = rule["value"]
                        row_flags.append("outcome_from_log")
                        prov_rows.append({"field": fname, "source_column": "; ".join(list(rule.get("when", {})) + list(rule.get("match", {}))),
                                          "raw_value": "; ".join(f"{c}={row.get(c, '')!r}" for c in list(rule.get("when", {})) + list(rule.get("match", {}))),
                                          "unit_raw": None, "unit": None, "parser": "rule", "value_lo": None, "value_hi": None,
                                          "qualifier": "eq", "flags": ["outcome_from_log"]})
                    break
        label_cols = spec.get("label", [])
        label = "-".join(P.clean(row.get(c, "")) for c in label_cols if P.clean(row.get(c, "")))
        if not label:
            label = f"row{row['_row']}"
            row_flags.append("label_from_row")
        elif spec.get("label_append_row"):
            label = f"{label}-r{row['_row']}"
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-")
        uid = f"{spec['psi_id']}:{spec['table_id']}:{slug}"
        seen[uid] = seen.get(uid, 0) + 1
        if seen[uid] > 1:
            uid = f"{uid}~{seen[uid]}"
            row_flags.append("duplicate_label")
        rec.update({
            "record_uid": uid, "test_label": label, "notes": " | ".join(notes) or None,
            "flags": sorted(set(row_flags)), "source_file": info["file"], "source_sheet": row["_sheet"],
            "source_row": row["_row"], "source_sha256": info["sha256"],
            "raw_json": json.dumps(original, ensure_ascii=False, sort_keys=True),
        })
        records.append(rec)
        for m in meas_rows:
            measurements.append({"record_uid": uid, **m})
        for p in prov_rows:
            provenance.append({"record_uid": uid, **p})

    stats = {"table_id": spec["table_id"], "psi_id": spec["psi_id"], "grain": spec["grain"], "file": info["file"],
             "rows_total": info["rows_total"], "rows_blank": info["rows_blank"], "rows_dropped": dropped,
             "rows_kept": len(records), "n_source_columns": len(info["headers"]),
             "n_columns_ignored": sum(1 for tg in spec["columns"].values() if all("ignore" in t for t in _targets(tg)))
             + (len([h for h in info["headers"] if h not in spec["columns"]]) if spec.get("unlisted") else 0)}
    return {"records": records, "measurements": measurements, "provenance": provenance, "rejects": rejects, "stats": stats}
