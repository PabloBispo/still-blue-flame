"""T2 (coverage of the target atmospheres) and T5 (BASS-II oxygen consumption vs initial oxygen).

    uv run python -m flame.analysis.hypotheses --data data/processed --docs docs

Reads gold tables and marts only. Writes data/processed/analysis/coverage.json, bass2_o2.json and docs/results.md.
T2 is a count and a distance, no model. T5 is a Spearman correlation with a percentile bootstrap interval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from flame.etl.gold import TARGET_ATMOSPHERES

# Half-widths of the "near the target" box, and the scales of the combined distance. Declared, not fitted.
BOX_O2_POINTS = 2.0
BOX_KPA = 10.0
SEED = 20261115
N_BOOT = 10000


def _cite(r) -> dict:
    return {"record_uid": r.record_uid, "investigation": r.investigation, "doi": r.doi, "fuel": r.fuel,
            "o2_pct": round(100 * r.o2_mole_frac, 2), "pressure_kpa": None if pd.isna(r.pressure_kpa) else round(float(r.pressure_kpa), 1)}


def coverage(tests: pd.DataFrame) -> dict:
    """Distance from each target atmosphere to the nearest test, per fuel phase. Ambient-oxidizer tests only."""
    amb = tests[(tests.oxidizer_location.fillna("ambient") == "ambient") & tests.o2_mole_frac.notna()]
    out = {"definition": {
        "population": "tests with oxidizer_location = ambient and chamber O2 reported; inverse flames (burner-stream O2) excluded",
        "box": f"within {BOX_O2_POINTS:g} O2 points and {BOX_KPA:g} kPa of the target",
        "scaled_distance": f"sqrt((dO2 points / {BOX_O2_POINTS:g})^2 + (dP kPa / {BOX_KPA:g})^2); 1.0 is the corner of the box. Scales are declared, not fitted",
        "note": "Tests without a reported pressure enter the O2-only gap but not the pressure gap, the box or the scaled distance."},
        "targets": []}
    for name, (x, p, src) in TARGET_ATMOSPHERES.items():
        entry = {"atmosphere": name, "o2_pct": round(100 * x, 1), "pressure_kpa": p, "ppo2_kpa": round(x * p, 1), "basis": src, "phases": {}}
        for phase in ("solid", "liquid", "gas"):
            a = amb[amb.fuel_phase == phase]
            b = a[a.pressure_kpa.notna()]
            d_o2 = (a.o2_mole_frac - x).abs() * 100
            ph = {"n_tests_with_o2": int(len(a)), "n_tests_with_o2_and_pressure": int(len(b)),
                  "o2_range_pct": [round(100 * a.o2_mole_frac.min(), 1), round(100 * a.o2_mole_frac.max(), 1)] if len(a) else None,
                  "pressure_range_kpa": [round(float(b.pressure_kpa.min()), 1), round(float(b.pressure_kpa.max()), 1)] if len(b) else None,
                  "nearest_o2_gap_points": round(float(d_o2.min()), 2) if len(a) else None,
                  "nearest_o2_test": _cite(a.loc[d_o2.idxmin()]) if len(a) else None}
            if len(b):
                do, dp = (b.o2_mole_frac - x) * 100, b.pressure_kpa - p
                dist = np.sqrt((do / BOX_O2_POINTS) ** 2 + (dp / BOX_KPA) ** 2)
                i = dist.idxmin()
                ph.update(nearest_pressure_gap_kpa=round(float(dp.abs().min()), 1),
                          n_tests_in_box=int(((do.abs() <= BOX_O2_POINTS) & (dp.abs() <= BOX_KPA)).sum()),
                          nearest_scaled_distance=round(float(dist.min()), 2),
                          nearest_test=dict(_cite(b.loc[i]), o2_gap_points=round(float(do.loc[i]), 2), pressure_gap_kpa=round(float(dp.loc[i]), 1)),
                          ppo2_gap_kpa_min=round(float((b.o2_mole_frac * b.pressure_kpa - x * p).abs().min()), 2))
            entry["phases"][phase] = ph
        out["targets"].append(entry)
    return out


def _spearman_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(x)
    if n < 8 or len(np.unique(x)) < 3 or len(np.unique(y)) < 3:
        return {"n": int(n), "rho": None, "ci95": None, "note": "too few tests or too few distinct values"}
    rho = float(spearmanr(x, y).statistic)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = np.array([spearmanr(x[i], y[i]).statistic for i in idx])
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": int(n), "rho": round(rho, 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def bass2_o2(gas: pd.DataFrame) -> dict:
    """T5: does O2 consumed grow with the initial O2? Spearman, percentile bootstrap, overall and within groups."""
    g = gas[gas.usable_gas_balance & gas.o2_mole_frac.notna() & gas.o2_consumed.notna()].copy()
    rng = np.random.default_rng(SEED)
    out = {"definition": {"x": "o2_mole_frac (initial chamber O2)", "y": "o2_consumed (initial minus final O2, mole fraction)",
                          "population": "BASS-II tests with a usable gas balance (marts/bass2_gas.parquet)",
                          "method": f"Spearman rho; 95% percentile bootstrap, {N_BOOT} resamples of tests, seed {SEED}",
                          "criterion": "rho > 0 with the interval excluding zero"},
           "overall": _spearman_ci(g.o2_mole_frac.to_numpy(float), g.o2_consumed.to_numpy(float), rng), "by_fuel": {}, "by_geometry": {}}
    for col, key in (("fuel", "by_fuel"), ("geometry", "by_geometry")):
        for name, d in g.groupby(col, dropna=True):
            r = _spearman_ci(d.o2_mole_frac.to_numpy(float), d.o2_consumed.to_numpy(float), rng)
            r["median_o2_consumed_points"] = round(100 * float(d.o2_consumed.median()), 2)
            r["o2_range_pct"] = [round(100 * float(d.o2_mole_frac.min()), 1), round(100 * float(d.o2_mole_frac.max()), 1)]
            out[key][str(name)] = r
    return out


def results_md(cov: dict, o2: dict) -> str:
    L = ["# Results", "",
         "Generated by `uv run python -m flame.analysis.hypotheses`. Numbers come from `data/processed/analysis/coverage.json` and `bass2_o2.json`.",
         "Only T2 and T5 were run. T3 (CFI-G model), T4 (FLEX diluents) and T6 (Saffire pairs) were not run; what exists for them is descriptive, in `docs/eda.md`.", "",
         "## T2. Solid fuels were not tested near the target atmospheres", "",
         f"Population: {cov['definition']['population']}. Box: {cov['definition']['box']}.", "",
         "| Target | Phase | Tests with O2 and pressure | In the box | Nearest O2 (points away) | Nearest pressure (kPa away) | Nearest test by combined distance (O2, pressure) |",
         "|---|---|---|---|---|---|---|"]
    for t in cov["targets"]:
        for ph, v in t["phases"].items():
            nt = v.get("nearest_test")
            L.append(f"| {t['atmosphere']} ({t['o2_pct']:g}% O2, {t['pressure_kpa']:g} kPa) | {ph} | {v['n_tests_with_o2_and_pressure']} | {v.get('n_tests_in_box', 0)} | "
                     f"{v['nearest_o2_gap_points']:g} | {v.get('nearest_pressure_gap_kpa', 'n/a')} | "
                     + (f"{nt['record_uid']} ({nt['o2_pct']:g}%, {nt['pressure_kpa']:g} kPa)" if nt else "n/a") + " |")
    s = {t["atmosphere"]: t["phases"]["solid"] for t in cov["targets"]}
    L += ["", "Reading:"]
    for name, v in s.items():
        L.append(f"- {name}: {v.get('n_tests_in_box', 0)} solid tests in the box; the closest solid test in O2 is {v['nearest_o2_gap_points']:g} points away"
                 + (f" and the closest in pressure is {v['nearest_pressure_gap_kpa']:g} kPa away." if 'nearest_pressure_gap_kpa' in v else "."))
    L += ["- Status: **confirmed** for the two exploration atmospheres. This is a count over what PSI published, not a statistical test.",
          "- Caveat: in oxygen partial pressure the solid tests do cover the targets; the gap is in mole fraction and in total pressure.", "",
          "## T5. BASS-II: oxygen consumed versus initial oxygen", "",
          f"Method: {o2['definition']['method']}. Criterion: {o2['definition']['criterion']}.", "",
          "| Group | n | Spearman rho | 95% interval | Excludes zero | Median O2 consumed (points) |", "|---|---|---|---|---|---|"]

    def row(name, r):
        ci = f"{r['ci95'][0]:g} to {r['ci95'][1]:g}" if r.get("ci95") else "n/a"
        return f"| {name} | {r['n']} | {r['rho'] if r.get('rho') is not None else 'n/a'} | {ci} | {r.get('ci_excludes_zero', 'n/a')} | {r.get('median_o2_consumed_points', '')} |"
    L.append(row("All tests", o2["overall"]))
    L += [row(f"Fuel: {k}", v) for k, v in o2["by_fuel"].items()]
    L += [row(f"Geometry: {k}", v) for k, v in o2["by_geometry"].items()]
    ov = o2["overall"]
    verdict = "confirmed" if ov.get("ci_excludes_zero") and ov["rho"] > 0 else "not confirmed"
    L += ["", f"- Status: **{verdict}** over all tests (rho {ov['rho']}, interval {ov['ci95'][0]:g} to {ov['ci95'][1]:g}, n = {ov['n']}).",
          "- Groups with fewer than 8 tests or fewer than 3 distinct values are not tested.",
          "- The groups are not independent comparisons and no multiplicity correction was applied; read the per-group rows as description.",
          "- Oxygen consumed depends on how long and how much of the sample burned, which the test matrix does not record; geometry is a proxy for that.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--docs", type=Path, default=Path("docs"))
    a = ap.parse_args()
    cov = coverage(pd.read_parquet(a.data / "tests.parquet"))
    o2 = bass2_o2(pd.read_parquet(a.data / "marts/bass2_gas.parquet"))
    out = a.data / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "coverage.json").write_text(json.dumps(cov, indent=1, ensure_ascii=False) + "\n")
    (out / "bass2_o2.json").write_text(json.dumps(o2, indent=1, ensure_ascii=False) + "\n")
    (a.docs / "results.md").write_text(results_md(cov, o2))
    print(json.dumps({"coverage_solid": {t["atmosphere"]: {k: t["phases"]["solid"].get(k) for k in
                                                           ("n_tests_in_box", "nearest_o2_gap_points", "nearest_pressure_gap_kpa")} for t in cov["targets"]},
                      "t5_overall": o2["overall"]}, indent=1))


if __name__ == "__main__":
    main()
