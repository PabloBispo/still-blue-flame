# Exploratory data analysis and data-utilisation audit

Run: `uv run python -m flame.analysis.eda --data data/processed --raw ../flame-freefall/data/raw`.
Every number below is in `data/processed/eda/eda.json`. Descriptive only: counts, ranges and rates with n. No hypothesis test was run here; associations are confounded by the test designs and are stated as patterns, not effects.

> Sections 1 to 9 describe the dataset before the ACME script logs were added (2,010 tests). Section 10 describes what the logs, PSI-117 and the fan calibration added on top (2,299 tests). `eda.json` always reflects the current build.

## 1. Shape of the data

| Item | Value |
|---|---|
| Tests | 2,010 (gas 1,129; liquid 501; solid 380) |
| Core columns | 46 |
| Measurements | 16,432 values of 79 variables; 67.3% numeric, the rest text |
| Campaign records (not tests) | 130 |
| Tests with a full date | 750 (37%), from 2009 to 2018 |
| Largest fuels | propylene 368, ethylene 354, PMMA 178, propane 169, methanol 157, n-butane 140 |
| Geometries | spherical burner 660, droplet 501, co-flow jet 469, heated sample 117, fabric 55, rod 54, sphere 52, slab 36, thin film 21, candle 18 |
| Conditions with replicates | 62 of 577 condition groups (CFI-G, FLEX, CFI) |

The dataset is three different experiments wearing one schema: gaseous flames on burners (56%), liquid droplets (25%) and solids (19%). Only the solids are habitat materials. Any statistic taken over all tests mixes them; always split by `fuel_phase` and `oxidizer_location`.

## 2. Are we throwing data away?

### Inside the files we load: no

| Cells | n | Share |
|---|---|---|
| Non-empty source cells | 31,713 | 100% |
| Loaded into gold | 28,758 | 90.7% |
| Ignored: redundant with a loaded column (duplicate unit, constant, part of the id) | 1,727 | 5.4% |
| Ignored: camera settings (BASS as-run sheet) | 1,184 | 3.7% |
| In 17 dropped rows (setup, stow, flow-visualisation rows) | 44 | 0.1% |

95 numeric cells did not become numbers, all for a stated reason: 69 say `RAMP` / `RAMPED` (no single value exists), 8 negative air velocities in SPICE, 6 unreadable characters in FLEX, 4 `Insignificant` burn times, the rest text notes. Each is preserved as text and flagged.

Local spreadsheets not loaded were opened and checked: the Saffire `MergeResult` workbooks (12 MB and 8 MB) are image-file indexes with no science values; the BASS planned matrix is superseded by the as-run matrix.

### Information we load but do not yet use as structure

1. **BASS test observations.** All 121 BASS tests carry a free-text observation (median 185 characters). 63 mention an extinction word, 13 a burn-out, 6 a failed ignition, 26 a burn time, 25 nitrogen. This is the outcome label for solids that the tables lack. It is in `notes`, not in `outcome`. Turning it into labels needs a human or a typed classifier with review; not done.
2. **Flow sequences.** 130 of the 250 BASS / BASS-II tests have a flow sequence of two or more steps and 54 go below 1 cm/s. In 18 tests (16 in BASS-II) the cell itself says the flame went out at the last step ("5 down to ext"). Those 18 are quenching-limit points (O2, flow at extinction) available without any video work.
3. **CFI comments.** 227 of 227 tests have comments (median 300 characters); 177 mention extinction, 142 a duration.

### Outside the files we load: yes, a lot

The inventory lists 16,368 files; 201 were downloaded. 7,214 tabular or text files (4.9 GB) were not:

| Investigation | Not downloaded | What it is |
|---|---|---|
| ACME CFI-G (PSI-159) | 1,701 files, 1.5 GB | Per test: time-series `_data.csv` (337 tests), as-run settings, test-point and comment files |
| ACME Flame Design (PSI-10) | 1,695 files, 1.3 GB | Same structure, 317 tests |
| ACME s-Flame (PSI-23) | 1,711 files, 0.9 GB | Same structure, 286 tests. **Our table has only 18 day summaries for this investigation** |
| CFI (PSI-39) | 1,793 files, 0.2 GB | Per test: chamber readings, radiometer readings, test-point settings (124 tests) |
| SAME-R, SAME (PSI-101, 102) | 144 analysed workbooks, 0.7 GB | Particle size distributions per run |
| PSI-115 | 44 `.dat`, 0.25 GB | Smoke simulation outputs |
| SPICE (PSI-107) | 35 files, 25 MB | Raw workbooks |
| **PSI-117** | **10 files, 0.3 MB** | **Analysed CSVs: alkane extinction diameters, C12 extinction diameter versus pressure. This investigation has no experimental table, so today it has zero rows** |
| **BASS-II (PSI-25)** | **1 file, 24 KB** | **Fan calibration: converts fan display to cm/s** |
| PSI-142 | 1 workbook, 0.7 MB | Data summary with links |

The small per-test text files of the three ACME investigations (as-run, test-point, comments: about 1 to 2 KB each, roughly 4 MB in total) would give test-level rows for s-Flame and independent checks for CFI-G and Flame Design.

## 3. Atmospheres: what was tested against what a habitat will use

Ambient-oxidizer tests with O2 reported: 1,148; with O2 and pressure: 1,007.

| | Gas | Liquid | Solid |
|---|---|---|---|
| Tests with O2 >= 30% | 246 | 24 | **0** |
| Tests with pressure <= 70.3 kPa | 25 | 20 | **0** |
| Both | 20 | 0 | **0** |
| O2 range | 18.0% to 40.0% | 12% to 34% | 13.6% to 22.7% |
| ppO2, median (kPa) | 37.9 | 20.3 | 21.4 |

All three target atmospheres have nearly the same oxygen partial pressure: 21.3 kPa (ISS), 21.1 kPa (30% at 10.2 psia) and 19.2 kPa (34% at 8.2 psia). Solid tests match that partial pressure (102 of the 121 with pressure are within 2 kPa of the 19.2 to 21.3 range) but only at about 1 atm and 21% O2. No solid was burned at reduced pressure or enriched oxygen. The gap is two-dimensional, in mole fraction and in total pressure, not in partial pressure.

## 4. CFI-G: where there are labels

**Normal flames (179 tests, chamber O2 18% to 40%).** 132 ignited. 102 tests are marked self-extinguished and 31 flow-terminated (one flow-terminated row has ignition "No" in the source, so 30 of the 31 are among the ignited).
- Below 30% O2: 41 of 41 ignited flames self-extinguished. No flame lasted to flow termination.
- At 30% O2 or above: 61 self-extinguished and 31 lasted to flow termination (34% of the 92). All 31 have O2 between 33.2% and 39.9%.
- The lasting flames appear at every pressure tested at high O2: 5 at 51 kPa, 9 at 94 kPa, 7 at 101 kPa, 10 at 203 kPa.
- Median burn time: 29.5 s when self-extinguished, 61.5 s when flow-terminated.

**Inverse flames (154 tests, burner-stream O2 12% to 85%).** Self-extinction among ignited flames goes with pressure: 0 of 29 at 50 to 55 kPa, 43% of 35 at 110 kPa, 89% of 35 at 221 kPa, 84% of 32 at 331 kPa. Pressure and O2 level were not varied independently (the low-pressure runs are mostly at 85% O2), so this is a pattern of the design, not an isolated pressure effect.

Caveat for both: spherical gaseous flames. They speak about flame sustainment, not about material flammability.

## 5. FLEX: diluents

274 droplet tests, methanol 157 and n-heptane 117, O2 from 12% to 34%, pressure 70 to 309 kPa (median 101). Outcomes: methanol 126 extinction, 23 disruption, 8 burnout; heptane 60, 38, 19.

Extinction diameter over initial diameter (larger = flame goes out earlier), median by dominant diluent:

| Fuel | N2 only | N2 + CO2 | CO2-dominant | N2 + He |
|---|---|---|---|---|
| Methanol | 0.54 (n = 38) | 0.51 (48) | 0.83 (9) | 0.57 (20) |
| n-Heptane | 0.80 (24) | 0.79 (37) | 0.99 (3) | 0.66 (14) |

CO2-dominant atmospheres show earlier extinction in both fuels, with small n and different O2 ranges per group. Helium atmospheres show higher burning-rate constants (methanol 0.58 to 0.65 against 0.44 mm2/s in N2). The analysis block must control for O2 before claiming a diluent effect.

## 6. CFI: cool flames

227 tests; the PI completed the analysis of 146. 144 have a hot-flame extinction diameter and 120 a cool-flame extinction diameter. None of the 44 farnesane tests has analysed results. For n-dodecane the hot-flame burning-rate constant rises with pressure, from 0.32 mm2/s at 51 kPa to 0.76 at 507 kPa (medians, n from 3 to 27 per level).

## 7. Solids

- BASS and BASS-II together: 250 tests, PMMA in five geometries (rod 54, sphere 52, slab 28, thin film 21, not stated 21), SIBAL 39, wax 24, Nomex 9, Ultem 2. O2 from 13.6% to 22.7%, first flow from 0 to 49 cm/s.
- **BASS-II gas balance** (124 usable tests): O2 consumed and CO2 produced agree (Spearman 0.94), so the two gas analysers are mutually consistent. O2 consumed is almost unrelated to initial O2 (Spearman 0.11) and strongly related to geometry: slabs 1.1 points and 389 ppm CO (medians), rods 0.5 points and 182 ppm, thin films 0.1 points and 23 ppm, fabrics 0.0. 18 tests ended with CO at 500 ppm or above.
- **Saffire-II** (9 samples at about 22% O2, 20 cm/s): silicone did not spread in microgravity in 4 of 4 samples while 3 of the 4 burned in the 1 g reference; SIBAL burned the full 29 cm in both; thick PMMA burned about 10 cm in microgravity against "Complete" in 1 g. Text values, n = 9, descriptive.

## 8. Detection and campaign tables

- SAME-R: 109 smoke runs on 5 materials at 0, 1, 2 and 8 cm/s; heater ratio Rt/R0 is set per material (Kapton 1.43, Teflon 1.45, silicone 1.35, lamp wick 1.26, Pyrell 1.23, medians).
- SPICE: 469 conditions, the largest single table, but no O2 and no pressure (cabin air).
- Campaign tables record 1,044 flames over 89 test days (ACME CLD 344, s-Flame 276, E-FIELD 230, BRE 194) with only day-level detail. ACME BRE ran O2 setpoints of 21%, 25%, 26.5%, 34% and 40%.

## 9. Quality

799 tests carry at least one flag; most are informational (unit taken from documentation, burn time derived from clock times, size read from the label). 28 tests carry a flag that questions a value (suspect, out of range, composition not summing to 1, O2 final above initial).

## 10. Added after the first pass: script logs, PSI-117, fan calibration

Downloaded from PSI with `flame/etl/fetch_psi.py`: 2,978 per-test text files of the three ACME investigations (about 3 MB), the 10 PSI-117 analysed files, the BASS-II fan calibration and the PSI-142 summary workbook (opened, not loaded: ground counterflow burner).

**Dataset now:** 2,299 tests (gas 1,418, liquid 501, solid 380), 843 campaign records, 24,905 measurements.

| Addition | Rows | Where |
|---|---|---|
| ACME s-Flame test points, rebuilt from script files | 289 tests over 18 days (was 18 day summaries) | `tests.parquet` |
| CFI-G script logs | 337 (327 match a PI-table test) | `campaigns.parquet`, grain `test_log` |
| Flame Design script logs | 322 (308 match a PI-table test) | `campaigns.parquet`, grain `test_log` |
| PSI-117 published points | 54 | `campaigns.parquet`, grain `published_point` |
| BASS-II fan calibration | 555 points, 25 dates | `reference/bass2_fan_calibration.parquet` |

### The outcome rule and its check

The script files record the ignition command, an "extinction detected" timestamp and the flow-off time. Rule: extinction detected -> `not_sustained`; flow off with no extinction -> `flow_terminated`; burner temperature limit -> `aborted_temperature_limit`. Checked against the CFI-G PI tables, which state the outcome:

| PI table outcome | log: not sustained | log: flow terminated | log: temperature abort |
|---|---|---|---|
| self-extinguished | 163 | 2 | 0 |
| no ignition | 52 | 0 | 0 |
| flow terminated | 13 | 43 | 11 |

- Sustained versus not sustained agrees in 94.7% (269 of 284 tests with an outcome on both sides).
- The script cannot tell a failed ignition from a self-extinction: all 52 PI "no ignition" tests appear as extinction detected. Hence the label `not_sustained`, not `self_extinguished`.
- 13 tests the PI calls flow-terminated show extinction 3 s before flow-off in the log. Unexplained; counted as disagreement.
- The log window (ignition command to end) is a median 14 s longer than the PI burn time, so it is stored as the measurement `ignition_to_end`, never as `burn_time_s`.

### What the logs show (descriptive)

- **Flame Design, normal flames** (chamber O2, ethylene and propane; outcome from the log rule): below 30% O2, 55 of 58 were not sustained; at 30% or more, 86 of 119 lasted (70 to flow termination, 16 to the temperature limit). Same direction as CFI-G normal flames, in different fuels and a different investigation. This gives 289 Flame Design tests an outcome label they did not have (`marts/model_flame_design.parquet`).
- **Flame Design, inverse flames:** below 30% burner O2, 15 of 15 lasted; at 30% or more, 56 of 87.
- **s-Flame** (263 with an outcome): 219 not sustained, 29 temperature aborts, 15 flow-terminated. No chamber O2 or pressure in the files, so these tests cannot enter the atmosphere analysis.

### Solids: extinction points

`marts/bass2_extinction_points.parquet`: 26 rows for 20 BASS-II tests, one row per evidence source (16 from the test-matrix cell that says the flame went out, 10 from the 5 videos processed by the video session: radiometer decay and camera). 17 tests end up with a velocity at extinction, read directly or converted from the fan reading with the calibration closest in date. The calibration matters: fan display 122 corresponds to 29 to 59 cm/s depending on the date, so conversion without the date-matched curve can be wrong by a factor of two. In 4 of the 16 matrix cells the operator wrote only "down to ext" with no value.

### Utilisation, updated

With the logs loaded the bronze layer has 111,205 non-empty cells. 69,945 of them are script configuration of the ACME runs (camera exposure, fibre, igniter and timing parameters), kept in `raw_json` but not mapped. Of the remaining 41,260 cells, 38,302 (92.8%) are loaded. 4,224 tabular files (4.9 GB) remain on PSI, almost all of it the per-test time series of the three ACME investigations and the SAME workbooks.
