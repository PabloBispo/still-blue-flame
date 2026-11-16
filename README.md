# Still Blue: Where the fire data stops

NASA Space Apps Challenge 2026, Uberlândia (Brazil). Challenge: **Flame in Freefall: SI-Powered Fire Safety Insights from Microgravity Combustion Data**.

NASA burned 2,299 samples in microgravity. None of the solid ones in the air a Moon base will breathe. We unified the combustion tables of NASA's Physical Sciences Informatics (PSI) repository into one citable dataset, measured that gap, and show what the existing evidence says.

- **Dashboard:** https://still-blue-flame.vercel.app
- **Slides (7):** https://still-blue-flame.vercel.app/slides/
- **Dataset:** https://github.com/PabloBispo/still-blue-flame/releases/tag/v1.0.0

## What is here

| Piece | What it is |
|---|---|
| Unified dataset | 2,299 tests from 13 investigations, 24,905 measurements, one schema, SI units, a source on every row (PSI ID, DOI, file, sheet, row) |
| Outcome labels from logs | 552 tests with no outcome in the published tables got one from the experiment script logs. Against the PI's own table in CFI-G the rule agrees in 269 of 284 tests (94.7%) |
| Video to data | The on-screen telemetry of 16 of 244 BASS-II videos (289,149 frames) read and joined to the test matrix |
| Dashboard | Coverage and gaps, what the evidence says, video to data, the 24 investigations side by side |

## Findings (descriptive, no statistical test)

- **The gap.** 0 of 261 solid-fuel tests with reported oxygen reach 30% O2; the highest is 22.7%. None was run at 70 kPa or less. The exploration atmosphere is 34% O2 at 8.2 psia (56.5 kPa). In oxygen partial pressure the solids do cover the targets (19.2 to 21.3 kPa); the gap is in oxygen fraction and total pressure.
- **Gas flames below 30% O2 rarely sustain.** ACME CFI-G (label by the PI): 0 of 64 sustained below 30%, 31 of 115 at 30% or more. ACME Flame Design (label from script log): 3 of 58 below 30%, 86 of 119 at 30% or more. These are spherical gas flames, not habitat materials.
- **Video and spreadsheet agree.** Air speed at extinction matches in 7 of 8 BASS-II tests (median absolute difference 0.13 cm/s). Both readings use the same calibration curve.

## What we do not claim

No fitted flammability or quenching boundary. Nothing about partial gravity (PSI holds no such test). No diluent effect in FLEX. The camera alone is not a reliable extinction detector (4 of 16 videos).

## Reproduce

Raw PSI files are not in this repository. `flame/etl/fetch_psi.py` downloads them from psi.nasa.gov.

```
uv sync
uv run python -m flame.etl.pipeline --raw <path-to-raw> --out data/processed --docs docs
FLAME_RAW=<path-to-raw> uv run pytest
```

## Layout

- `flame/etl/`: harmonization engine, gold tables, marts, docs generator, PSI fetcher.
- `flame/harmonize/`: 32 curated column maps (one per source table) and vocabularies.
- `flame/video/`: overlay reader, fan calibration, per-test summary, annotation.
- `flame/analysis/`: EDA, dashboard export, statistics helpers.
- `docs/`: data dictionary, data quality report, ETL, EDA and video method.
- `site/`: the dashboard and the slides (static).
- `tests/`: automated tests and a golden set.

## Data and credits

NASA Physical Sciences Informatics (PSI), https://psi.nasa.gov, CC0. Each investigation is cited by PSI ID and DOI in the dataset and on the dashboard.

## Use of SI

Code and documents were written with SI assistance (Claude). No number in this project was written by a language model; all are computed from PSI data by the code in this repository.

## Prior study

Before the hackathon (Nov 11 to 13) the team read the challenge, surveyed the PSI repository and its file inventory. That exploratory work is not in this repository. All code here was written during the event (Nov 14 and 15, 2026).

## Team

Still Blue: Pablo Bispo and Gabriel Bispo.
