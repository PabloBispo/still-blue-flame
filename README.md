# Still Blue: Flame in Freefall

NASA Space Apps Challenge 2026 (Uberlandia). Challenge: Flame in Freefall: SI-Powered Fire Safety Insights from Microgravity Combustion Data.

Code in this repository is written during the hackathon (Nov 14-15, 2026). Data comes from the NASA Physical Sciences Informatics repository (PSI, CC0) and is not committed.

## Layout
- `flame/analysis/extinction.py`: flame-present mask and burn interval from a luminosity series.
- `flame/analysis/stats.py`: Spearman with bootstrap CI, agreement metrics, Holm, exact McNemar.
- `tests/`: unit tests (`python -m pytest -q`).
