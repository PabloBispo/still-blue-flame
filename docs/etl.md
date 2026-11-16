# ETL: PSI combustion tables to one citable dataset

One command, no network, no LLM, deterministic (two runs give byte-identical outputs).

```
python -m flame.etl.pipeline --raw ../flame-freefall/data/raw --out data/processed --docs docs
FLAME_RAW=../flame-freefall/data/raw python -m pytest tests/test_etl.py
```

Requires Python 3.10+, pandas, pyarrow, pyyaml, openpyxl.

## Layers

| Layer | What it is | Where |
|---|---|---|
| Bronze | Every non-empty source cell as text, with file SHA-256, sheet and row | `data/processed/bronze/cells.parquet` |
| Silver | Rows renamed to canonical fields, parsed, converted to SI. Driven only by the YAML maps | `data/processed/silver/records.parquet` |
| Gold | Conformed to vocabularies, joined to PSI metadata, derived columns, quality flags, split by grain | `tests`, `campaigns`, `measurements`, `provenance`, `investigations.json` |
| Marts | Analytic and model-ready views | `data/processed/marts/` |

## Design rules

1. **Curated maps, strict engine.** One YAML per source table in `flame/harmonize/maps/`. The engine refuses to run if a source column is neither mapped nor explicitly ignored with a reason, or if a map names a variable that is not in the vocabulary.
2. **Semantic consolidation, not only renaming.** Every source column is mapped to a concept: a core field of the contract or a variable of the controlled vocabulary (`vocab/variables.yaml`, grouped by category such as `ambient.composition`, `flow.velocity`, `outcome.extinction`, `emission.gas`). Fuels and materials go through `vocab/fuels.yaml`. Record grain and oxidizer location are explicit, so a burner-stream O2 fraction is never compared with a cabin atmosphere and a test-day summary is never counted as a test.
3. **Nothing is inferred, nothing is corrected.** Missing stays null. Suspect values keep their value and get a flag. Values outside the physical range of the contract become null in the core column and are listed in `rejects.csv` with the raw text.
4. **Full lineage.** Every record carries source file, sheet, row, file hash and the whole original row as JSON. Every core value has a provenance row with the raw cell, the raw unit and the parser.
5. **Documentation is generated.** `docs/data-dictionary.md` and `docs/data-quality.md` are rendered from the contract, the maps and the data.

## Adding a table (no code change)

1. Put the file under the raw directory.
2. Write `flame/harmonize/maps/<table_id>.yaml`: `table_id`, `psi_id`, `grain`, `source`, `constants`, `label`, and one entry per column (`field`, `variable` or `ignore`, plus `parser` and `unit`).
3. If the table brings a new quantity, add it to `vocab/variables.yaml`; a new fuel goes to `vocab/fuels.yaml`.
4. Run the pipeline. The engine tells you which columns are still unaccounted for.

Parsers: `number` (units inside the cell, `~`, `<`, `>`, ranges, custom tokens), `sequence`, `multi`, `enum`, `date`, `text`.
Source options: `sheet` / `sheets`, `header_row`, `header_aliases`, `join` (other sheets on a key), `keep_if`, `drop_if`, `ffill`, `sheet_field`, `label_append_row`.

## Marts

| File | Use |
|---|---|
| `marts/model_cfig.parquet` | CFI-G normal and inverse flames: features, `label_ignited`, `label_self_extinguished`, group keys `group_fuel` and `group_day` for grouped validation |
| `marts/model_flex.parquet` | FLEX droplets: ambient composition, diluent, extinction diameter, burning rate, `label_extinction` |
| `marts/bass2_gas.parquet` | BASS-II gas balance: O2 consumed, CO2 produced, CO, with `usable_gas_balance` excluding suspect rows |
| `marts/condition_envelope.parquet` | Range of O2, pressure, ppO2 and flow by investigation, table and fuel |
| `marts/insights.json` | Coverage against target atmospheres by fuel phase, label inventory, flag counts |
