"""Data contract for the gold layer.

Single source of truth for column names, types, units, valid ranges and
descriptions. The data dictionary is rendered from this module, so the
documentation cannot drift from the code.
"""
from __future__ import annotations

from dataclasses import dataclass

PIPELINE_VERSION = "1.0.0"

# origin: how the value gets into the column
#   source      read from a column of a PSI experimental table (see crosswalk)
#   constant    fixed for the whole source table, declared in its YAML map
#   metadata    taken from the PSI investigation metadata record, not from the table
#   vocabulary  normalised through a curated controlled vocabulary
#   derived     computed from other columns of the same record
#   system      produced by the pipeline (ids, lineage)


@dataclass(frozen=True)
class Field:
    name: str
    dtype: str  # string | float | bool | int | list[string]
    unit: str | None
    origin: str
    description: str
    lo: float | None = None
    hi: float | None = None
    enum: tuple[str, ...] | None = None


GRAINS = ("test", "test_log", "published_point", "test_day", "test_matrix", "session", "data_product")
OUTCOMES = (
    "self_extinguished", "flow_terminated", "extinction", "disruption",
    "burnout", "no_ignition", "not_sustained", "aborted_temperature_limit",
)

FIELDS: tuple[Field, ...] = (
    Field("record_uid", "string", None, "system",
          "Deterministic id: <psi_id>:<table_id>:<native test label>. Stable across runs."),
    Field("psi_id", "string", None, "constant", "PSI accession of the investigation (e.g. PSI-159)."),
    Field("investigation", "string", None, "metadata", "Investigation acronym from PSI metadata."),
    Field("doi", "string", None, "metadata", "Dataset DOI from PSI metadata. Cite this."),
    Field("table_id", "string", None, "constant", "Id of the source table map (flame/harmonize/maps/<table_id>.yaml)."),
    Field("record_grain", "string", None, "constant",
          "What one row represents. 'test' = one burn/run. Other grains are summaries "
          "(a test day, a planned matrix, a session, a data product), script logs of tests that already have a row "
          "(test_log) or points digitised from publications (published_point). They live in campaigns.parquet.",
          enum=GRAINS),
    Field("test_label", "string", None, "source", "Test identifier exactly as written by the investigators."),
    Field("test_date", "string", None, "source", "Test date, ISO 8601 (YYYY-MM-DD). Null when the table has no full date."),
    Field("campaign_day", "string", None, "source", "Operations day / GMT day as given. Use as a group key for validation splits."),
    Field("pi", "string", None, "source", "Principal investigator the test belongs to, when the table says so."),
    Field("fuel_raw", "string", None, "source", "Fuel / sample material text exactly as in the source."),
    Field("fuel", "string", None, "vocabulary", "Canonical fuel or material name (flame/harmonize/vocab/fuels.yaml)."),
    Field("fuel_phase", "string", None, "vocabulary", "Phase of the fuel as tested.", enum=("gas", "liquid", "solid", "aerosol")),
    Field("fuel_class", "string", None, "vocabulary", "Chemical / material family (alkane, alkene, alcohol, polymer, fabric, wax, ...)."),
    Field("fuel_mole_frac", "float", "mole fraction", "source",
          "Mole fraction of fuel in the fuel stream (rest is inert diluent). For inverse flames: fuel fraction of the ambient.",
          0.0, 1.0),
    Field("geometry", "string", None, "constant", "Burner or sample configuration (spherical_burner, droplet, coflow_jet, flat_sample, rod, ...)."),
    Field("thickness_mm", "float", "mm", "source", "Sample thickness for solid fuels.", 0.0, 100.0),
    Field("char_diameter_mm", "float", "mm", "source", "Characteristic diameter: droplet initial diameter, rod or sphere diameter, nozzle diameter.", 0.0, 1000.0),
    Field("flame_type", "string", None, "constant", "normal = fuel issues into oxidizer; inverse = oxidizer issues into fuel.", enum=("normal", "inverse")),
    Field("oxidizer_location", "string", None, "constant",
          "Where o2_mole_frac was set: 'ambient' (chamber atmosphere) or 'burner_stream' (inverse flames). "
          "Only 'ambient' rows describe a habitat-like atmosphere.", enum=("ambient", "burner_stream")),
    Field("o2_mole_frac", "float", "mole fraction", "source", "Oxygen mole fraction of the oxidizer at test start.", 0.0, 1.0),
    Field("o2_mole_frac_final", "float", "mole fraction", "source", "Oxygen mole fraction measured after the test. A measurement, not a setpoint.", 0.0, 1.0),
    Field("n2_mole_frac", "float", "mole fraction", "source", "Nitrogen mole fraction of the ambient.", 0.0, 1.0),
    Field("co2_mole_frac", "float", "mole fraction", "source", "Carbon dioxide mole fraction of the ambient at test start (diluent / suppressant).", 0.0, 1.0),
    Field("he_mole_frac", "float", "mole fraction", "source", "Helium mole fraction of the ambient.", 0.0, 1.0),
    Field("diluent", "string", None, "derived", "Inert species present above 1% in the ambient, joined by '+', e.g. 'N2', 'CO2+N2'. Null if composition was not reported."),
    Field("pressure_kpa", "float", "kPa", "source", "Ambient pressure at test start.", 1.0, 1000.0),
    Field("ppo2_kpa", "float", "kPa", "derived", "Oxygen partial pressure = o2_mole_frac * pressure_kpa. Only for oxidizer_location = ambient.", 0.0, 1000.0),
    Field("flow_velocity_cm_s", "float", "cm/s", "source",
          "Forced flow velocity. When the source gives a sequence (BASS, BASS-II) this is the FIRST value; "
          "the full sequence is in measurements (variable flow_velocity_seq).", 0.0, 1000.0),
    Field("flow_direction", "string", None, "source", "Flow relative to flame spread.", enum=("concurrent", "opposed", "coflow")),
    Field("gravity_g", "float", "g", "metadata",
          "Nominal gravity level. NOT read from the table: assigned from the PSI investigation metadata field "
          "flightProgram (ISS or Cygnus -> 0.0). It is a label for 'microgravity', not a measured acceleration.", 0.0, 1.0),
    Field("gravity_level", "string", None, "metadata", "microgravity | normal_gravity. Same provenance as gravity_g."),
    Field("gravity_source", "string", None, "system", "Literal text of the metadata value that gravity_g was assigned from."),
    Field("platform", "string", None, "metadata", "flightProgram from PSI metadata (e.g. International Space Station (ISS))."),
    Field("ignited", "bool", None, "source", "True if the source says the flame ignited. Null when the table does not say. Never inferred."),
    Field("outcome", "string", None, "source",
          "How the test ended, only where the source states it. 'not_sustained' comes from script logs and means the flame was gone "
          "before the planned end of flow: a failed ignition or a self-extinction, the log cannot tell which (flag outcome_from_log).",
          enum=OUTCOMES),
    Field("burn_time_s", "float", "s", "source", "Burn duration.", 0.0, 100000.0),
    Field("has_outcome_label", "bool", None, "derived", "True when ignited or outcome is not null. Rows usable as supervised labels."),
    Field("flags", "list[string]", None, "system", "Quality flags raised for this record (see flag glossary). Values are never silently corrected."),
    Field("notes", "string", None, "source", "Free-text comments / observations from the source."),
    Field("source_file", "string", None, "system", "Path of the source file relative to data/raw."),
    Field("source_sheet", "string", None, "system", "Worksheet name for xlsx sources."),
    Field("source_row", "int", None, "system", "1-based row number in the source file or sheet."),
    Field("source_sha256", "string", None, "system", "SHA-256 of the source file."),
    Field("raw_json", "string", None, "system", "The complete original row as JSON (header -> cell text). Nothing is lost."),
    Field("pipeline_version", "string", None, "system", "Version of this pipeline."),
)

FIELD_BY_NAME = {f.name: f for f in FIELDS}
CORE_COLUMNS = [f.name for f in FIELDS]

MEASUREMENT_COLUMNS = {
    "record_uid": "Record the value belongs to (tests.parquet or campaigns.parquet).",
    "variable": "Canonical variable name (flame/harmonize/vocab/variables.yaml).",
    "category": "Semantic category of the variable.",
    "seq_index": "Position inside a sequence or list of values (0 for scalars).",
    "value_num": "Numeric value in the canonical unit. Null if the source is not numeric.",
    "value_lo": "Lower bound when the source gives a range.",
    "value_hi": "Upper bound when the source gives a range.",
    "qualifier": "eq | approx | lt | gt | range.",
    "unit": "Canonical unit of value_num.",
    "value_text": "Original cell text (or sequence element).",
    "unit_raw": "Unit as declared for the source column.",
    "source_column": "Header of the source column.",
    "flags": "Quality flags for this value.",
}

PROVENANCE_COLUMNS = {
    "record_uid": "Record id.",
    "field": "Core column of tests.parquet / campaigns.parquet.",
    "source_column": "Header of the source column the value came from.",
    "raw_value": "Cell text before parsing.",
    "unit_raw": "Unit in the source.",
    "unit": "Canonical unit after conversion.",
    "parser": "Parser applied.",
    "value_lo": "Lower bound if the cell was a range.",
    "value_hi": "Upper bound if the cell was a range.",
    "qualifier": "eq | approx | lt | gt | range.",
    "flags": "Flags raised while parsing.",
}

FLAG_GLOSSARY = {
    "approx": "Source value carried '~' or 'approx'.",
    "range_midpoint": "Source gave a range; value is the midpoint, bounds kept in provenance.",
    "censored_gt / censored_lt": "Source gave a bound ('>500', '<1'); value is the bound.",
    "censored_over_range": "Sensor over range ('OR'); value is the sensor limit, qualifier gt.",
    "non_numeric": "Cell holds text where a number was expected; value is null, text preserved.",
    "missing_token": "Cell holds a placeholder ('none', 'n/a', '-', '?').",
    "sequence_first": "Cell holds a sequence; the core column keeps the first value.",
    "suspect_leading_zero": "Number written like '037'; probable missing decimal point. Not corrected.",
    "out_of_range:<field>": "Value outside the physical range of the contract; core column set to null, row listed in rejects.csv.",
    "suspect_o2_drop": "o2 final is more than 5 points below o2 initial; probable typo. Not corrected.",
    "o2_final_above_initial": "o2 final exceeds initial by more than 0.5 point.",
    "composition_sum_mismatch": "O2 + N2 + CO2 + He differs from 1 by more than 0.02.",
    "suspect_date": "Date outside 1995-2026 or not parseable; original kept in raw_json.",
    "forward_filled:<col>": "Value inherited from the row above (merged-cell layout in the source).",
    "fuel_unmapped": "Fuel text not found in the vocabulary.",
    "material_implied": "Material inferred from the sample description convention of the investigation (e.g. 'clear rod' = PMMA).",
    "autofill_artifact": "Source text shows a spreadsheet drag-fill artifact (CH4 -> CH5, CH6 ...). Normalised and flagged.",
    "fuel_frac_from_label": "fuel_mole_frac taken from the percentage written in the fuel label.",
    "unit_assumed": "Unit is not in the header; taken from investigation documentation and declared in the map.",
    "derived_from_timestamps": "Value is the difference between two timestamps of the test script log.",
    "from_video": "Value read from the video overlay or image by flame.video, not from a PSI table (marts/bass2_video_measurements.parquet).",
    "radiometer_saturated": "The radiometer display reached its ceiling (4.10) during the test; the fall onset is later than the true one.",
    "censored": "The event was not observed before the video ended; the value is a bound.",
    "outcome_from_log": "Outcome assigned by a declared rule over the test script log (see the map), not written by the investigators in a table.",
    "inherited:<field>": "Value copied from the test-day summary table of the same investigation, joined on the test day.",
    "label_from_row": "No usable test id in the source; label built from the row number.",
}
