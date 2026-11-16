# BASS-II video pipeline (PSI-25)

Turns BASS-II videos into tables: the telemetry burned into every frame, the flame signal seen by the camera, the fan setting converted to air speed, and a per-test summary keyed by `record_uid` (same key as `tests.parquet`).

Status: 16 of 244 videos processed (289,149 frames, 2 h 41 min): O1, O3b, O4, O8a, O14, O19, P1, P9, P13a, P13b, P22a, P22b, P25, T2, B3, B21. The B14 file is truncated at the source and cannot be opened. Every number below is measured on those 16 files.

## Run

```
uv run python -m flame.video.pipeline --videos "../flame-freefall/data/raw/files/PSI-25/Raw Data" --out data/processed/video
uv run python -m flame.video.mart                 # data/processed/marts/bass2_video.parquet
uv run python -m flame.video.annotate --videos "../flame-freefall/data/raw/files/PSI-25/Raw Data" --stride 3
uv run python -m flame.video.plot                 # docs/img/video_overview.png (needs matplotlib)
uv run pytest tests/test_video.py                 # BASS_VIDEO_DIR=<Raw Data> enables the regression test
```

The pipeline skips videos already extracted (cache in `frames/`), saves each video as soon as it is read, and records a damaged file in `video_quality.json` without stopping the batch. Measured on 2 cores: 245 frames/s for extraction (8x real time); about 3.5 s of video per second for the annotated render with `--stride 3`.

## Modules

| Module | Role |
|---|---|
| `bass.py` | Frame reader: overlay position, digit templates, overlay values, flame region |
| `units.py` | FAN display to cm/s with the calibration closest in date; no extrapolation |
| `summarize.py` | Per second table, FAN plateaus, radiometer fall onset, camera check, per-test summary |
| `pipeline.py` | Batch CLI and cache |
| `mart.py` | Join with the BASS-II rows of `tests.parquet` and the gas balance |
| `annotate.py` | Video with the measurements drawn on it |
| `plot.py` | Overview figure |

## Outputs (`data/processed/video/`)

| File | Grain | Content |
|---|---|---|
| `frames/<label>.parquet` | frame | Overlay values, camera signal, flame geometry |
| `video_seconds.parquet` | test x second | Medians per second plus `flow_cm_s` |
| `video_summary.parquet` / `.json` | test | FAN plateaus, radiometer fall onset, air speed at that moment, camera reliability, notes |
| `video_quality.json` | file | Reader self-checks; damaged files carry `error` |
| `annotated/<label>_annotated.mp4` | video | Annotated video |
| `../marts/bass2_video.parquet` | test (129) | BASS-II tests with `has_video` |

Column notes:
- `fuel`, `air_cm_s`, `fan_display`, `rad`, `rad_1dp`: values as shown on the overlay. Null when a cell was not read with confidence; nothing is interpolated. Use `rad_1dp`: the second decimal of RAD is cut by the frame edge.
- `clock_tod_s`: overlay clock, seconds of day. The overlay date is the test date plus 1 day (leap-year bug stated in the PSI ReadMe): 16 of 16.
- `dim_px`, `lit_px`, `core_px`: pixels below the overlay with max(R,G,B) above 24, 60, 200. `lights_on`: chamber lights on, no flame geometry.
- `flame_area_px`, `flame_x0/y0/w/h`: flame body, pixels above 25% of the frame's peak level, the connected region holding the most light.
- `head_x`, `head_y`, `head_radius_px`: flame origin and head radius, from the largest circle inscribed in the body.
- `flame_cx`, `flame_cy`, `core_area_px`: hot core, pixels above 85% of the peak.
- `flow_cm_s`: FAN converted with the calibration closest in date for the test's flow restrictor. Null outside the calibrated display range (1 unit tolerance). Display 38 is a stopped fan.
- `camera_extinction_s` is filled only when `camera_reliable` is true.

## How the overlay is read

1. Fixed-pitch font (pitch 28.63 px, glyph 24x13 px). Its position shifts between recording sessions (0 to 5 px). `overlay_offset` locates it from the label row (`FUEL AIR FAN RAD`), which never changes.
2. Digit templates come from the video itself: the units-of-seconds digit of the clock advances at 1 Hz in a known order, so it labels itself. No manual annotation, no OCR model.
3. Every other cell is matched by normalised correlation with 2 px horizontal tolerance. Minus sign by shape.

## Checks

| Check | Result |
|---|---|
| Clock digits against the clock's own sequence | 98.0% to 100% per file, 16 files |
| Readings compared by eye | 52 frames in 5 files, 52 match (FUEL, AIR, FAN, RAD to 1 decimal, clock) |
| New files with unchanged parameters | 11 files, clock check 99.6% or more; no frame of these compared by eye |
| Overlay date minus test date | +1 day in 16 of 16 |
| FAN plateaus vs display values in the test matrix | 10 of 11 values found within 1.5 display units (6 tests) |
| Air speed at extinction, operator note vs video | 7 of 8 tests within 0.25 cm/s, median difference 0.13 cm/s (same calibration curve on both sides) |
| FAN converted vs anemometer | O1 at 20 cm/s: mean absolute error 0.03 cm/s over 41 s. Near zero the anemometer reads lower (0.5 to 1.6 cm/s) |
| O1 extinction vs ProofSheet stills | 200.9 s, inside the window given by the stills |

## Findings

1. The video holds data the test matrix does not: the full sequence of fan steps with the time of each one.
2. The video camera is not an extinction detector. It agrees with the radiometer in 4 of 16 tests; in the others the flame is dim and blue, or the chamber lights come back on. Use the radiometer read from the overlay.
3. Air speed at the radiometer fall for the 11 PMMA rods is 0.75 to 2.25 cm/s at 14.8% to 22.2% initial O2, with no trend. All of them go out one step above a stopped fan, and the calibration has a single point between stopped and its first step. Read it as "went out below about 2 cm/s". Do not fit a quenching boundary.
4. The radiometer is not an absolute scale: it saturates at 4.10 and also sees the hot sample.

## Limits

- 16 of 244 files. Thresholds and the radiometer-onset rule were tuned on 5 files and applied to 11 more without ground truth.
- Flame geometry has no ground truth and is in pixels. Do not use it as a model feature.
- Tests with flow restrictor 1 (O14, O19, B21) have no air speed for part or all of the test.
- P9, P22b, O19 and O14 have stretches without overlay (24%, 5%, 12%, 2% of frames).
- BASS (PSI-26) videos were not tested.
