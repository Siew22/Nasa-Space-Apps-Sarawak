# Scientific integrity audit — Flame in Freefall (2026-10-24)

Scope: the data, preprocessing, evaluation, API and dashboard of this repository at commit `572ddeb`,
audited and refactored on 2026-10-24. Each finding gives the file, the implementation and the evidence.

## 1. Findings before changes

| # | Finding | Evidence |
|---|---|---|
| F1 | **The 80 "NASA" BASS-II/FLEX records were generated, not measured.** | `src/download_nasa_tables.py` (now `src/generate_synthetic_demo_tables.py`) wrote them with formulas, e.g. `d0 = 1.50 + (i*0.04) % 1.2`, `pressure = 750 + (i % 5)*5.2`, and labelled them "NASA PSI-69 (FLEX) Experimental Table full series". `src/provenance_audit.py` regenerated the generator output and matched **40/40** rows for each dataset. Identifiers `CAL_01…CAL_40` and `BASS_SAMPLE_101…140` exist only in that generator. |
| F2 | **Ingestion attached the real DOI and CC0 licence to generated rows.** | `src/build_dashboard_data.py` copied `doi`/`license_status` from `config/dataset_catalog.csv` by filename regex. |
| F3 | **The 5×5 CV dataset leaked its BASS-II label from an input.** | `src/legacy/research_dataset_curation.py:97`: `target_rate_mm_s = fan_display × 0.0016 × 1.5`. The same fan setting drives the Reynolds-number feature. The audit recomputed the formula: **40/40** BASS-II targets match. Its extinction label is a rule (`final CO ≤ 2 ppm`, line 95), not an observation. |
| F4 | **Physically different targets were merged.** | The same file put a BASS-II proxy "spread" value and the FLEX burning rate into one column `target_rate_mm_s`. |
| F5 | **The CV dataset's metadata was an unfilled template.** | `research_curated_dataset.metadata.json` is byte-identical to `.template.json` and has 34 `REPLACE_WITH_*` placeholders. It still declares `"data_status": "curated_empirical_measurements"` and `"contains_synthetic_rows": false`. |
| F6 | **"Repeated" GroupKFold could not be repeated.** | scikit-learn 1.5.2 `GroupKFold(n_splits)` has no `shuffle`/`random_state` (checked with `inspect.signature`), so seed repeats give identical folds (`src/legacy/evaluate_baseline.py`). |
| F7 | **`spacecraft_fire_benchmark.csv` holds fallback constants.** | Every value column has one distinct value (flux 50, duration 450, mean 320, peak 650). The cause is in `src/legacy/ingest_macfp_data.py`: a bare `except: continue` hid a units-row parsing error, and lines 91–94 then wrote constants as if they were measurements. |
| F8 | **The referenced `spacecraft_fire_multitask_v1.keras` does not exist.** | `data/processed/logs of fire model.txt` logs a run of code that was never committed. The deployed model is `models/spacecraft_fire_peak_hrr_v1.keras`, trained on `macfp_measured_cone_experiments.csv` (per its metadata). |
| F9 | **The UI and API overstated the evidence.** | Examples: "100% 真实数据驱动…绝不硬编码" (`app/main.py`, `research_engine.py`); "NASA PSI-25 (BASS-II Flight Records)" for generated rows; README "Zero Hardcoding", "absolute zero data leakage"; dashboard tiles and "100% PASSED" typed in by hand; prediction preview with invented peaks (650, 2400). |
| F10 | **The MaCFP data and model are genuine and reproducible.** | Rebuilding from MaCFP commit `b72eb23` gives a byte-identical CSV (SHA-256 `de10787f…`), equal to the model metadata's `csv_sha256`. Rerunning the stored grouped evaluation reproduces the metrics exactly, and retraining with seed 42 reproduces the checkpoint's predictions with 0.0 kW/m² difference. |

## 2. Data provenance after changes

- **Official NASA tables obtained.** The PSI portal's public API (the endpoints its web app calls; no login) serves the
  experimental tables. Each was downloaded **three times independently**: SHA-256 identical, size equal to NASA's metadata
  `file_size`, served from NASA's `psi-repo-prod` S3 bucket with the recorded `s3_version_id`. They are stored unchanged in
  `data/raw/nasa_psi/` and registered in `config/data_sources.json`. `.gitattributes` prevents end-of-line rewriting.
  - PSI-25 BASS-II: 129 tests, 6 PIs, 30 test dates.
  - PSI-69 FLEX: 274 tests, 80 test dates.
  - Licence `CC0-1.0` in both metadata records.
- **Computed statuses.** `src/provenance_audit.py` writes `reports/data_provenance_audit.json` with evidence codes, and
  `reports/provenance_mapping.csv` with one row per record: internal ID, investigation, original identifier, source file and
  row, persistent identifier, fields and units, transformations, status and explanation.
  - Result: BASS-II, FLEX and MaCFP are all `VERIFIED_SOURCE`.
  - The retired files are classified with evidence (synthetic, unfilled template, unverifiable log).
- **Synthetic demo kept, never mixed.** The generator now writes `data/demo/SYNTHETIC_DEMO_*.csv` with an honest note, and is
  ingested only with `--include-demo`. Demo files never receive a DOI or licence.

## 3. Targets, labels and features (`config/target_registry.json`)

| Task | Decision | Evidence |
|---|---|---|
| FLEX burning-rate constant | unit **mm²/s**; the PSI header "Burning rate; mm" is truncated | NASA/TP-2015-216046 App. A.1 ("average burning rate constant from a linear fit of the droplet diameter squared versus time"), Table IX header "Burning rate, mm2/s" |
| FLEX composition columns | "O", "N", "CO", "He" = O₂, N₂, **CO₂**, He | Table IX header "O2 N2 CO2 He" |
| FLEX extinction label | Extinction = 1, Completion = 0, **Disruption excluded** (not pooled) | App. A.1: "-" extinction diameter = no extinction "because the droplet burned to completion or because a droplet disruption ended the test" |
| FLEX inputs | fuel, pressure, O₂, CO₂, He, initial diameter (all known before the test) | N₂ omitted (= 1 − others); burn time, extinction diameter and outcome excluded as results |
| FLEX exclusions | duplicate ID `193F001` (conflicting values) both excluded; `C11M201` pressure 0 mmHg excluded; rows with missing inputs excluded | `data/processed/tasks/flex_exclusions.csv` lists every row and reason |
| BASS-II spread speed | **not evaluated**: no spread column in the PSI-25 table | Spread rates exist only in report tables, e.g. NASA/TM-20210011385 Table 5.1 |
| BASS-II extinction | **not evaluated**: no observed outcome recorded | — |
| MaCFP peak HRR | kW/m²; two inputs come from the same curve, so after-test only | Unit rows checked by `src/build_measured_cone_dataset.py` |
| BASS-II cells | `OR`/`over` = over range (> 1000 ppm, PSI-25 ReadMe), `>500` censored, `none`/"–" missing, multi-value cells flagged | `app/services/psi_parsing.py`; counts disclosed in rankings |

## 4. Evaluation protocol (`src/group_cv.py`, `src/run_cv_evaluation.py`)

**Splits**
- **Seeds:** 5 documented seeds `[42, 101, 2024, 3060, 9999]` × 5 outer folds, made of whole groups.
- **Group units:** a laboratory for MaCFP, a test date for FLEX.
- **Order:** the outer test fold is fixed first, then inner validation groups are drawn from the outer-training groups.

**Model selection and scoring**
- **Tuning:** only on inner validation (MAE for regression, log loss for classification).
- **Final fit:** refit on outer-train and scored once.
- **Fixed settings:** the classification threshold is fixed at 0.5 and never tuned. The deployed network uses its predefined recipe, untuned.

**Preprocessing:** imputation and scaling sit inside each pipeline, fitted on training rows only. A test shows that test rows cannot influence each other's predictions.

**Leakage audit**
- **Checks, per split:** groups shared between train/validation, train/test and validation/test; records in two partitions; any group not tested exactly once per repeat.
- **Results:** 0 on every check for all 75 outer folds.
- **Independent re-check:** the dashboard recomputes these checks from the saved manifest.

**Outputs:** `reports/cv_fold_metrics.csv`, `experiment_split_manifest.csv`, `cv_predictions.csv`, `cv_summary.json`, `cv_protocol.json` and `data_audit_report.json`. They record versions, seeds, grids, group definitions, timestamps and data checksums.

**Uncertainty:** fold-level mean/std/min/max and per-repeat pooled scores. The 25 folds are correlated re-splits.

**Results (2026-10-24 run; see the reports for every fold):**

| Task | Model | Fold-level score (mean ± sd) |
|---|---|---|
| MaCFP peak HRR | mean baseline / Ridge / random forest / deployed NN | MAE 364 ± 108 / **123 ± 51** / 187 ± 86 / 152 ± 68 kW/m² |
| FLEX K | mean baseline / Ridge / random forest | MAE 0.109 ± 0.016 / **0.048 ± 0.009** / 0.056 ± 0.014 mm²/s (Ridge pooled R² 0.71) |
| FLEX extinction | majority / logistic / random forest | balanced accuracy 0.50 / **0.68 ± 0.14** / 0.58 ± 0.11; ROC-AUC 0.50 / 0.79 / 0.78 |

The deployed MaCFP network's metadata reports MAE 118.9 kW/m² from one 5-fold run. Under the repeated protocol its configuration averages 152 kW/m², and the linear Ridge model is better. The network was kept, and the dashboard shows this comparison.

## 5. Frontend and API changes

**Dashboard (`src/app.py`)**
- **Statuses:** read from the reports through `app/services/provenance.py`. A missing report gives an explicit error and treats everything as `UNVERIFIED_SOURCE`, never a silent default.
- **Header tiles:** computed from the reports.
- **Evidence tab:**
  - verified data;
  - a provenance panel listing the evidence;
  - a record inspector;
  - CO censoring disclosed;
  - wording such as "ranks tests as run, not intrinsic toxicity".
- **Audit tab:** one view per task, with the leakage re-check, model diagrams, fold strips, measured-vs-predicted plots, report downloads, and the retired evaluation with its evidence.
- **Prediction tab:** the preview uses two real burns (a fixed selection rule) with out-of-fold predictions, labelled `REAL_MEASUREMENT` / `MODEL_PREDICTION`.
- **Sweep and sandbox:** labelled `SIMULATED_RESULT`.
- **Languages:** every config and report text is bilingual (English/中文) via `*_zh` keys.

**API (`app/main.py`)**
- **New endpoint:** `GET /api/v1/research/provenance`.
- **Ranking:** returns `data_status`, `result_type` and `warning`.
- **Sweep:** returns `result_type: SIMULATED_RESULT`.
- **Prediction:** returns `result_type: MODEL_PREDICTION`, the training-data status, and a note pointing to the repeated evaluation.

**Display fixes:** metric cards in a row now share one height.

## 6. Change log

- **Added**
  - `src/fetch_nasa_psi_tables.py`, `src/provenance_audit.py`, `src/group_cv.py`, `src/run_cv_evaluation.py`, `src/build_task_datasets.py`
  - `app/services/provenance.py`, `app/services/psi_parsing.py`
  - `config/target_registry.json`, `config/data_sources.json` (replaces `data_provenance.json`)
  - `data/raw/nasa_psi/*`, `data/processed/tasks/*`
  - new `reports/*`
  - `.gitattributes`
  - four test files
  - `docs/SCIENTIFIC_AUDIT.md`
  - READMEs in `src/legacy`, `data/legacy`, `reports/legacy`
- **Changed**
  - `src/app.py`, `app/main.py`, `app/services/research_engine.py`, `src/build_dashboard_data.py`
  - `src/train_spacecraft_fire_model.py` (default data = verified MaCFP CSV)
  - `config/simulation_model.json` (bilingual text)
  - `data/processed/dashboard_records.csv` and companions (rebuilt from the official tables)
  - `README.md`
- **Renamed:** `src/download_nasa_tables.py` → `src/generate_synthetic_demo_tables.py` (same values, honest labels, writes to `data/demo/`).
- **Quarantined, not deleted** (moved with `git mv`):
  - **To `src/legacy/`:** 10 retired scripts (import blocked unless `FLAME_ALLOW_LEGACY=1`).
  - **To `data/legacy/`:** 6 retired data files.
  - **To `reports/legacy/`:** 3 retired reports.
  - **To `config/legacy/`:** the old CV config.

## 7. Commands executed and test results

```
python -m pytest tests -q                    # before changes: 8 passed
python -m src.fetch_nasa_psi_tables          # downloaded + verified both tables
python -m src.fetch_nasa_psi_tables --verify # VERIFIED (ok) x2
python -m src.build_dashboard_data           # 403 rows (129 BASS-II + 274 FLEX)
python -m src.build_task_datasets            # 255 rows (K task), 204 rows (extinction task), 89 exclusions
python -m src.provenance_audit               # BASS_II, FLEX, MACFP_PMMA_CONE: VERIFIED_SOURCE
python -m src.run_cv_evaluation              # 3 tasks evaluated, 2 NOT_EVALUATED, split audit passed
python -m pytest tests -q                    # after changes: 52 passed
```

The full pipeline was rerun from a clean rebuild and reproduced identical scores.

## 8. Remaining limitations and open items

- **BASS-II has no evaluable target.** Spread rates would have to be transcribed from report tables (for example NASA/TM-20210011385 Table 5.1) and verified before a spread-rate task can exist.
- **The FLEX grouping unit is the test date.** Tests on one day share a chamber fill. Tests on different days can still share hardware and campaign conditions, so independence is approximate.
- **FLEX test numbers drift from row 156.** The CSV's "FLEX Test #" no longer matches the NASA/TP numbering from that row on (the identifier is used as the key, so results are unaffected). Ten Completion and twenty Disruption rows still list an extinction diameter; the label uses "Test end" only.
- **BASS-II fan and air displays** sometimes record sequences of values within one test. They are flagged as `multiple_values` and not used numerically.
- **The MaCFP model is an after-test estimator for ground-based PMMA.** It is not validated for microgravity, spacecraft or other materials. Its 350 kW/m² hazard flag cannot be validated because every measured peak is above it.
- **The 2D sandbox is educational.** The temperature field is an illustrative Gaussian, not a CFD solution.
- **`app/services/data_services.py` is unused** and looks for `dataset_catalog.csv` in the wrong folder. It was left untouched because nothing imports it.
- **Apollo 11 launch replay** (section 9):
  - Values between NASA's event times are interpolated for drawing only.
  - The "what if" coast ignores air drag and Earth's flattening.
  - The engine-flame look, the clouds and the camera are illustrations.
  - The 1969 primary report is checked from a non-NASA mirror (no NASA-hosted copy was found).

## 9. Addition: Apollo 11 launch data (2026-10-24)

**Why:** the 2D sandbox tab gained a rocket-launch replay that answers "when does the rocket's flame go out?". Every
number in it had to be real NASA data, not typed in.

**Source and provenance** (dataset `APOLLO11_ASCENT` in `config/data_sources.json`):
- **Publication:** NASA/SP-2000-4029 *Apollo by the Numbers* (R. W. Orloff, 2000), NTRS 20010008244, which NTRS marks "government public use permitted".
- **Pages used:** printed page 103 (*Apollo 11 Ascent Phase* and *Earth Orbit Phase*) and page 105 (*Timeline*, launch rows).
- **Files stored:**
  - NTRS's full-text file of the book, byte-for-byte: downloaded twice, identical SHA-256, registered and fetchable with `src/fetch_nasa_psi_tables.py`.
  - The two page scans, extracted losslessly from NASA's 255 MB PDF, which is registered by checksum but not stored.
- **Transcription:** values were typed by hand, exactly as printed, into `data/transcribed/`. `src/build_rocket_ascent.py` then checks them:
  - **OCR:** every one of the 99 numbers and 14 times on page 103 is found in NASA's own OCR text of that page, and no OCR number is left untranscribed.
  - **Shared times:** the 11 event times shared by the ascent table and the timeline are identical.
  - **Burn durations:** each cutoff time minus its burn duration equals the printed ignition time (S-IC −6.40 s, S-II 164.00 s, S-IVB 552.20 s).
  - **Primary report:** 18 of 19 values shared with MPR-SAT-FE-69-9 Table 4-1 agree within its printed precision.
- **Status:** `src/provenance_audit.py` recomputes all of this, rebuilds the four processed tables byte-for-byte, and sets `VERIFIED_SOURCE`.

**Findings in the source:**
- **Copying error in the book.** The timeline prints the spent first stage's apex at 000:04:59.10. The primary report gives 269.1 s (115.0 km), and a drag-free two-body coast from NASA's separation state gives 269.1 s (114.9 km). The value is flagged and not used.
- **Misprinted unit.** The ascent table's "Event Duration (deg E)" column is in seconds.
- **Suspect value.** One longitude (S-IC centre cutoff, −81.1517°) lies west of the pad. It is kept as printed and is not used.

**Physics** (`app/services/rocket_ascent.py`; no fitted parameters and no machine-learning model):
- **Methods:**
  - two-body (Kepler) motion from NASA's space-fixed speed and flight-path angle;
  - circular orbital speed;
  - WGS-84 radius and GM;
  - U.S. Standard Atmosphere 1976 pressure (0–86 km).
- **Checks against NASA** (all 8 pass):
  - speed at liftoff = Earth's spin at the pad: 408.632 vs 408.645 m/s;
  - parking-orbit period from the insertion height and speed: 88.184 vs 88.18 min;
  - spent S-IC apex time / altitude / downrange: 269.104 s, 114.9 km, 327.4 km vs 269.1 s, 115.0 km, 327.4 km;
  - spent S-II apex: 586.5 s, 188.8 km, 1,860.7 km vs 587.0 s, 188.8 km, 1,862.9 km.
- **Interpretation:** the spent-stage values in the primary report were themselves computed by NASA (no tracking data), so those rows check our physics against NASA's calculation, not against a measurement.

**Why the trained models are not used here:** the project's trained models describe ground-based PMMA cone burns and
ISS droplet flames. A rocket engine is a different kind of flame (high-pressure combustion with its own oxidiser),
so applying them would be invalid.

**Commands:** `python -m src.build_rocket_ascent`, `python -m src.provenance_audit`. Tests: `tests/test_rocket_ascent.py`
(14 tests). The full suite went from 52 to 66 passed.
