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
## 9. Apollo 11 launch replay: added, then removed (2026-10-24)

A rocket-launch replay ("when does the engine flame go out?") was added to the 2D sandbox tab, with Saturn V flight
values transcribed from NASA/SP-2000-4029 (NTRS 20010008244) and checked against NASA's OCR text and the 1969
flight evaluation report MPR-SAT-FE-69-9. It was removed the same day because it falls outside the challenge:
a rocket engine is high-pressure combustion carrying its own oxidiser, not the accidental fires in a crewed
cabin that the BASS-II, FLEX and MaCFP data describe, and none of the project's models apply to it.

Removed: dataset `APOLLO11_ASCENT` from `config/data_sources.json` and `src/provenance_audit.py`,
`app/services/rocket_ascent.py`, `src/build_rocket_ascent.py`, `assets/rocket_launch.html`, `data/processed/rocket/`,
`data/raw/nasa_ntrs/`, `data/transcribed/`, `data/reference/` and `tests/test_rocket_ascent.py`. Everything is
recoverable from git commit `13ab59a`.


## 10. Deployed FLEX models: the AI Predictions tab (2026-10-25)

**Why:** the two FLEX models were only scored in cross-validation, so the dashboard had no AI a user could run on
microgravity data (the only runnable model was the ground-based MaCFP network).

**What is deployed** (`src/train_flex_models.py` → `models/flex_burning_rate_ridge_v1.json`, `models/flex_extinction_logistic_v1.json`):
- **Families:** Ridge for the burning-rate constant and logistic regression for flame-out. They are the best-scoring
  families in `reports/cv_summary.json` and the ones this report and the README quote.
- **Recipe:** identical to the CV (imported from `src/run_cv_evaluation.py`): median imputation, standard scaling,
  Ridge / class-balanced logistic regression, same tuning grid.
- **Final setting:** the grid value with the best mean MAE / log loss over group-disjoint 5-fold splits of all rows,
  averaged over the 5 protocol seeds, then refit on all rows (255 and 204 rows).
  - Ridge alpha → mean MAE: 0.1: 0.0479, 1: 0.0478, 10: 0.0474; chosen alpha = 10.
  - Logistic C → mean log loss: 0.1: 0.5218, 1: 0.5252, 10: 0.5349; chosen C = 0.1.
- **Performance to quote:** the nested CV in `reports/cv_summary.json`, which evaluated this whole procedure on unseen
  test days. It is copied into each model file, and a test checks the copy matches.

**How it is served** (`app/services/flex_inference.py`):
- **Format:** the models are linear, so they are stored as plain numbers (medians, scaling, coefficients), not pickles.
  A test confirms the numbers reproduce scikit-learn's predictions to 1e-9.
- **Explanations:** each prediction is split exactly into one contribution per input, measured from the average training test.
- **Input checks:** values outside the tested ranges, and gas mixes above 100 %, are refused.
- **Nearest real tests:** same fuel, pressure within 10 %, O₂ within 2 points, CO₂ and He within 5 points.

**Limits stated in the tab:**
- **Flame-out score:** class-balanced training (177 flame-outs to 27 burn-outs) moves scores toward 0.5, so the score
  ranks risk and 0.5 is the evaluated decision line, but it is not a calibrated probability.
- **Scope:** n-heptane and methanol droplets in the tested pressure and gas-mixture ranges only.
- **Model vs data:** the model's call can disagree with the nearest real tests; both are shown side by side.

**Commands:** `python -m src.train_flex_models`. Tests: `tests/test_flex_models.py` (16 tests).

## 11. Crew fire-safety takeaways on the front page (2026-10-25)

**Why:** the challenge asks for fire-safety insights for human space exploration. The findings were spread across tabs.

**What:** six cards above the tabs (`app/services/insights.py`), every number recomputed on each data change:

| Takeaway | Evidence | Computed from |
|---|---|---|
| Airflow keeps a fire alive in orbit | `SIMULATED_RESULT` | the STEM sweep formulas at g = 0, 21 % O₂ |
| More oxygen, faster fire | `REAL_MEASUREMENT` | FLEX burning-rate means by O₂ group, tests without added CO₂ or He |
| CO₂ in the air slows the flame | `REAL_MEASUREMENT` | FLEX burning rate and flame-out share with ≥ 15 % CO₂ vs none |
| Don't count on a fire putting itself out | `REAL_MEASUREMENT` | FLEX outcome counts (disrupted tests excluded) |
| How much can burn matters for toxic gas | `REAL_MEASUREMENT` | BASS-II mean final CO, PMMA forms with ≥ 5 tests |
| No "up" in orbit | `SIMULATED_RESULT` | the 2D model at 9.80665 vs 0 m/s², no airflow |

**Honesty notes:**
- **Raw group comparisons:** the FLEX comparisons are not controlled experiments. The CO₂ card reports the average O₂ in
  both groups, and the AI Predictions tab separates the effects with a model.
- **Where the methanol data disagree:** with CO₂, methanol flames went out slightly *less* often (91 % vs 96 %), so the
  card claims only that CO₂ slows the flame, not that it always puts it out.
- **Unverified data:** if a dataset's provenance status is not `VERIFIED_SOURCE`, its cards show that status instead of
  `REAL_MEASUREMENT`.
- **Recommendations:** the "➜" lines are general fire-safety practice that follows from each finding, not results of
  the data.

**Tests:** `tests/test_insights.py` (12 tests).

## 12. Merged: local LLM research assistant from the team repository (2026-10-26)

**What:** ten commits uploaded to the team repository (Siew22/Nasa-Space-Apps-Sarawak, 2026-10-26) were merged. Code the
dashboard already had was kept; code it lacked was taken from those commits:
- **AI module:** `app/services/local_llm_service.py`, `config/ai_config.json`, `scripts/` and `tests/test_local_ai_pipeline.py`.
- **Data:** the corpus, training splits and FAISS index, moved to the `data/knowledge/`, `data/ai_training/` and
  `data/vector_index/` paths the scripts use.
- **Run records:** the training and benchmark reports.
- **Settings:** `find_project_root()` and `read_ai_config()`.
- **Dashboard:** the SI question-answering tab, added as tab 6.

**Changes made while merging, so the dashboard keeps running:**
- **The tab now exists:** it is created in `st.tabs` (`tab_si_intelligence` was used but never created, a NameError on start).
- **Settings keep the API working:** `allowed_origins()` stays in `app/settings.py`; `app/main.py` imports it.
- **Status keys added:** `offline_mode_enforced`, `base_model` and `base_model_license`, which the tab and its tests read.
- **No crash without the engine:** without torch, transformers and faiss (`requirement.txt`, not installed on Streamlit
  Cloud), the tab explains how to run it locally and shows the last benchmark. Engine errors are shown in the tab.
- **Tests skip instead of failing:** the local AI tests skip when torch or the downloaded weights are missing.
- **`tf-keras` version:** pinned to 2.16.0, because `tf-keras==2.21.0` cannot be installed with `tensorflow==2.16.2`
  (pip: ResolutionImpossible), which would break the cloud install.
- **Housekeeping:** committed `scripts/__pycache__` files removed; downloaded weights (`models/base_models/`,
  `models/embeddings/`) git-ignored.
- **Labels:** "official DOI verification" and "original text" relabelled as "DOI" and "passage", and the corpus badged
  `UNVERIFIED_SOURCE`, because of the checks below.

**Checks on the knowledge corpus (2026-10-26, against the NTRS API and DOI resolvers):**

| Corpus citation | NTRS ID used | Result |
|---|---|---|
| Saffire, NASA/TM-20205008182 | 20205008182 | not found; report number not on NTRS. The DOI 10.60555/0t15-1z43 is real: NASA PSI-98, Spacecraft Fire Safety Experiment-I (a dataset) |
| BASS-II, NASA/TM-2016-219083 | 20160005786 | the ID belongs to an unrelated presentation on photon counting; report number not on NTRS |
| FLEX, NASA/CR-2017-219502 | 20170008529 | not found; report number not on NTRS |
| SoFIE | 20200000361 | real (SoFIE project presentation). Its abstract describes ISS (microgravity) tests, not the Lunar/Martian-gravity testing the passage states |

- **FLEX passage contradicts the data:** the passage places radiative extinction at 0.8–1.5 mm. In the verified FLEX
  table the median extinction diameter is 2.77 mm for heptane (7 % of extinctions in 0.8–1.5 mm) and 1.67 mm for methanol.
- **Training data:** the question-answer examples reuse one target answer whatever the passage says. For the FLEX test
  item it states that flames "cannot sustain steady combustion" in quiescent microgravity, which droplet burning (the
  d²-law) contradicts.
- **Benchmark:** fine-tuning on 12 examples (3 steps) changed neither citation match (0.67) nor refusal F1 (0.00) on the
  4 test questions, and the dataset manifest's total (20) does not match its splits (12 + 1 + 4).

**Status:** the SI tab is an optional, local-only module and is not part of the evaluated results. The corpus needs
verbatim NASA text with checked citations, and the training answers need to come from their passages, before the
module's answers can be treated as evidence.
