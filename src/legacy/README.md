# Retired scripts (quarantined)

These scripts were moved here during the 2026-10-24 scientific-integrity audit. They are kept
for traceability only. None of them is used by the dashboard or the API, and importing this
package exits unless `FLAME_ALLOW_LEGACY=1` is set. Line numbers refer to the files in this folder.

| Script | Why it was retired |
|---|---|
| `generate_scientific_validation.py` | Ran the old 5×5 CV on `data/legacy/research_curated_dataset.csv`, which is built from synthetic tables and contains a BASS-II target computed from an input feature. Superseded by `src/run_cv_evaluation.py`. |
| `research_dataset_curation.py` | Builds that dataset. BASS-II `target_rate_mm_s` = fan setting × 0.0016 × 1.5 (label leakage, line 97); BASS-II extinction label = rule `final CO <= 2 ppm` (line 95); silent defaults for pressure, O₂ and material (lines 42–44). Combines BASS-II spread and FLEX burning rate into one target despite different meanings. |
| `physics_engine.py` | Feature formulas with uncited material constants; substring matching sends most real material names to a default. |
| `train_model.py` | Labels built from its own inputs (lines 65–66); reads a raw file that does not exist; group-unaware split. |
| `run_research_experiment.py` | Reads column names that do not exist in its input (cannot run); "PINN" naming without a physics loss. |
| `split_audit.py` | Hardcoded `"all_train_validation_test_group_overlaps_zero": True` (backed by an earlier check, but only checks group overlap). |
| `evaluate_baseline.py` | Uses every non-metadata column as a feature unless told otherwise; `GroupKFold` repeats give identical folds. |
| `ingest_macfp_data.py` | MaCFP files have a units row, so parsing failed; a bare `except: continue` hid it and the fallback wrote constants (50/650/320/450) as if measured. |
| `build_space_fire_benchmark.py` | Turns those constants into `data/legacy/spacecraft_fire_benchmark.csv` (20 rows whose values are all the same constants). |
| `simulation_service.py` | Byte-identical, unused duplicate of `app/services/simulation_service.py`. |

The current pipeline is described in the project README (Reproducible pipeline).
