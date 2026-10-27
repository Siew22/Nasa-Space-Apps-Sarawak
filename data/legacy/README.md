# Retired data files (quarantined)

Moved here during the 2026-10-24 scientific-integrity audit. Kept unchanged for traceability;
not used by the dashboard, the API or the current evaluation pipeline.
`reports/data_provenance_audit.json` records the evidence for each classification.

| File | Status | Evidence |
|---|---|---|
| `research_curated_dataset.csv` | SYNTHETIC_DEMO (derived) | Built by `src/legacy/research_dataset_curation.py` from the synthetic BASS-II/FLEX tables; BASS-II target computed from an input (fan setting). |
| `research_curated_dataset.metadata.json` | Unfilled template | Byte-identical to the `.template.json`; 34 `REPLACE_WITH_*` placeholders, yet declares `curated_empirical_measurements` and `contains_synthetic_rows: false`. |
| `research_curated_dataset.metadata.template.json` | Template | — |
| `macfp_empirical_fire_properties.csv` | SYNTHETIC_DEMO (fallback constants) | Every row holds the constants 50/650/320/450 from `src/legacy/ingest_macfp_data.py`; most rows cite FDS model-prediction files, not cone tests. |
| `spacecraft_fire_benchmark.csv` | SYNTHETIC_DEMO (fallback constants) | 20 rows derived from the file above: the file names differ, but every value column holds one constant (flux 50, duration 450, mean HRR 320, peak 650). The deployed model was **not** trained on it (see `models/spacecraft_fire_peak_hrr_v1.metadata.json`). |
| `logs of fire model.txt` | Unverifiable log | Output of code that was never committed; refers to `models/spacecraft_fire_multitask_v1.keras`, which does not exist. |
