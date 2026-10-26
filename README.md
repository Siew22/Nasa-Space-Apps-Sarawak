# Flame in Freefall: Physics-Informed Microgravity Combustion Research & Local Explainable AI Platform

[![NASA Space Apps Challenge](https://img.shields.io/badge/NASA%20Space%20Apps-Sarawak%202026-blue.svg)](https://www.spaceappschallenge.org/)
[![Python Version](https://img.shields.io/badge/python-3.11-brightgreen.svg)](https://www.python.org/)
[![Physics Limits](https://img.shields.io/badge/physics%20tests-5%2F5%20passed-success.svg)](tests/)
[![Offline AI](https://img.shields.io/badge/local%20LLM-Qwen2.5--1.5B%20BF16%20LoRA-orange.svg)](models/)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

> **NASA Space Apps Challenge 2026**: *Flame in Freefall: AI-Powered Fire Safety Insights from Microgravity Combustion Data*  
> An interactive, physics-grounded research intelligence and STEM educational platform that bridges multi-decade NASA spaceflight archives with first-principles fluid dynamics and locally fine-tuned, explainable artificial intelligence (XAI) for Lunar and Martian exploration fire safety.

---

## 🚀 Quick Start for Teammates & Collaborators (队友极速上手指南)

> 💡 **Notice for Collaborators**:  
> You **do not need pre-existing model weights** to run the platform!  
> The interactive dashboard, empirical flight browser, and 2D physics simulation sandbox run out-of-the-box from clean local data and analytical solvers. A single command reproduces the local LLM fine-tuning in under 1 minute.

### 1. Environment Setup (Linux / WSL2 / macOS)
```bash
# Clone the repository
git clone https://github.com/Siew22/Nasa-Space-Apps-Sarawak.git
cd Nasa-Space-Apps-Sarawak

# Create and activate Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install strictly locked dependencies (pinned to prevent NumPy 2.0 / PyArrow ABI breakage)
pip install --upgrade pip
pip install -r requirements.txt
pip install "pyarrow==16.1.0" tf-keras
```

### 2. Launch the Web Platform
```bash
streamlit run src/app.py
```
Open **`http://localhost:8501`** in your browser to interact with all live research and educational modules.

### 3. Run Automated Physics Conservation Tests
Verify the 5 first-principles boundary limit assertions:
```bash
pytest tests/test_simulation_physics_limits.py -v
```

### 4. (Optional) Run the Local AI Pipeline Offline
Train and benchmark your own local LoRA model on NVIDIA GPU (RTX 3060 12GB):
```bash
# 1. Download base weights once (3.09 GB open weights; offline afterwards)
python -m scripts.download_model_assets

# 2. Build local FAISS cosine vector index from NASA flight corpus
python -m scripts.build_local_retrieval_index

# 3. Fine-tune local LoRA adapter using native BF16 with prompt loss masking
python -m scripts.train_local_llm

# 4. Run side-by-side comparative evaluation (Base vs LoRA RAG)
python -m scripts.evaluate_local_llm
```

---

## 🌌 Scientific Problem & Core Methodology

Human spaceflight missions targeting the Moon (Artemis program) and Mars operate in alien gravitational regimes ($0g$, $0.16g$, $0.38g$) with non-terrestrial atmospheres (elevated oxygen up to 34%, reduced cabin pressures).

### The Microgravity Fire Paradox:
1. **Suppression of Buoyant Plumes**: Without Archimedean buoyancy, hot combustion products do not rise. Flames form spherical envelopes governed by slow molecular diffusion.
2. **The Ventilation Double-Edged Sword**: In quiescent microgravity ($u < 0.02\text{ m/s}$), fires choke in their own exhaust ($\text{CO}/\text{CO}_2$), triggering **Oxygen Starvation Quenching**. However, moderate ventilation from spacecraft Environmental Control and Life Support Systems (ECLSS) delivers fresh oxygen directly to the reaction zone, causing rapid concurrent flame propagation.
3. **Decoupling Physical Regimes (Anti-Shortcut Learning)**:
   Many commercial ML models fail by pooling incompatible physical data together. Our platform enforces strict **physical track separation**:
   * **Solid Fuel Track (BASS-II)**: Linear flame spread rate ($S_f$ in mm/s), solid pyrolysis, and toxic carbon monoxide yield.
   * **Liquid Droplet Track (FLEX)**: Spherically symmetric evaporation, classical $d^2$-law diameter regression, and radiative cool-flame extinction limits.

---

## 🔬 Core System Architecture (Platform Capabilities)

```text
Flame in Freefall Platform
├── 🔬 Tab 1: NASA Empirical Evidence Explorer (139 Verified BASS-II & FLEX Flight Runs)
├── 📈 Tab 2: STEM Sensitivity Analysis & Narration (Richardson Number Ri Dynamic Sweep)
├── 🧪 Tab 3: 2D Microgravity Physics Sandbox & Saturn V / Apollo 11 Staging Validator
├── 📊 Tab 4: Scientific Audit & Leakage-Audited Benchmark (5x5 GroupKFold Cross-Validation)
├── 📂 Tab 5: Post-Test Prediction & Calorimeter Heat Release Modeling (MaCFP Integration)
└── 🧠 Tab 6: Local Research Intelligence (Offline BF16 LoRA + FAISS Cosine Retrieval)
```

### 1. NASA Empirical Evidence Explorer
* **Genuine Flight Records**: Ingests **139 verified flight runs from NASA BASS-II (PSI-25)** and international liquid droplet studies (FLEX, PSI-69).
* **Zero-Hardcoding Ranking**: Dynamically aggregates actual flight measurements across materials (Thin PMMA, SIBAL fabric, Delrin rods, Japan wax candles, Nomex III) to rank carbon monoxide production ($\text{CO}$ ppm) and oxygen depletion.
* **Open Science Provenance**: Direct, clickable resolution to official NASA Physical Sciences Informatics (PSI) digital object identifiers.

### 2. STEM Sensitivity Analysis & Physical Narration
* **Dimensionless Regime Transitions**: Computes the **Richardson Number** $\log_{10}(Ri) = \log_{10}(Gr / Re^2)$ across user-defined parametric sweeps ($u_{\text{forced}}$, $g$, $X_{O_2}$).
* **Automated Scientific Narration**: Converts fluid mechanics equations into plain-language causal explanations for STEM students, clarifying the competition between forced ventilation and natural convection.

### 3. 2D Physics Spatiotemporal Sandbox & Apollo 11 Staging Validator
* **Analytical 2D Thermal Solver**: Computes real-time scalar temperature grids $T(x,y,t)$ and induced velocity fields based on reduced gravity $g' = g \frac{\Delta \rho}{\rho}$, buoyancy velocity $u_b = \sqrt{g' L}$, and thermal diffusion radius $r(t) = \sqrt{r_0^2 + 2Dt}$.
* **Apollo 11 Saturn V Trajectory Benchmark**: Interactively tracks engine staging ($T+2:16$ S-IC flameout, $T+3:24$ S-II center flameout, $T+11:39.3$ orbit insertion) and validates calculated kinetic trajectory against official **NASA SP-4029** historical flight data.

### 4. Scientific Audit & 25-Fold Group-Aware Evaluation
* **Zero Group Leakage**: Enforces strict `GroupKFold` partitioning across independent physical runs (`assert len(train_groups & test_groups) == 0`).
* **Multi-Model Transparency**: Audits Dummy Baselines, Random Forests, and Multi-Task Neural Networks, transparently publishing descriptive Mean ± Std metrics and cross-group duplicate warnings.

### 5. Local Explainable AI (XAI) & Offline Research Intelligence
* **100% Local Inference**: Powered by an open-weight `Qwen2.5-1.5B-Instruct` backbone fine-tuned locally using **Native BFloat16 LoRA** on an NVIDIA RTX 3060 (12GB VRAM). Zero remote API calls.
* **Strict Prompt Loss Masking**: Implements `-100` target-only label masking to ensure the model learns source-grounded response generation rather than memorizing prompt tokens.
* **Safe Context Budgeting**: Dynamically allocates token length budgets to guarantee target supervision signals are never truncated.
* **Mathematical Vector Retrieval**: Local `FAISS` index using `IndexFlatIP` on $L_2$-normalized dense embeddings, ensuring cosine similarity metric consistency without negative-index leakage.

---

## 🧪 Automated Physics Sanity Checks

Our test suite (`tests/test_simulation_physics_limits.py`) programmatically validates that the educational simulation adheres to first-principles physics:

| Test Assertion | Physical Law / Boundary Condition Validated | Status |
| :--- | :--- | :---: |
| `test_microgravity_limit_zero_buoyancy_motion` | At $g=0$, net buoyancy $F_{\text{net}} \equiv 0$, vertical velocity $v_y \equiv 0$, and centroid stays on horizontal axis ($y_c \equiv 0$). | **PASSED** ✅ |
| `test_gravity_monotonic_buoyancy_scaling` | Characteristic buoyancy velocity scale strictly satisfies $u_b(\text{Earth}) > u_b(\text{Mars}) > u_b(\text{Moon}) > u_b(0)$. | **PASSED** ✅ |
| `test_plume_deflection_angle_comparison` | Under identical flow, Earth buoyant plume stands upright while Lunar plume deflects downwind ($\theta_{\text{Earth}} > \theta_{\text{Moon}}$). | **PASSED** ✅ |
| `test_temporal_continuity_and_smooth_centroid` | Centroid displacement originates at origin ($t=0$); diffusion radius expands monotonically with $\sqrt{t}$. | **PASSED** ✅ |
| `test_unphysical_temperature_rejected` | Pydantic validation barrier strictly rejects temperature inversions ($T_{\text{hot}} \le T_{\text{ambient}}$). | **PASSED** ✅ |

---

## 📂 Repository Layout

```text
├── app/                                # FastAPI backend & analytical core
│   ├── main.py                         # REST API endpoints & OpenAPI specification
│   ├── schemas.py                      # Pydantic v2 schemas for simulation & validation
│   ├── settings.py                     # Deterministic anchor-based root finder
│   └── services/                       # Modular business logic
│       ├── data_service.py             # Track-separated empirical data serving
│       ├── simulation_service.py       # 2D spatiotemporal Gaussian solver
│       ├── research_engine.py          # Parametric sweep & physical narration engine
│       └── local_llm_service.py        # Local RAG & LoRA inference service (with disable_adapter)
├── config/                             # Configuration files
│   ├── ai_config.json                  # Model paths, LoRA hyperparams & offline flags
│   ├── dataset_catalog.csv             # Provenance catalog & verified DOIs
│   ├── scientific_validation.json      # Cross-validation seed & split settings
│   └── simulation_model.json           # Physical constants & parameter limits
├── data/                               # Data directory
│   ├── raw/                            # Ingested NASA PSI experimental tables
│   ├── processed/                      # Normalized records (dashboard_records.csv)
│   ├── knowledge/                      # Structured NASA flight corpus (NTRS citations)
│   ├── ai_training/                    # Train/Val/Test splits with dataset manifest
│   └── vector_index/                   # Local FAISS binary index & passage metadata
├── models/                             # Local model artifacts (excluded from Git)
│   ├── base_models/                    # Downloaded base model checkpoints
│   ├── embeddings/                     # Local sentence-transformer embeddings
│   └── local_research_llm/             # Trained project LoRA adapter
├── reports/                            # Auditable evaluation reports
│   ├── data_audit_report.json          # Data provenance & quality review
│   ├── cv_fold_metrics.csv             # 25-fold cross-validation metrics
│   ├── local_ai_training_report.json   # GPU VRAM & Loss convergence audit
│   └── local_ai_comparative_benchmark.json # Base vs LoRA side-by-side benchmark
├── scripts/                            # Reproducible pipeline scripts
│   ├── download_model_assets.py        # One-time explicit offline asset fetcher
│   ├── prepare_ai_dataset.py           # Dataset extraction & group-isolated splitting
│   ├── build_local_retrieval_index.py  # FAISS IndexFlatIP vector index builder
│   ├── train_local_llm.py              # Native BF16 QLoRA fine-tuning script
│   └── evaluate_local_llm.py           # Scientific evaluation suite (Base vs LoRA)
├── src/                                # Interactive Streamlit frontend
│   └── app.py                          # 6-module interactive aerospace dashboard
├── tests/                              # Automated test suite
│   └── test_simulation_physics_limits.py # Physics sanity check test suite
├── pytest.ini                          # Test runner configuration
└── requirements.txt                    # Pinned Python package dependencies
```

---

## 📚 Official NASA Data Sources & Provenance

All data ingested into this system adheres to open-science data rights and is credited to the respective flight investigations:

1. **NASA BASS-II (Burning and Suppression of Solids II)**  
   * International Space Station / Microgravity Science Glovebox  
   * NASA Physical Sciences Informatics: [PSI-25](https://psi.nasa.gov/physci/repo/data/investigations/PSI-25) | DOI: `10.60555/4qc4-de67` | License: **CC0-1.0**  
   * Reference: NASA/TM-2016-219083
2. **NASA FLEX (Flame Extinguishment Experiment)**  
   * International Space Station / Combustion Integrated Rack  
   * NASA Physical Sciences Informatics: [PSI-69](https://psi.nasa.gov/physci/repo/data/investigations/PSI-69) | DOI: `10.60555/mbq8-0451` | License: **CC0-1.0**  
   * Reference: NASA/CR-2017-219502
3. **NASA Saffire Spacecraft Fire Safety Demonstration Project**  
   * Cygnus Uncrewed Cargo Vehicle Flight Demonstrations (Saffire I–VI)  
   * NASA Technical Reports Server: [NASA/TM-20205008182](https://ntrs.nasa.gov/citations/20205008182) | DOI: `10.60555/0t15-1z43`
4. **NASA SoFIE (Solid Fuel Ignition and Extinction)**  
   * International Space Station / Lunar Exploration Precursor  
   * NASA Technical Reports Server: [NASA NTRS 20200000361](https://ntrs.nasa.gov/citations/20200000361)
5. **NASA Apollo 11 Saturn V Ascent Aerodynamics**  
   * NASA Historical Data Series: NASA SP-4029 (Saturn V Launch Vehicle Flight Evaluation Report)
6. **IAFSS MaCFP Condensed Phase Material Database**  
   * International Association for Fire Safety Science: [MaCFP GitHub](https://github.com/MaCFP/matl-db)

---

## 👥 Contributors & Team

* **Team**: Sarawak Space Apps Team 2026
* **Competition Chapter**: NASA Space Apps Challenge — Sarawak, Malaysia
* **Repository**: [https://github.com/Siew22/Nasa-Space-Apps-Sarawak](https://github.com/Siew22/Nasa-Space-Apps-Sarawak)
* **Live Web Platform**: [https://nasa-space-apps-sarawak.streamlit.app](https://nasa-space-apps-sarawak.streamlit.app)
