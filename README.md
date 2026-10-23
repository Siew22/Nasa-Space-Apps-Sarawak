# Flame in Freefall: Physics-Informed Microgravity Combustion Research & STEM Platform

[![NASA Space Apps Challenge](https://img.shields.io/badge/NASA%20Space%20Apps-Sarawak%202026-blue.svg)](https://www.spaceappschallenge.org/)
[![Python Version](https://img.shields.io/badge/python-3.11-brightgreen.svg)](https://www.python.org/)
[![Tests Passing](https://img.shields.io/badge/tests-5%2F5%20passed-success.svg)](tests/)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

> **NASA Space Apps Challenge Topic**: *Flame in Freefall: AI-Powered Fire Safety Insights from Microgravity Combustion Data*  
> An interactive, physics-grounded scientific research and STEM education platform bridging NASA flight combustion archives with first-principles fluid dynamics to deliver transparent human spaceflight fire safety insights.

---

## 🚀 Quick Start for Collaborators & Teammates

> 💡 **Important Note regarding ML Models:**  
> Our interactive frontend, evidence database, and 2D physics sandbox work **out-of-the-box without requiring pre-downloaded large model weights!**  
> The core analysis and STEM simulations are completely driven by the first-principles physics engine and empirically curated NASA data.

### 1. Environment Setup
We recommend using **Python 3.11** under Linux / WSL2 (Ubuntu) / macOS:

```bash
# Clone the repository
git clone https://github.com/Siew22/Nasa-Space-Apps-Sarawak.git
cd Nasa-Space-Apps-Sarawak

# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies (strictly pinning pyarrow==16.1.0 and numpy==1.26.4 to avoid conflicts)
pip install --upgrade pip
pip install -r requirements.txt
pip install "pyarrow==16.1.0"
```

### 2. Launch the STEM Interactive Dashboard
```bash
streamlit run src/app.py
```
Open your browser at **`http://localhost:8501`** to experience all 4 research and educational modules!

### 3. Run Automated Physics Sanity Checks
Validate the 5 conservation laws and boundary limits of our 2D simulation engine:
```bash
pytest tests/ -v
```

### 4. (Optional) Train the Deep Learning Model Locally
To generate your own TensorFlow neural network weights locally (GPU supported):
```bash
# Automatically outputs spacecraft_fire_multitask_v1.keras in models/
python -m src.train_spacecraft_fire_model
```

---

## 🌌 Project Background & Challenge Problem

Under normal Earth gravity ($1g$), hot combustion products generate strong buoyancy-driven convection, forming familiar upward flame plumes. However, in microgravity ($\mu g$):
1. **Absence of Buoyancy:** Natural convection is blocked.
2. **Diffusion & Ventilation Dominated:** At extremely low flow speeds, flames suffer from **"Oxygen Starvation Quenching"** due to accumulated product gases; moderate ventilation, however, can trigger rapid flame spread.
3. **Fragmented NASA Flight Data:** Historical data from BASS (Solid fuels) and FLEX (Droplets) belong to fundamentally different physical regimes.

A fatal mistake in many hackathon projects is dumping heterogeneous datasets into a single "black-box" model, leading to **Shortcut Learning** and **Data Leakage**.  
**Our solution completely rejects hardcoded pseudo-rules. We built a scientifically auditable, physically decoupled platform.**

---

## 🔬 Core System Architecture

Our interactive dashboard features four core modules designed for both aerospace engineers and STEM students:

### 1. Decoupled NASA Evidence Explorer
* **Solid Surface Combustion Track (BASS-II)**: DOI `10.60555/4qc4-de67`. Explores the effect of forced flow on solid flame spread rates and toxic CO/CO₂ yields.
* **Liquid Droplet Combustion Track (FLEX)**: DOI `10.60555/mbq8-0451`. Explores ambient pressure and dilution effects on extinction diameters ($d_{ext}$).
* **Zero Hardcoding**: Material fire hazard rankings are dynamically aggregated from empirical NASA statistics using native Python `groupby` logic.

### 2. STEM Parametric Sensitivity & Physical Narration
Students can freely select independent variables (Forced flow speed $u$, Gravity $g$, Oxygen concentration $O_2$) to execute continuous parametric sweeps:
* **Dual-Axis Fluid Response Curves**: Dynamically calculates the **Richardson Number $\log_{10}(Ri)$** and the Effective Oxygen Flux Index.
* **Scientific Narration**: Provides auto-generated, first-principles explanations—clarifying why flames self-extinguish in quiescent microgravity but spread aggressively under forced ventilation.

### 3. 2D Microgravity Physics Spatiotemporal Sandbox
Calculates 2D spatial grid fields using a first-principles dynamics solver:
* **Strict Computed Variables**: Ideal gas density differences, Net buoyancy $F_{net}$, Reduced gravity $g' = g \frac{\Delta \rho}{\rho}$, Characteristic buoyancy velocity $u_b = \sqrt{g' L}$, and damped centroid kinematics $[x_c(t), y_c(t)]$.
* **Reduced-Order Educational Visualization**: Renders a 2D Gaussian temperature heatmap and induced velocity vectors.
* **Interactive Comparison**: By dragging the timeline and gravity slider, users visually witness the transition from zero-g spherical diffusion to 1-g upward buoyant plumes.

### 4. Scientific Audit & 25-Fold Validation Monitor
* Executes **5 Seeds × 5 Folds = 25 Rounds of strict GroupKFold cross-validation** across 80 independent experiment groups.
* Enforces absolute zero data leakage (`assert len(train_groups & val_groups) == 0`).
* Transparently compares the generalization capabilities (Mean ± Std) across Dummy Baselines, Random Forests, and Multi-Task Neural Networks.

---

## 🧪 Scientific Boundaries & Physics Sanity Checks

We implemented strict physical conservation tests in `tests/test_simulation_physics_limits.py` to prove our engine computes real physics, not pre-scripted animations:

| Test Case | First-Principles Rule Validated | Status |
| :--- | :--- | :---: |
| `test_microgravity_limit_zero_buoyancy_motion` | At $g=0$, vertical buoyancy, displacement, and vertical velocity are strictly 0. | **PASSED** ✅ |
| `test_gravity_monotonic_buoyancy_scaling` | Buoyancy velocity scale $u_b$ strictly increases monotonically with $\sqrt{g}$. | **PASSED** ✅ |
| `test_plume_deflection_angle_comparison` | Under identical flow, Earth plumes are upright while Lunar plumes deflect downwind ($\theta_{earth} > \theta_{moon}$). | **PASSED** ✅ |
| `test_temporal_continuity_and_smooth_centroid` | Centroid starts at origin at $t=0$; diffusion radius expands continuously without jumps. | **PASSED** ✅ |
| `test_unphysical_temperature_rejected` | Pydantic barrier strictly rejects unphysical inputs like temperature inversions. | **PASSED** ✅ |

---

## 📂 Repository Structure

```text
├── app/                            # FastAPI backend & core computational services
│   ├── main.py                     # REST API routing
│   ├── schemas.py                  # Pydantic strict typing & 2D Grid schemas
│   └── services/                   # Business logic layer
│       ├── data_service.py         # Track separation & physics dictionary
│       ├── simulation_service.py   # 2D spatial grid physics solver
│       └── research_engine.py      # Parametric sweep & narration engine
├── src/                            # Interactive frontend & data pipelines
│   ├── app.py                      # Streamlit 4-tab interactive dashboard
│   └── generate_scientific_validation.py # 5x5 GroupKFold scientific audit script
├── config/                         # Configuration & Standards
│   ├── dataset_catalog.csv         # Source files, DOIs, and licensing
│   └── scientific_validation.json  # Physical constants & parameter limits
├── data/                           # Real observational data directory
│   ├── raw/                        # Original NASA experimental tables
│   └── processed/                  # Normalized tables (dashboard_records.csv)
├── reports/                        # Auditable evaluation reports (JSON/CSV)
│   ├── data_audit_report.json      # Data provenance & quality review
│   ├── cv_fold_metrics.csv         # Detailed metrics for all 25 folds
│   └── experiment_split_manifest.csv # Row-by-row group isolation manifest
├── tests/                          # Automated testing suite
│   └── test_simulation_physics_limits.py # Physical limit assertions
├── pytest.ini                      # Pytest configurations
└── requirements.txt                # Strictly pinned dependencies
```

---

## 📚 Data Provenance & Citations

All real data ingested in this project adheres to NASA's Open Science protocols:

1. **NASA BASS-II (Burning and Suppression of Solids II)**  
   * NASA Physical Sciences Informatics: [PSI-25](https://psi.nasa.gov/physci/repo/data/investigations/PSI-25)  
   * DOI: `10.60555/4qc4-de67` | License: **CC0-1.0**  
2. **NASA FLEX (Flame Extinguishment Experiment)**  
   * NASA Physical Sciences Informatics: [PSI-69](https://psi.nasa.gov/physci/repo/data/investigations/PSI-69)  
   * DOI: `10.60555/mbq8-0451` | License: **CC0-1.0**  
3. **NASA Saffire Spacecraft Fire Safety Demonstration**  
   * NASA Technical Reports: NASA/TM-20205008182 / NASA/TM-20240002981  
4. **MaCFP Condensed Phase Material Database**  
   * International Association for Fire Safety Science (IAFSS): [MaCFP GitHub](https://github.com/MaCFP/matl-db)

---

## 👥 Contributors

* **Team**: Sarawak Space Apps Team 2026
* **Repository**: [https://github.com/Siew22/Nasa-Space-Apps-Sarawak](https://github.com/Siew22/Nasa-Space-Apps-Sarawak)
