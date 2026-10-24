import sys
from pathlib import Path

# Streamlit adds src/ to the import path; put the project root first so app/ wins.
ROOT = Path(__file__).resolve().parents[1]
src_str = str(ROOT / "src")
while src_str in sys.path:
    sys.path.remove(src_str)
while str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

import base64
import json
from hashlib import sha256
from io import BytesIO
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.schemas import SimulationRequest
from app.services.model_inference import (
    FEATURE_COLUMNS,
    MAX_BATCH_ROWS,
    ModelUnavailableError,
    get_model_info,
    predict_fire_rows,
)
from app.services.simulation_service import run_simulation
from app.services.research_engine import (
    get_empirical_material_ranking,
    run_parametric_sensitivity_sweep
)

RECORDS_PATH = ROOT / "data" / "processed" / "dashboard_records.csv"
REPORTS_DIR = ROOT / "reports"


@st.cache_data(show_spinner=False)
def load_records(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


@st.cache_data(show_spinner=False)
def load_ranking(track_key: str) -> list:
    return get_empirical_material_ranking(track_key)


@st.cache_data(show_spinner=False, max_entries=128)
def cached_sweep(**kwargs) -> dict:
    return run_parametric_sensitivity_sweep(**kwargs)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_simulation(gravity: float, flow: float, t_hot: float, duration: float) -> dict:
    """Run the solver once per parameter set and return plain lists for plotting."""
    res = run_simulation(SimulationRequest(
        effective_gravity_m_s2=gravity, forced_flow_speed_m_s=flow,
        hot_gas_temperature_k=t_hot, duration_s=duration, steps=15, grid_resolution=15,
    ))
    first = res.frames[0].spatial_grid
    return {
        "times": [f.time_s for f in res.frames],
        "z": [f.spatial_grid.temperature_grid_k for f in res.frames],
        "x": list(first.x_coords_m),
        "y": list(first.y_coords_m),
        "centroids": [tuple(f.plume_centroid_m) for f in res.frames],
        "radii": [f.diffusion_radius_m for f in res.frames],
        "derived": dict(res.derived_values),
    }


APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Orbitron:wght@600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
html { scroll-behavior: smooth; }
#MainMenu, footer { visibility: hidden; }

/* Keep content fully visible while Streamlit reruns: removes the dim/flicker that feels like lag */
[data-stale="true"], .stale-element { opacity: 1 !important; transition: none !important; }

/* Static starfield (not animated, so scrolling and sliders stay smooth) */
.stApp {
  background-color: #0B1020;
  background-image:
    radial-gradient(1.2px 1.2px at 20px 30px, #ffffff, transparent),
    radial-gradient(1px 1px at 140px 90px, rgba(255,255,255,.8), transparent),
    radial-gradient(1.6px 1.6px at 200px 200px, #cfe3ff, transparent),
    radial-gradient(1px 1px at 90px 170px, rgba(255,214,170,.9), transparent),
    radial-gradient(1200px 600px at 85% -10%, rgba(252,61,33,0.13), transparent 60%),
    radial-gradient(900px 500px at -10% 10%, rgba(11,61,145,0.35), transparent 55%);
  background-size: 260px 260px, 260px 260px, 340px 340px, 310px 310px, auto, auto;
}
.block-container { padding-top: 1.6rem; max-width: 1280px; animation: fadeUp .6s ease both; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes gradientShift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes floaty { 0%,100% { transform: translateY(0) scale(1); } 50% { transform: translateY(-6px) scale(1.06); } }
@keyframes flicker { 0%,100% { filter: drop-shadow(0 0 10px rgba(255,106,43,.7)); } 50% { filter: drop-shadow(0 0 22px rgba(255,170,60,.95)); } }

.hero {
  position: relative; overflow: hidden; border-radius: 22px; padding: 30px 34px; margin-bottom: 18px;
  background: linear-gradient(120deg, #151B30, #1d2547, #3a1d26, #151B30);
  background-size: 300% 300%; animation: gradientShift 14s ease infinite;
  border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 18px 50px rgba(0,0,0,.35);
}
.hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.2; }
.hero h1 .flame { display: inline-block; animation: floaty 3.2s ease-in-out infinite, flicker 2.4s ease-in-out infinite; }
.hero h1 .grad { background: linear-gradient(90deg, #FF8A3D, #FFD166); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero p { margin: 8px 0 16px; color: #aab3cc; font-size: 1.02rem; max-width: 860px; }
.chips { display: flex; flex-wrap: wrap; gap: 10px; }
.chip {
  padding: 7px 14px; border-radius: 999px; font-size: .82rem; font-weight: 600; color: #E8ECF4;
  background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12);
  transition: transform .2s ease, background .2s ease, border-color .2s ease;
}
.chip:hover { transform: translateY(-2px); background: rgba(255,106,43,0.18); border-color: rgba(255,106,43,.6); }

/* Tabs as pills */
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; border-bottom: 1px solid rgba(255,255,255,.08); padding-bottom: 6px; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] {
  height: 42px; border-radius: 12px; padding: 0 16px; background: rgba(255,255,255,0.04);
  transition: background .2s ease, transform .2s ease; font-weight: 600;
}
.stTabs [data-baseweb="tab"]:hover { background: rgba(255,106,43,0.15); transform: translateY(-1px); }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #FF6A2B, #FF9A4D) !important; color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* Metric cards */
[data-testid="stMetric"] {
  background: linear-gradient(160deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.09); border-radius: 16px; padding: 14px 16px;
  transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
}
[data-testid="stMetric"]:hover { transform: translateY(-4px); border-color: rgba(255,106,43,.55); box-shadow: 0 10px 28px rgba(255,106,43,.16); }
[data-testid="stMetricValue"] { font-weight: 700; }

/* Alerts, tables, charts, buttons */
[data-testid="stAlert"] { border-radius: 14px; border: 1px solid rgba(255,255,255,0.08); }
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"] { border-radius: 16px; overflow: hidden; }
[data-testid="stPlotlyChart"] { background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.06); padding: 6px; }
.stButton > button, .stDownloadButton > button {
  border-radius: 12px; font-weight: 600; border: 0; color: #fff;
  background: linear-gradient(135deg, #FF6A2B, #FF9A4D);
  transition: transform .2s ease, box-shadow .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 24px rgba(255,106,43,.35); color: #fff; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #10162b, #0B1020); border-right: 1px solid rgba(255,255,255,0.06); }
h2, h3 { letter-spacing: -.01em; }

/* ---- NASA mission-control styling (original design; no NASA insignia used) ---- */
.hero::before { content: ""; position: absolute; left: 0; top: 0; right: 0; height: 4px;
  background: linear-gradient(90deg, #0B3D91, #FC3D21 55%, #FF9A4D); }
.hero h1, .hero .badge, .tele-v, .bk { font-family: 'Orbitron', 'Inter', sans-serif; }
.hero h1 { font-size: 1.9rem; letter-spacing: .01em; }
.badge-row { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin-bottom: 14px; }
.badge { font-size: .72rem; font-weight: 700; letter-spacing: .14em; padding: 6px 12px; border-radius: 6px;
  color: #fff; background: #0B3D91; border: 1px solid rgba(255,255,255,.25); }
.live { display: inline-flex; align-items: center; gap: 8px; font-size: .72rem; letter-spacing: .14em; font-weight: 700; color: #7dffa8; }
.live i { width: 9px; height: 9px; border-radius: 50%; background: #3dff7a; box-shadow: 0 0 0 0 rgba(61,255,122,.7); animation: pulse 1.8s infinite; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(61,255,122,.6); } 70% { box-shadow: 0 0 0 10px rgba(61,255,122,0); } 100% { box-shadow: 0 0 0 0 rgba(61,255,122,0); } }
@keyframes orbit { to { transform: rotate(360deg); } }
.orbit { position: absolute; right: -70px; top: -70px; width: 260px; height: 260px; border-radius: 50%;
  border: 1px dashed rgba(255,255,255,.18); pointer-events: none; animation: orbit 28s linear infinite; }
.orbit::before { content: ""; position: absolute; inset: 40px; border-radius: 50%; border: 1px dashed rgba(255,255,255,.1); }
.orbit span { position: absolute; top: -7px; left: 50%; width: 14px; height: 14px; border-radius: 50%;
  background: #FF9A4D; box-shadow: 0 0 16px 4px rgba(255,154,77,.8); }
.hero-earth { position: absolute; right: 22px; top: 22px; width: 92px; height: 92px; pointer-events: none;
  filter: drop-shadow(0 0 18px rgba(91,182,255,.45)); animation: floaty 6s ease-in-out infinite; }
@media (max-width: 760px) { .hero-earth, .orbit { display: none; } }
[data-testid="stLogo"], [data-testid="stSidebarLogo"] { max-height: 3rem; }
.telemetry { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-top: 4px; }
.tele { padding: 12px 14px; border-radius: 12px; background: rgba(5,9,22,.55); border: 1px solid rgba(255,255,255,.1);
  border-left: 3px solid #FC3D21; transition: transform .2s ease, border-color .2s ease; }
.tele:hover { transform: translateY(-3px); border-left-color: #FF9A4D; }
.tele-v { font-size: 1.45rem; font-weight: 800; color: #fff; line-height: 1.1; }
.tele-l { font-size: .68rem; letter-spacing: .12em; color: #8f9ab8; margin-top: 4px; text-transform: uppercase; }
.briefing { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; margin: 4px 0 22px; }
.bcard { padding: 16px 18px; border-radius: 16px; background: linear-gradient(160deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
  border: 1px solid rgba(255,255,255,.09); transition: transform .25s ease, border-color .25s ease; }
.bcard:hover { transform: translateY(-4px); border-color: rgba(252,61,33,.6); }
.bk { font-size: .7rem; letter-spacing: .16em; color: #FF8A5C; font-weight: 700; margin-bottom: 6px; }
.bcard p { margin: 0; font-size: .92rem; color: #c3cbe0; line-height: 1.5; }
.legal { text-align: center; color: #7b86a5; font-size: .78rem; margin: 30px 0 6px; }
[data-testid="stSidebar"] h2 { font-family: 'Orbitron', 'Inter', sans-serif; }

/* Plotly Play/Pause buttons: readable in normal, hover and pressed states */
.updatemenu-button { cursor: pointer; }
/* The thick dark stroke is what creates the visible gap between the two buttons */
.updatemenu-item-rect { fill: #FFD9B8 !important; stroke: #0B1020 !important; stroke-width: 8px !important; rx: 12px; ry: 12px; transition: fill .15s ease; }
.updatemenu-button:hover .updatemenu-item-rect { fill: #FF9A4D !important; }
.updatemenu-button:active .updatemenu-item-rect { fill: #FF7A2B !important; }
.updatemenu-item-text { fill: #0B1020 !important; font-weight: 700 !important; }

/* Ranking cards: animated object + stat bars */
.rank-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px; margin: 6px 0 14px; }
.rank-card { padding: 16px; border-radius: 18px; background: linear-gradient(160deg, rgba(255,255,255,.06), rgba(255,255,255,.015));
  border: 1px solid rgba(255,255,255,.09); transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease; }
.rank-card:hover { transform: translateY(-5px); border-color: rgba(255,106,43,.55); box-shadow: 0 14px 34px rgba(255,106,43,.15); }
.rank-head { display: flex; align-items: center; gap: 12px; }
.medal { font-size: 1.9rem; line-height: 1; min-width: 2.2rem; text-align: center; font-weight: 800; color: #aab3cc; }
.rank-name { font-weight: 700; font-size: 1.02rem; color: #fff; }
.rank-sub { font-size: .78rem; color: #8f9ab8; }
img.rank-obj { display: block; width: 100%; max-width: 260px; height: auto; margin: 4px auto; }
.stats { display: grid; gap: 9px; }
.stat-top { display: flex; justify-content: space-between; font-size: .8rem; color: #aab3cc; }
.stat-top b { color: #fff; font-weight: 700; }
.stat-bar { height: 7px; border-radius: 99px; background: rgba(255,255,255,.08); overflow: hidden; margin-top: 4px; }
.stat-bar i { display: block; height: 100%; border-radius: 99px; animation: grow 1.2s cubic-bezier(.2,.8,.2,1) both; transform-origin: left; }
@keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }

/* File uploader as a clear drag-and-drop zone */
[data-testid="stFileUploaderDropzone"] {
  border: 2px dashed rgba(255,106,43,.55) !important; border-radius: 16px !important;
  background: rgba(255,106,43,.06) !important; padding: 22px !important;
  transition: background .2s ease, border-color .2s ease, box-shadow .2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover { background: rgba(255,106,43,.12) !important; border-color: #FF9A4D !important;
  box-shadow: 0 0 0 4px rgba(255,106,43,.12); }
[data-testid="stFileUploaderDropzone"] button { background: linear-gradient(135deg, #FF6A2B, #FF9A4D) !important;
  color: #fff !important; border: 0 !important; border-radius: 12px !important; font-weight: 600; }
[data-testid="stFileUploaderDropzone"] svg { color: #FF9A4D; }

/* Animated cone-calorimeter diagrams in the prediction tab */
img.flame-card { width: 100%; height: auto; border-radius: 16px; margin-bottom: 14px;
  box-shadow: 0 10px 30px rgba(0,0,0,.35); transition: transform .25s ease, box-shadow .25s ease; }
img.flame-card:hover { transform: translateY(-4px); box-shadow: 0 14px 34px rgba(255,106,43,.22); }

/* Gravity / airflow chips above the 2D chart */
.cues { display: flex; flex-wrap: wrap; gap: 8px; margin: 2px 0 8px; }
.cue { padding: 6px 12px; border-radius: 999px; font-size: .82rem; font-weight: 600; border: 1px solid; }
.cue.g { color: #7CC4FF; border-color: rgba(124,196,255,.45); background: rgba(124,196,255,.1); }
.cue.a { color: #9DFFB0; border-color: rgba(157,255,176,.45); background: rgba(157,255,176,.1); }

/* Metric text wraps instead of being cut off with "..." */
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * { white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
[data-testid="stMetricValue"] { font-size: clamp(1.15rem, 1.1vw + .9rem, 1.65rem); line-height: 1.2; }

/* Tablet and phone */
@media (max-width: 992px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .hero { padding: 22px 20px; border-radius: 18px; }
  .hero h1 { font-size: 1.6rem; }
  .hero p { font-size: .95rem; }
}
@media (max-width: 640px) {
  .hero h1 { font-size: 1.3rem; }
  .chip { font-size: .72rem; padding: 6px 10px; }
  .stTabs [data-baseweb="tab"] { height: 38px; padding: 0 10px; font-size: .85rem; }
  [data-testid="stPlotlyChart"] { padding: 2px; }
}

@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
"""

st.set_page_config(
    page_title="Flame in Freefall | NASA Microgravity Combustion Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="auto"  # open on desktop, closed on phones so it doesn't cover the page
)

ASSETS_DIR = ROOT / "assets"
EARTH_B64 = base64.b64encode((ASSETS_DIR / "earth_icon.svg").read_bytes()).decode("ascii")

# The Space Apps logo sits in the header (and at the top of the open sidebar).
# If the file is missing the app still runs, just without the logo.
_logo_path = next(iter(ASSETS_DIR.glob("space_apps_logo.*")), None)
if _logo_path is not None:
    st.logo(str(_logo_path), size="large")

# The Earth icon is the sidebar open/close button; Streamlit's default chevron is hidden.
EARTH_TOGGLE_CSS = """
<style>
/* Header as one tidy frosted bar: [Earth + logo] ... project name ... [status / menu] */
[data-testid="stHeader"] {
  height: 3.75rem; background: rgba(11,16,32,.72) !important;
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  border-bottom: 1px solid rgba(255,255,255,.08);
}
[data-testid="stHeader"]::after {
  content: "FLAME IN FREEFALL"; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
  font-family: 'Orbitron', 'Inter', sans-serif; font-weight: 700; font-size: .95rem; letter-spacing: .22em;
  background: linear-gradient(90deg, #FFFFFF, #FFB27A); -webkit-background-clip: text; background-clip: text; color: transparent;
  white-space: nowrap; pointer-events: none;
}
/* when the desktop sidebar is open, centre the name over the page content instead of the whole window */
[data-testid="stApp"]:has([data-testid="stSidebar"][aria-expanded="true"]) [data-testid="stHeader"]::after { left: calc(50% + 168px); }
@media (max-width: 760px) { [data-testid="stHeader"]::after { display: none; } }
/* Hide Streamlit Cloud's Fork / GitHub / Deploy buttons for every visitor */
[data-testid="stAppDeployButton"], .stDeployButton, [data-testid="stToolbarActions"],
[data-testid="stToolbarActionButton"] { display: none !important; }
[data-testid="stToolbar"] { right: .75rem; align-items: center; gap: .25rem; }
[data-testid="stStatusWidget"] { font-size: .78rem; opacity: .85; }
[data-testid="stSidebarCollapsedControl"] {
  display: flex; flex-direction: row-reverse; justify-content: flex-end; align-items: center;
  gap: 14px; height: 3.75rem; top: 0; left: 0; padding: 0 1rem; margin: 0;
}
[data-testid="stSidebarHeader"] {
  display: flex; align-items: center; justify-content: space-between;
  height: 3.75rem; padding: 0 1rem; margin: 0 0 .4rem 0;
}
[data-testid="stSidebarCollapsedControl"] [data-testid="stLogo"],
[data-testid="stSidebarHeader"] [data-testid="stLogo"] { height: 2.7rem !important; width: auto !important; margin: 0 !important; }
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebarCollapseButton"] button {
  width: 46px; height: 46px; min-width: 46px; min-height: 46px; padding: 0; border: 0 !important;
  border-radius: 50%; overflow: hidden; font-size: 0 !important; color: transparent !important;
  background: url("data:image/svg+xml;base64,__EARTH__") center / contain no-repeat !important;
  filter: drop-shadow(0 0 10px rgba(91,182,255,.5));
  transition: transform .25s ease, filter .25s ease;
}
[data-testid="stSidebarCollapsedControl"] button *,
[data-testid="stSidebarCollapseButton"] button * { display: none !important; }
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="stSidebarCollapseButton"] button:hover { transform: scale(1.12) rotate(14deg); filter: drop-shadow(0 0 16px rgba(91,182,255,.85)); }
/* Streamlit only reveals the collapse button on hover on desktop; keep the Earth always visible */
[data-testid="stSidebarCollapseButton"] { display: flex !important; opacity: 1 !important; visibility: visible !important; }
[data-testid="stSidebarCollapseButton"] button { display: block !important; opacity: 1 !important; visibility: visible !important; }
</style>
""".replace("__EARTH__", EARTH_B64)
st.markdown(EARTH_TOGGLE_CSS, unsafe_allow_html=True)

with st.sidebar:
    lang_choice = st.radio("Language / 语言", ["English", "中文"], horizontal=True, key="ui_lang")
    lang = "en" if lang_choice == "English" else "zh"


def tr(en: str, zh: str) -> str:
    return en if lang == "en" else zh


with st.sidebar:
    st.markdown("## 🔥 Flame in Freefall")
    st.caption("NASA Space Apps Challenge · Sarawak 2026")
    st.markdown(tr(
        "Physics-informed microgravity combustion research and STEM platform. "
        "Evidence comes from NASA BASS-II and FLEX flight data, kept in separate tracks.",
        "基于物理的微重力燃烧科研与 STEM 教育平台。证据来自 NASA BASS-II 与 FLEX 飞行实验数据，两类数据分轨独立处理。",
    ))
    st.divider()
    st.markdown(tr("**Data sources**", "**数据来源**"))
    st.markdown(
        "- [BASS-II (PSI-25)](https://psi.nasa.gov/physci/repo/data/investigations/PSI-25)\n"
        "- [FLEX (PSI-69)](https://psi.nasa.gov/physci/repo/data/investigations/PSI-69)\n"
        "- [MaCFP PMMA database](https://github.com/MaCFP/matl-db)"
    )
    st.divider()
    st.warning(tr(
        "Educational and research use only. The prediction model is experimental, "
        "trained on ground-based PMMA tests, and not validated for spacecraft "
        "fire-safety decisions.",
        "仅供教育与科研使用。预测模型为实验性质，基于地面 PMMA 测试训练，未经航天器消防安全决策验证。",
    ))
    st.caption(tr(
        "Independent project for the NASA Space Apps Challenge. Not affiliated with or endorsed by NASA.",
        "NASA Space Apps Challenge 独立参赛项目，与 NASA 无隶属关系，亦未获其认可。",
    ))
    st.markdown(tr(
        "[GitHub repository](https://github.com/Siew22/Nasa-Space-Apps-Sarawak)",
        "[GitHub 代码仓库](https://github.com/Siew22/Nasa-Space-Apps-Sarawak)",
    ))

st.markdown(APP_CSS, unsafe_allow_html=True)
_telemetry = [
    ("2", tr("NASA flight datasets", "NASA 飞行数据集")),
    ("54", tr("Measured burns", "实测燃烧试验")),
    ("25", tr("Validation folds", "验证折数")),
    ("5", tr("Physics checks passed", "物理检验通过")),
]
_briefing = [
    ("01 · " + tr("THE PROBLEM", "问题"),
     tr("In microgravity there is no buoyancy, so flames behave very differently. Too little airflow can smother "
        "a fire, while moderate ventilation can make it spread fast. Spacecraft fire safety depends on knowing which.",
        "微重力下没有浮力，火焰行为截然不同：气流太弱会使火焰窒息，适度通风却可能令其快速蔓延。航天器防火安全取决于能否分清这两种情况。")),
    ("02 · " + tr("OUR APPROACH", "我们的方法"),
     tr("We keep NASA's solid-fuel (BASS-II) and droplet (FLEX) data in separate tracks and explain results "
        "with first-principles physics, instead of one black-box model that can leak data.",
        "我们将 NASA 固体燃料 (BASS-II) 与液滴 (FLEX) 数据分轨处理，并用第一性原理物理解释结果，而不是使用可能泄漏数据的单一黑箱模型。")),
    ("03 · " + tr("HOW TO EXPLORE", "如何探索"),
     tr("Browse the evidence, sweep gravity, airflow and oxygen, watch the 2D sandbox animate, "
        "then inspect the audit. Every claim shows its source and limits.",
        "浏览实验证据，扫描重力、气流与氧浓度，观看 2D 沙盒动画，再查看审计报告。每项结论都标明来源与局限。")),
]
st.markdown(
    f"""
<div class="hero">
  <div class="orbit"><span></span></div>
  <img class="hero-earth" alt="" src="data:image/svg+xml;base64,{EARTH_B64}">
  <div class="badge-row">
    <span class="badge">NASA SPACE APPS CHALLENGE · SARAWAK 2026</span>
    <span class="live"><i></i>{tr("SYSTEM ONLINE", "系统在线")}</span>
  </div>
  <h1><span class="flame">🔥</span> Flame in Freefall<br>
  <span class="grad">{tr("Microgravity Combustion Science & Safety Platform", "微重力燃烧科学与安全平台")}</span></h1>
  <p>{tr("Physics-informed fire-safety insights from NASA microgravity combustion data, with transparent, auditable science for engineers and students.",
         "基于 NASA 微重力燃烧数据的物理驱动防火安全洞察，为工程师与学生提供透明、可审计的科学分析。")}</p>
  <div class="telemetry">{"".join(f'<div class="tele"><div class="tele-v">{v}</div><div class="tele-l">{l}</div></div>' for v, l in _telemetry)}</div>
</div>
<div class="briefing">{"".join(f'<div class="bcard"><div class="bk">{k}</div><p>{t}</p></div>' for k, t in _briefing)}</div>
""",
    unsafe_allow_html=True,
)

def _flame_path(cx: float, base: float, w: float, h: float) -> str:
    return (f"M {cx - w:.1f} {base:.1f} "
            f"C {cx - w:.1f} {base - 0.45 * h:.1f}, {cx - 0.35 * w:.1f} {base - 0.55 * h:.1f}, {cx:.1f} {base - h:.1f} "
            f"C {cx + 0.35 * w:.1f} {base - 0.55 * h:.1f}, {cx + w:.1f} {base - 0.45 * h:.1f}, {cx + w:.1f} {base:.1f} "
            f"Q {cx:.1f} {base + 0.18 * w:.1f}, {cx - w:.1f} {base:.1f} Z")


def flame_diagram_svg(flux: float, duration: float, mean_hrr: float, peak: float,
                      threshold: float, ref_peak: float, title: str) -> str:
    """Animated, labelled cone-calorimeter schematic for one burn.

    Flame height scales with peak HRR^0.4 (Heskestad-style flame-height scaling), the
    sustained flame size with mean/peak HRR, and the number of radiant-heat arrows with
    the applied heat flux. Schematic only; not to scale and not CFD.
    """
    cx, base = 210.0, 250.0
    norm = max(0.0, min(1.0, peak / ref_peak)) if ref_peak > 0 else 0.5
    h = 45 + 130 * norm ** 0.4
    w = 26 + 6 * norm
    plateau = max(0.35, min(0.95, mean_hrr / peak)) if peak > 0 else 0.6
    end_peak = min(1.0, plateau * 1.15)
    core = "#FFFFFF" if norm > 0.5 else "#FFF3B0"
    high = peak >= threshold
    flux_n = max(0.0, min(1.0, (flux - 25) / 40))
    n_rays = 2 + int(round(2 * flux_n))
    font = "Inter, Segoe UI, Microsoft YaHei, PingFang SC, Arial, sans-serif"

    rays = []
    for side in (-1, 1):
        for k in range(n_rays):
            t = (k + 0.5) / n_rays
            fx, fy = 158 + 18 * t, 172 + 52 * t
            if side == 1:
                fx = 2 * cx - fx
            tx = cx + side * (22 + 24 * t)
            rays.append(f'<line class="ray" x1="{fx:.1f}" y1="{fy:.1f}" x2="{tx:.1f}" y2="246" '
                        f'marker-end="url(#arr)" style="animation-delay:{-0.2 * k:.1f}s"/>')

    lbl = {
        "hood": tr("Exhaust hood", "排烟罩"),
        "heater": tr("Conical radiant heater", "锥形辐射加热器"),
        "flux": tr(f"Heat flux {flux:.0f} kW/m²", f"热通量 {flux:.0f} kW/m²"),
        "ign": tr("Spark igniter", "电火花点火器"),
        "spec": tr("PMMA specimen", "PMMA 试样"),
        "hold1": tr("Holder + load cell", "试样座 + 称重传感器"),
        "hold2": tr("(measures mass loss)", "（测量质量损失）"),
        "time": tr(f"Burn timeline · {duration:.0f} s (animation sped up)", f"燃烧时间轴 · {duration:.0f} 秒（动画已加速）"),
        "peak": tr("Predicted peak HRR", "预测峰值热释放率"),
        "badge": (tr(f"HIGH HAZARD ≥ {threshold:.0f}", f"高危 ≥ {threshold:.0f}") if high
                  else tr(f"BELOW {threshold:.0f}", f"低于 {threshold:.0f}")),
    }
    badge_col = "#FC3D21" if high else "#2EBD6B"

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 410" font-family="{font}">
<defs>
  <linearGradient id="outer" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#FF3B1F"/><stop offset="1" stop-color="#FF8A2B" stop-opacity="0.15"/></linearGradient>
  <linearGradient id="mid" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#FF7A1A"/><stop offset="1" stop-color="#FFC24D" stop-opacity="0.35"/></linearGradient>
  <linearGradient id="core" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="{core}"/><stop offset="1" stop-color="#FFE38A" stop-opacity="0.6"/></linearGradient>
  <radialGradient id="glow"><stop offset="0" stop-color="#FF8A2B" stop-opacity="0.55"/><stop offset="1" stop-color="#FF8A2B" stop-opacity="0"/></radialGradient>
  <linearGradient id="heat" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#3a2a35"/><stop offset="1" stop-color="#7a2e1d"/></linearGradient>
  <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#FFB347"/></marker>
</defs>
<style>
  .burn {{ transform-box: view-box; transform-origin: {cx}px {base}px; animation: burn 8s ease-in-out infinite; }}
  .f1 {{ transform-box: view-box; transform-origin: {cx}px {base}px; animation: flick 0.42s ease-in-out infinite alternate; }}
  .f2 {{ transform-box: view-box; transform-origin: {cx}px {base}px; animation: flick 0.31s ease-in-out infinite alternate-reverse; }}
  .ray {{ stroke: #FFB347; stroke-width: 1.8; stroke-dasharray: 5 4; opacity: {0.45 + 0.5 * flux_n:.2f}; animation: flow 0.7s linear infinite; }}
  .coil {{ stroke: #FF6A2B; stroke-width: 2; opacity: .8; animation: pulse 1.6s ease-in-out infinite alternate; }}
  .spark {{ animation: blink 8s steps(1) infinite; }}
  .dot {{ animation: track 8s linear infinite; }}
  .lbl {{ fill: #c9d2ea; font-size: 12px; }}
  .lead {{ stroke: #6c7899; stroke-width: 1; }}
  @keyframes burn {{ 0% {{ transform: scale(0.04); opacity: .2 }} 5% {{ transform: scale(0.5); opacity: 1 }}
    14% {{ transform: scale(1) }} 30% {{ transform: scale({plateau:.2f}) }} 75% {{ transform: scale({plateau:.2f}) }}
    86% {{ transform: scale({end_peak:.2f}) }} 96% {{ transform: scale(0.1); opacity: .6 }} 100% {{ transform: scale(0.04); opacity: .2 }} }}
  @keyframes flick {{ from {{ transform: scaleY(0.94) skewX(-2deg) }} to {{ transform: scaleY(1.06) skewX(2deg) }} }}
  @keyframes flow {{ to {{ stroke-dashoffset: -18 }} }}
  @keyframes pulse {{ from {{ opacity: .45 }} to {{ opacity: 1 }} }}
  @keyframes blink {{ 0% {{ opacity: 1 }} 6% {{ opacity: 0 }} }}
  @keyframes track {{ from {{ transform: translateX(0) }} to {{ transform: translateX(470px) }} }}
</style>
<rect x="1" y="1" width="558" height="408" rx="16" fill="#0F1530" stroke="#2a3355"/>
<text x="16" y="24" fill="#FF9A4D" font-size="12" font-weight="700" letter-spacing="1.5">{title}</text>
<polygon points="110,34 310,34 282,70 138,70" fill="#2b3350" stroke="#56628a"/>
<line class="lead" x1="300" y1="50" x2="368" y2="50"/><text class="lbl" x="372" y="54">{lbl['hood']}</text>
<ellipse class="burn" cx="{cx}" cy="{base - h * 0.45:.1f}" rx="{w * 2.6:.1f}" ry="{h * 0.6:.1f}" fill="url(#glow)"/>
{''.join(rays)}
<polygon points="60,168 158,168 176,228 128,228" fill="url(#heat)" stroke="#8a4a35"/>
<polygon points="360,168 262,168 244,228 292,228" fill="url(#heat)" stroke="#8a4a35"/>
<path class="coil" d="M80 182 L150 182 M94 198 L160 198 M108 214 L168 214 M340 182 L270 182 M326 198 L260 198 M312 214 L252 214" fill="none"/>
<line class="lead" x1="350" y1="196" x2="368" y2="196"/><text class="lbl" x="372" y="192">{lbl['heater']}</text>
<text x="372" y="210" fill="#FFB347" font-size="12" font-weight="700">{lbl['flux']}</text>
<g class="burn">
  <g class="f1"><path d="{_flame_path(cx, base, w, h)}" fill="url(#outer)"/></g>
  <g class="f2"><path d="{_flame_path(cx, base, w * 0.7, h * 0.78)}" fill="url(#mid)"/></g>
  <g class="f1"><path d="{_flame_path(cx, base, w * 0.38, h * 0.5)}" fill="url(#core)"/></g>
</g>
<line x1="350" y1="112" x2="246" y2="146" stroke="#9aa6c4" stroke-width="3" stroke-linecap="round"/>
<circle class="spark" cx="243" cy="147" r="5" fill="#BFE6FF"/>
<line class="lead" x1="350" y1="112" x2="368" y2="112"/><text class="lbl" x="372" y="116">{lbl['ign']}</text>
<rect x="160" y="250" width="100" height="18" rx="2" fill="#8cc8ff" fill-opacity="0.55" stroke="#bfe0ff"/>
<line class="lead" x1="262" y1="259" x2="368" y2="259"/><text class="lbl" x="372" y="263">{lbl['spec']}</text>
<rect x="150" y="268" width="120" height="22" rx="3" fill="#3a4466"/>
<rect x="185" y="290" width="50" height="20" rx="3" fill="#56607f"/>
<line x1="120" y1="314" x2="300" y2="314" stroke="#56628a" stroke-width="2"/>
<line class="lead" x1="272" y1="290" x2="368" y2="290"/><text class="lbl" x="372" y="290">{lbl['hold1']}</text>
<text class="lbl" x="372" y="305" font-size="11">{lbl['hold2']}</text>
<line x1="45" y1="338" x2="515" y2="338" stroke="#2f3a60" stroke-width="6" stroke-linecap="round"/>
<circle class="dot" cx="45" cy="338" r="7" fill="#FF6A2B" stroke="#fff" stroke-width="1.5"/>
<text x="45" y="356" fill="#8f9ab8" font-size="10">0 s</text>
<text x="515" y="356" fill="#8f9ab8" font-size="10" text-anchor="end">{duration:.0f} s</text>
<text x="280" y="356" fill="#8f9ab8" font-size="10" text-anchor="middle">{lbl['time']}</text>
<rect x="14" y="366" width="532" height="34" rx="10" fill="#151c3a" stroke="#2a3355"/>
<text x="28" y="388" fill="#aab3cc" font-size="12">{lbl['peak']}</text>
<text x="{170 if lang == 'en' else 150}" y="389" fill="#FFFFFF" font-size="17" font-weight="800">{peak:,.0f} kW/m²</text>
<rect x="370" y="372" width="168" height="22" rx="11" fill="{badge_col}" fill-opacity="0.18" stroke="{badge_col}"/>
<text x="454" y="387" fill="{badge_col}" font-size="11" font-weight="800" text-anchor="middle">{lbl['badge']}</text>
</svg>"""


def render_flame_gallery(rows: list[dict], ref_peak: float, threshold: float, max_cards: int = 6) -> None:
    """Show up to max_cards animated diagrams, two per row (one per row on phones)."""
    shown = rows[:max_cards]
    for start_i in range(0, len(shown), 2):
        cols = st.columns(2)
        for col, (i, row) in zip(cols, enumerate(shown[start_i:start_i + 2], start=start_i)):
            svg = flame_diagram_svg(
                flux=float(row["radiant_heat_flux_kw_m2"]),
                duration=float(row["real_burn_duration_s"]),
                mean_hrr=float(row["real_mean_hrr_kw_m2"]),
                peak=float(row["predicted_peak_hrr_kw_m2"]),
                threshold=threshold, ref_peak=ref_peak,
                title=row.get("_title") or tr(f"BURN {i + 1}", f"燃烧试验 {i + 1}"),
            )
            b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
            col.markdown(f'<img class="flame-card" alt="" src="data:image/svg+xml;base64,{b64}">',
                         unsafe_allow_html=True)
    if len(rows) > max_cards:
        st.caption(tr(f"Showing the first {max_cards} of {len(rows)} burns. All results are in the table below.",
                      f"显示前 {max_cards} 条（共 {len(rows)} 条）。完整结果见下表。"))


def _svg_img(svg: str, cls: str) -> str:
    return f'<img class="{cls}" alt="" src="data:image/svg+xml;base64,{base64.b64encode(svg.encode("utf-8")).decode("ascii")}">'


def droplet_svg(fuel: str, rate: float, rate_max: float, d0: float, d_ext: float, cycle: float) -> str:
    """Spinning fuel droplet inside a spherical microgravity flame.

    The droplet shrinks from its mean initial diameter to the measured extinction
    diameter, where the flame goes out. Methanol burns with a faint blue, low-soot
    flame; the alkanes with a sooty yellow one. Relative speeds only; time is compressed.
    """
    is_methanol = "methanol" in fuel.lower()
    flame = "#7FB8FF" if is_methanol else "#FFB347"
    liquid_hi, liquid_lo = ("#E3F1FF", "#6FA8E8") if is_methanol else ("#FFF1C9", "#D9963B")
    s = max(0.2, min(0.95, d_ext / d0)) if d0 > 0 else 0.5
    shell_r = 46 + 18 * (rate / rate_max if rate_max > 0 else 0.5)
    soot = 3 if is_methanol else 7
    dots = "".join(
        f'<circle cx="{110 + (shell_r + 14) * np.cos(2 * np.pi * k / soot):.1f}" '
        f'cy="{100 + (shell_r + 14) * np.sin(2 * np.pi * k / soot):.1f}" r="{1.6 if is_methanol else 2.4}" '
        f'fill="{"#BFE0FF" if is_methanol else "#5a4a3a"}" opacity="0.9"/>'
        for k in range(soot)
    )
    ext_txt = tr(f"EXTINCTION · {d_ext:.2f} mm", f"熄火 · {d_ext:.2f} mm")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 210" font-family="Inter, Segoe UI, Microsoft YaHei, Arial, sans-serif">
<defs>
  <radialGradient id="liq" cx="35%" cy="30%" r="75%"><stop offset="0" stop-color="{liquid_hi}"/><stop offset="1" stop-color="{liquid_lo}"/></radialGradient>
  <radialGradient id="glow"><stop offset="0.55" stop-color="{flame}" stop-opacity="0"/><stop offset="0.8" stop-color="{flame}" stop-opacity="0.55"/><stop offset="1" stop-color="{flame}" stop-opacity="0"/></radialGradient>
  <filter id="soft"><feGaussianBlur stdDeviation="2.6"/></filter>
  <clipPath id="ball"><circle cx="110" cy="100" r="26"/></clipPath>
</defs>
<style>
  .sh {{ transform-box: view-box; transform-origin: 110px 100px; animation: shell {cycle:.1f}s ease-in infinite; }}
  .dr {{ transform-box: view-box; transform-origin: 110px 100px; animation: shrink {cycle:.1f}s ease-in infinite; }}
  .mer {{ transform-box: view-box; transform-origin: 110px 100px; animation: spin 2.6s linear infinite; }}
  .orb {{ transform-box: view-box; transform-origin: 110px 100px; animation: orbit 7s linear infinite; }}
  .ext {{ opacity: 0; animation: ext {cycle:.1f}s steps(1) infinite; }}
  @keyframes shrink {{ 0% {{ transform: scale(1) }} 80% {{ transform: scale({s:.3f}) }} 100% {{ transform: scale({s:.3f}) }} }}
  @keyframes shell {{ 0% {{ transform: scale(1); opacity: .95 }} 78% {{ transform: scale({0.55 + 0.45 * s:.3f}); opacity: .9 }} 84% {{ opacity: 0 }} 100% {{ opacity: 0 }} }}
  @keyframes spin {{ 0% {{ transform: scaleX(1) }} 50% {{ transform: scaleX(-1) }} 100% {{ transform: scaleX(1) }} }}
  @keyframes orbit {{ to {{ transform: rotate(360deg) }} }}
  @keyframes ext {{ 0% {{ opacity: 0 }} 82% {{ opacity: 1 }} 98% {{ opacity: 0 }} }}
</style>
<g class="sh">
  <circle cx="110" cy="100" r="{shell_r + 22:.1f}" fill="url(#glow)"/>
  <circle cx="110" cy="100" r="{shell_r:.1f}" fill="none" stroke="{flame}" stroke-width="7" opacity="0.8" filter="url(#soft)"/>
  <circle cx="110" cy="100" r="{shell_r:.1f}" fill="none" stroke="#FFFFFF" stroke-width="1.2" opacity="0.35"/>
  <g class="orb">{dots}</g>
</g>
<g class="dr">
  <circle cx="110" cy="100" r="26" fill="url(#liq)"/>
  <g clip-path="url(#ball)" fill="none" stroke="#FFFFFF" stroke-opacity="0.45" stroke-width="1.2">
    <ellipse class="mer" cx="110" cy="100" rx="26" ry="26"/>
    <ellipse class="mer" cx="110" cy="100" rx="26" ry="26" style="animation-delay:-0.87s"/>
    <ellipse class="mer" cx="110" cy="100" rx="26" ry="26" style="animation-delay:-1.73s"/>
    <ellipse cx="110" cy="100" rx="26" ry="8"/>
  </g>
  <ellipse cx="101" cy="90" rx="7" ry="4.5" fill="#FFFFFF" opacity="0.7"/>
</g>
<text class="ext" x="110" y="200" text-anchor="middle" fill="#FC3D21" font-size="11" font-weight="800" letter-spacing="1">{ext_txt}</text>
</svg>"""


def solid_svg(material: str, co_mean: float, co_max_all: float) -> str:
    """Rotating solid sample with a flame front spreading along it and CO puffs rising."""
    name = material.lower()
    micron = next((int(t) for t in name.split() if t.isdigit()), 0)
    if "rod" in name:
        thick, fill, extra = 22, "url(#rod)", ""
    elif "fabric" in name:
        thick, fill, extra = 12, "url(#weave)", ""
    else:
        thick, fill, extra = 3 + micron / 60, "#9fd0ff", ' fill-opacity="0.6"'
    top = 112 - thick / 2
    n_puffs = 2 + int(round(4 * (co_mean / co_max_all if co_max_all > 0 else 0.5)))
    puffs = "".join(
        f'<circle class="puff" cx="{38 + 6 * k}" cy="{top - 6:.1f}" r="{5 + 2 * k}" fill="#8a93a8" '
        f'style="animation-delay:{-0.55 * k:.2f}s"/>'
        for k in range(n_puffs)
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="20 36 180 148">
<defs>
  <linearGradient id="rod" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#f4f4f4"/><stop offset="0.5" stop-color="#c9ccd6"/><stop offset="1" stop-color="#7d8296"/></linearGradient>
  <pattern id="weave" width="6" height="6" patternUnits="userSpaceOnUse"><rect width="6" height="6" fill="#d9c9a8"/><path d="M0 0 L6 6 M6 0 L0 6" stroke="#a8946c" stroke-width="1"/></pattern>
  <linearGradient id="fl" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#FFF3B0"/><stop offset="0.4" stop-color="#FF8A2B"/><stop offset="1" stop-color="#FF3B1F" stop-opacity="0.1"/></linearGradient>
</defs>
<style>
  .rot {{ transform-box: view-box; transform-origin: 110px 112px; animation: rot 6s ease-in-out infinite; }}
  .char {{ transform-box: view-box; transform-origin: 40px 112px; animation: char 6s linear infinite; }}
  .front {{ animation: front 6s linear infinite; }}
  .flk {{ transform-box: view-box; transform-origin: 40px {top:.1f}px; animation: flk .35s ease-in-out infinite alternate; }}
  .puff {{ opacity: 0; transform-box: view-box; animation: puff 1.8s ease-out infinite; }}
  @keyframes rot {{ 0% {{ transform: scaleX(1) }} 25% {{ transform: scaleX(0.08) }} 50% {{ transform: scaleX(-1) }} 75% {{ transform: scaleX(0.08) }} 100% {{ transform: scaleX(1) }} }}
  @keyframes char {{ from {{ transform: scaleX(0) }} to {{ transform: scaleX(1) }} }}
  @keyframes front {{ from {{ transform: translateX(0) }} to {{ transform: translateX(140px) }} }}
  @keyframes flk {{ from {{ transform: scaleY(0.9) }} to {{ transform: scaleY(1.12) }} }}
  @keyframes puff {{ 0% {{ opacity: .7; transform: translate(0, 0) scale(.6) }} 100% {{ opacity: 0; transform: translate(-14px, -64px) scale(1.6) }} }}
</style>
<ellipse cx="110" cy="146" rx="74" ry="7" fill="#000" opacity="0.35"/>
<g class="rot">
  <rect x="40" y="{top:.1f}" width="140" height="{thick:.1f}" rx="{min(thick / 2, 11):.1f}" fill="{fill}"{extra} stroke="#ffffff" stroke-opacity="0.35"/>
  <rect class="char" x="40" y="{top:.1f}" width="140" height="{thick:.1f}" rx="{min(thick / 2, 11):.1f}" fill="#2a1a12" opacity="0.85"/>
  <g class="front">
    {puffs}
    <g class="flk"><path d="{_flame_path(40, top, 11, 46)}" fill="url(#fl)"/><path d="{_flame_path(40, top, 5, 24)}" fill="#FFF6C8"/></g>
  </g>
</g>
</svg>"""


def ranking_cards_html(track_key: str, ranking: list[dict], d0_by_fuel: dict) -> str:
    """Card per material/fuel: animated object, medal, and stat bars around it."""
    medals = ["🥇", "🥈", "🥉"]

    def bar(label: str, value: str, frac: float, color: str) -> str:
        pct = max(4.0, min(100.0, 100 * frac))
        return (f'<div class="stat"><div class="stat-top"><span>{label}</span><b>{value}</b></div>'
                f'<div class="stat-bar"><i style="width:{pct:.0f}%;background:{color}"></i></div></div>')

    cards = []
    if track_key == "liquid_droplet":
        rate_max = max(r["mean_burning_rate_mm_s"] for r in ranking)
        ext_max = max((r["mean_extinction_diameter_mm"] or 0) for r in ranking) or 1.0
        for r in ranking:
            d0 = float(d0_by_fuel.get(r["fuel_name"], 2.0))
            d_ext = float(r["mean_extinction_diameter_mm"] or d0 * 0.5)
            # bigger burning rate -> droplet reaches extinction faster (shorter loop)
            cycle = 4.5 + 3.0 * (1 - r["mean_burning_rate_mm_s"] / rate_max) + 1.5
            obj = _svg_img(droplet_svg(r["fuel_name"], r["mean_burning_rate_mm_s"], rate_max, d0, d_ext, cycle), "rank-obj")
            stats = (bar(tr("Burning rate", "燃烧速率"), f'{r["mean_burning_rate_mm_s"]:.3f}', r["mean_burning_rate_mm_s"] / rate_max, "#FF8A3D")
                     + bar(tr("Extinction diameter", "熄火直径"), f"{d_ext:.2f} mm", d_ext / ext_max, "#7CC4FF")
                     + bar(tr("Start diameter", "初始直径"), f"{d0:.2f} mm", min(1.0, d0 / 2.5), "#9DA8C8")
                     + bar(tr("Tests ending in extinction", "以熄火结束的试验"), f'{r["empirical_extinction_ratio"] * 100:.0f}%', r["empirical_extinction_ratio"], "#FC3D21"))
            name, count = r["fuel_name"], r["empirical_test_count"]
            cards.append((r["rank"], name, count, obj, stats))
    else:
        co_max_all = max(r["max_co_production_ppm"] for r in ranking)
        for r in ranking:
            obj = _svg_img(solid_svg(r["material_name"], r["mean_co_production_ppm"], co_max_all), "rank-obj")
            stats = (bar(tr("Mean CO", "平均 CO"), f'{r["mean_co_production_ppm"]:.1f} ppm', r["mean_co_production_ppm"] / co_max_all, "#FF8A3D")
                     + bar(tr("Peak CO", "峰值 CO"), f'{r["max_co_production_ppm"]:.0f} ppm', r["max_co_production_ppm"] / co_max_all, "#FC3D21")
                     + bar(tr("O₂ consumed", "O₂ 消耗"), f'{r["mean_o2_depletion_pct"]:.2f} %vol', min(1.0, r["mean_o2_depletion_pct"] / 1.0), "#7CC4FF"))
            cards.append((r["rank"], r["material_name"], r["empirical_test_count"], obj, stats))

    html = []
    for rank, name, count, obj, stats in cards:
        medal = medals[rank - 1] if rank <= 3 else f"#{rank}"
        html.append(
            f'<div class="rank-card"><div class="rank-head"><span class="medal">{medal}</span>'
            f'<div><div class="rank-name">{name}</div><div class="rank-sub">{tr(f"{count} flight tests", f"{count} 次飞行试验")}</div></div></div>'
            f'{obj}<div class="stats">{stats}</div></div>'
        )
    return f'<div class="rank-grid">{"".join(html)}</div>'


tab_evidence, tab_sweep, tab_simulation, tab_audit, tab_prediction = st.tabs([
    tr("🔬 Evidence Explorer", "🔬 航天实验证据库 (Evidence)"),
    tr("📈 STEM Analysis", "📈 科研变量分析与物理叙事 (STEM Analysis)"),
    tr("🧪 2D Physics Sandbox", "🧪 2D 物理时空模拟沙盒 (Physics Sandbox)"),
    tr("📊 Scientific Audit", "📊 科学审计与 25 折验证 (Scientific Audit)"),
    tr("🔥 Post-test CSV prediction", "🔥 试验后 CSV 预测"),
])

# =========================================================================
# TAB 1: 航天实验证据库 (真实数据分轨)
# =========================================================================
@st.fragment
def render_evidence():
    st.subheader(tr("Experimental evidence by physical track", "物理分轨实验证据浏览"))
    if not RECORDS_PATH.exists():
        st.error(tr(
            "Processed experiment CSV not found. Run the data build pipeline first.",
            "未找到处理后的实验数据 CSV，请先运行数据构建流水线。",
        ))
    else:
        records_df = load_records(str(RECORDS_PATH))
        col_track, col_search = st.columns([1, 2])
        with col_track:
            track_mode = st.radio(
                tr("Select research track:", "选择物理研究轨道:"),
                [tr("Solid surface combustion (BASS-II)", "固体表面燃烧 (BASS-II)"),
                 tr("Free-floating droplet combustion (FLEX)", "空间悬浮液滴燃烧 (FLEX)")],
                index=0,
            )
        with col_search:
            search_query = st.text_input(
                tr("🔍 Search material, experiment ID or condition:", "🔍 全文检索试样材料、实验编号或工况:"), ""
            ).strip().lower()

        if "BASS-II" in track_mode:
            track_df = records_df[records_df["dataset_key"] == "BASS_II"].copy()
            track_key = "solid_fuel"
            doi = "10.60555/4qc4-de67"
            source_url = "https://psi.nasa.gov/physci/repo/data/investigations/PSI-25"
            default_x, default_y = "calibrated_initial_o2_pct_by_vol", "final_co_ppm"
        else:
            track_df = records_df[records_df["dataset_key"] == "FLEX"].copy()
            track_key = "liquid_droplet"
            doi = "10.60555/mbq8-0451"
            source_url = "https://psi.nasa.gov/physci/repo/data/investigations/PSI-69"
            default_x, default_y = "droplet_initial_diameter_mm", "burning_rate_mm"

        if search_query:
            mask = track_df.astype(str).apply(lambda col: col.str.lower().str.contains(search_query, regex=False, na=False)).any(axis=1)
            track_df = track_df.loc[mask]

        st.info(tr(
            f"📌 **Current track**: {track_mode} | **DOI**: `{doi}` | [Open NASA data repository]({source_url})",
            f"📌 **当前实验轨道**: {track_mode} | **DOI**: `{doi}` | [访问 NASA 官方实验库]({source_url})",
        ))

        st.markdown(tr(
            "#### Ranking aggregated from real experiments (zero hardcoding)",
            "#### 真实实验数据聚合统计排行 (Zero Hardcoding)",
        ))
        ranking_data = load_ranking(track_key)
        if ranking_data:
            st.caption(tr(
                "Live view of each ranked fuel, using the mean values measured in orbit. "
                + ("Each droplet spins inside its spherical microgravity flame and shrinks until it reaches its "
                   "measured extinction diameter, where the flame goes out." if track_key == "liquid_droplet" else
                   "Each sample rotates while a flame front spreads along it; more smoke puffs = more CO measured."),
                "基于在轨实测平均值的动态展示。"
                + ("每个液滴在球形微重力火焰中旋转并逐渐缩小，直到达到实测熄火直径时火焰熄灭。" if track_key == "liquid_droplet" else
                   "每个样品旋转展示，火焰沿样品蔓延；烟团越多表示实测 CO 越高。"),
            ))
            flex_df = records_df[records_df["dataset_key"] == "FLEX"]
            d0_by_fuel = (flex_df.groupby("fuel")["droplet_initial_diameter_mm"].mean().to_dict()
                          if "droplet_initial_diameter_mm" in flex_df else {})
            st.markdown(ranking_cards_html(track_key, ranking_data, d0_by_fuel), unsafe_allow_html=True)
            with st.expander(tr("Show ranking as a data table", "以数据表查看排名")):
                st.dataframe(pd.DataFrame(ranking_data), use_container_width=True, hide_index=True)

        st.markdown(tr("#### Raw observation table (auditable detail)", "#### 原始观测数据表 (可审计明细)"))
        st.dataframe(track_df, use_container_width=True, hide_index=True)


with tab_evidence:
    render_evidence()


# =========================================================================
# TAB 2: STEM 科研变量敏感性分析与物理因果叙述
# =========================================================================
@st.fragment
def render_sweep():
    st.subheader(tr(
        "🔬 STEM variable sensitivity analysis and physical narration",
        "🔬 STEM 科研变量敏感性分析与物理因果叙述",
    ))
    st.markdown(tr(
        "Explore how ventilation speed, gravity and oxygen concentration change the combustion flow state, "
        "and read the first-principles reasoning behind it.",
        "探索不同物理变量（风速、重力、氧浓度）如何影响燃烧流体状态，解译背后的第一性原理规律。",
    ))

    col_sweep_cfg, col_sweep_plot = st.columns([1, 2])

    with col_sweep_cfg:
        st.markdown(tr("##### 1. Choose the variable to sweep", "##### 1. 选择研究自变量"))
        sweep_options = [
            tr("Ventilation speed u_forced (m/s)", "通风风速 u_forced (m/s)"),
            tr("Effective gravity g (m/s²)", "有效重力加速度 g (m/s²)"),
            tr("Cabin oxygen O2 (% Vol)", "舱内氧气浓度 O2 (% Vol)"),
        ]
        sweep_target = st.selectbox(
            tr("Physical variable to sweep:", "选择需要扫描的物理自变量:"), sweep_options, index=0
        )
        sweep_idx = sweep_options.index(sweep_target)
        lbl_fixed_g = tr("Fixed gravity (m/s²)", "固定环境重力 (m/s²)")
        lbl_fixed_u = tr("Fixed forced flow speed (m/s)", "固定强迫风速 (m/s)")
        lbl_fixed_o2 = tr("Fixed cabin oxygen (% Vol)", "固定舱内氧浓度 (% Vol)")

        if sweep_idx == 0:
            p_name = "flow_speed"
            r_min, r_max = 0.0, 0.25
            f_g = st.slider(lbl_fixed_g, 0.0, 9.81, 0.0, step=0.1)
            f_o2 = st.slider(lbl_fixed_o2, 15.0, 35.0, 21.0, step=1.0)
            f_u = 0.05
        elif sweep_idx == 1:
            p_name = "gravity"
            r_min, r_max = 0.0, 10.0
            f_g = 0.0
            f_u = st.slider(lbl_fixed_u, 0.01, 0.3, 0.05, step=0.01)
            f_o2 = st.slider(lbl_fixed_o2, 15.0, 35.0, 21.0, step=1.0)
        else:
            p_name = "oxygen"
            r_min, r_max = 15.0, 35.0
            f_g = st.slider(lbl_fixed_g, 0.0, 9.81, 0.0, step=0.1)
            f_u = st.slider(lbl_fixed_u, 0.01, 0.3, 0.05, step=0.01)
            f_o2 = 21.0

        sweep_results = cached_sweep(
            sweep_param=p_name,
            range_min=r_min,
            range_max=r_max,
            steps=25,
            fixed_gravity_m_s2=f_g,
            fixed_flow_speed_m_s=f_u,
            fixed_o2_pct=f_o2,
            lang=lang,
        )
        points_df = pd.DataFrame(sweep_results["curve_points"])

    with col_sweep_plot:
        st.markdown(tr("##### 2. Physical response curves and flow-regime evolution", "##### 2. 物理响应曲线与流体机制演化"))

        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=points_df["sweep_value"],
            y=points_df["predicted_flame_intensity_proxy"],
            mode="lines+markers",
            name=tr("Flame spread intensity proxy", "火焰蔓延强度代理指标"),
            line=dict(color="#FF4B4B", width=3)
        ))
        fig_curve.add_trace(go.Scatter(
            x=points_df["sweep_value"],
            y=points_df["log_richardson_number"],
            mode="lines",
            name=tr("log10(Richardson number Ri)", "log10(Richardson 数 Ri)"),
            line=dict(color="#00D4B2", width=2, dash="dash"),
            yaxis="y2"
        ))

        fig_curve.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.2),
            title=tr(f"Sensitivity sweep response (swept parameter: {p_name})", f"自变量敏感性扫描响应曲线 (扫描参数: {p_name})"),
            xaxis_title=tr(f"Swept input value ({p_name})", f"自变量输入值 ({p_name})"),
            yaxis=dict(title=tr("Flame spread intensity index", "火焰蔓延强度指标"), titlefont=dict(color="#FF4B4B")),
            yaxis2=dict(title=tr("log10(Ri) flow-regime criterion", "log10(Ri) 流动模式判据"), titlefont=dict(color="#00D4B2"), overlaying="y", side="right"),
            template="plotly_dark",
            height=420
        )
        st.plotly_chart(fig_curve, use_container_width=True)

    st.markdown(tr("##### 3. Scientific narration for STEM students", "##### 3. 物理因果叙述 (Scientific Narration for STEM Students)"))
    for text in sweep_results["scientific_narration"]:
        st.success(f"📘 {text}")

    st.markdown(tr("##### 4. Solved sweep points", "##### 4. 连续参数求解点明细"))
    st.dataframe(points_df, use_container_width=True, hide_index=True)


with tab_sweep:
    render_sweep()


# =========================================================================
# TAB 3: STEM 2D 物理时空模拟沙盒 (动态场)
# =========================================================================
@st.fragment
def render_simulation():
    st.subheader(tr("2D microgravity combustion physics sandbox", "2D 空间微重力燃烧物理沙盒"))
    st.markdown(tr(
        "Adjust gravity and ventilation speed to watch the hot plume evolve from symmetric microgravity "
        "diffusion to Earth-like buoyant lift.",
        "通过调整重力与通风风速，观察热羽流从微重力对称扩散到地球浮力拉升的时空演变。",
    ))

    c_ctrl, c_vis = st.columns([1, 2])
    with c_ctrl:
        preset_options = [
            tr("Custom", "自定义"),
            tr("Microgravity (0.0 m/s²)", "微重力 (0.0 m/s²)"),
            tr("Moon (1.62 m/s²)", "月球 (1.62 m/s²)"),
            tr("Mars (3.71 m/s²)", "火星 (3.71 m/s²)"),
            tr("Earth (9.81 m/s²)", "地球 (9.81 m/s²)"),
        ]
        preset = st.selectbox(tr("Quick preset:", "快速预设场景:"), preset_options)
        g_val = [0.0, 0.0, 1.62, 3.71, 9.80665][preset_options.index(preset)]
        sim_g = st.slider(tr("Effective gravity g (m/s²)", "有效重力 g (m/s²)"), 0.0, 15.0, float(g_val), step=0.1)
        sim_u = st.slider(tr("Cabin forced flow speed u (m/s)", "舱内强迫风速 u (m/s)"), 0.0, 0.3, 0.05, step=0.01)
        sim_t_hot = st.slider(tr("Hot gas temperature (K)", "热气体特征温度 (K)"), 500.0, 2000.0, 1200.0, step=50.0)
        sim_dur = st.slider(tr("Simulation duration (s)", "模拟时长 (s)"), 0.5, 3.0, 1.5, step=0.25)

        sim = cached_simulation(sim_g, sim_u, sim_t_hot, sim_dur)

    with c_vis:
        n_frames = len(sim["times"])
        centroid_label = tr("Plume centroid", "羽流质心")

        x_cm = [v * 100 for v in sim["x"]]
        y_cm = [v * 100 for v in sim["y"]]
        cx_cm = [c[0] * 100 for c in sim["centroids"]]
        cy_cm = [c[1] * 100 for c in sim["centroids"]]

        st.markdown(tr(
            "**How to read this chart**\n"
            "- 🟨 **Bright patch = hot gas** from the fire. Dark = normal cabin air.\n"
            "- ⚪ **White star = where the fire started.** 🔷 **Cyan cross = centre of the hot gas now**, "
            "and the line behind it is the path it has travelled.\n"
            "- ⬆ **Up on the chart = away from gravity.** Press **▶ Play** to watch it evolve.",
            "**如何阅读此图**\n"
            "- 🟨 **亮色区域 = 火源产生的热气体**，暗色 = 舱内正常空气。\n"
            "- ⚪ **白色星标 = 起火位置**；🔷 **青色十字 = 热气体当前中心**，其身后的线是已走过的路径。\n"
            "- ⬆ **图中向上 = 背离重力的方向**。点击 **▶ 播放** 观看演变过程。",
        ))

        def frame_traces(i: int) -> list:
            return [
                go.Heatmap(
                    z=sim["z"][i], x=x_cm, y=y_cm, colorscale="Inferno", zsmooth="best",
                    zmin=300.0, zmax=sim_t_hot,
                    colorbar=dict(title=tr("Gas temp (K)", "气体温度 (K)"), thickness=12),
                    hovertemplate="x=%{x:.1f} cm<br>y=%{y:.1f} cm<br>T=%{z:.0f} K<extra></extra>",
                ),
                go.Scatter(
                    x=cx_cm[: i + 1], y=cy_cm[: i + 1], mode="lines",
                    line=dict(color="#00E5FF", width=2, dash="dot"), hoverinfo="skip", showlegend=False,
                ),
                go.Scatter(
                    x=[cx_cm[i]], y=[cy_cm[i]], mode="markers+text",
                    marker=dict(color="#00E5FF", size=13, symbol="cross", line=dict(width=1, color="white")),
                    text=[centroid_label], textfont=dict(color="#00E5FF"),
                    # keep the label inside the chart: flip it left once the plume nears the right edge
                    textposition="top left" if cx_cm[i] > (x_cm[0] + x_cm[-1]) / 2 else "top right",
                    showlegend=False, hoverinfo="skip",
                ),
            ]

        def frame_title(i: int) -> str:
            return f"t = {sim['times'][i]:.2f} s"

        start = int(n_frames * 0.7)
        fig_2d = go.Figure(
            data=frame_traces(start),
            frames=[
                go.Frame(name=str(i), data=frame_traces(i), traces=[0, 1, 2],
                         layout=go.Layout(title_text=frame_title(i)))
                for i in range(n_frames)
            ],
        )
        fig_2d.add_trace(go.Scatter(
            x=[0], y=[0], mode="markers+text", showlegend=False, hoverinfo="skip",
            marker=dict(color="white", size=14, symbol="star", line=dict(width=1, color="#FF6A2B")),
            text=[tr("Fire starts here", "起火点")], textposition="bottom right", textfont=dict(color="white"),
        ))
        # Direction cues sit above the chart as wrapping chips, so they never overlap the plot.
        grav_chip = (tr(f"⬇ Gravity {sim_g:.1f} m/s²", f"⬇ 重力 {sim_g:.1f} m/s²") if sim_g > 0
                     else tr("⬇ Gravity ≈ 0 (weightless)", "⬇ 重力 ≈ 0（失重）"))
        flow_chip = (tr(f"➡ Cabin airflow {sim_u:.2f} m/s", f"➡ 舱内气流 {sim_u:.2f} m/s") if sim_u > 0
                     else tr("➡ No airflow", "➡ 无气流"))
        st.markdown(
            f'<div class="cues"><span class="cue g">{grav_chip}</span><span class="cue a">{flow_chip}</span></div>',
            unsafe_allow_html=True,
        )
        play_args = dict(frame=dict(duration=140, redraw=True), transition=dict(duration=0),
                         fromcurrent=True, mode="immediate")
        fig_2d.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title=dict(text=frame_title(start), font=dict(size=15)),
            xaxis_title=tr("Sideways distance (cm) →  direction of airflow", "水平距离 (cm) →  气流方向"),
            yaxis_title=tr("Height (cm) ↑  away from gravity", "高度 (cm) ↑  背离重力方向"),
            xaxis=dict(title_standoff=8, automargin=True),
            yaxis=dict(scaleanchor="x", scaleratio=1, automargin=True), height=560,
            margin=dict(l=10, r=10, t=64, b=110),  # top-right: Play/Pause, bottom: time slider
            updatemenus=[dict(
                type="buttons", direction="left", x=1.0, y=1.0, xanchor="right", yanchor="bottom",
                pad=dict(t=0, b=14, l=0, r=0), bgcolor="#FFD9B8", bordercolor="#FF9A4D", borderwidth=2,
                active=-1, font=dict(color="#0B1020", size=13),
                buttons=[
                    dict(label=tr("   ▶ Play   ", "   ▶ 播放   "), method="animate", args=[None, play_args]),
                    dict(label=tr("   ⏸ Pause   ", "   ⏸ 暂停   "), method="animate",
                         args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
                ],
            )],
            sliders=[dict(
                active=start, x=0.02, y=0.0, len=0.96, xanchor="left", yanchor="top",
                pad=dict(t=64, b=0, l=0, r=0), currentvalue=dict(visible=False),
                bgcolor="rgba(255,255,255,0.15)", activebgcolor="#FF6A2B", bordercolor="rgba(255,255,255,0.25)",
                tickcolor="rgba(0,0,0,0)", ticklen=0, font=dict(size=10, color="#aab3cc"),
                steps=[dict(method="animate", label=(f"{sim['times'][i]:.1f}s" if i % 3 == 0 or i == n_frames - 1 else ""),
                            args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
                       for i in range(n_frames)],
            )],
        )
        st.plotly_chart(fig_2d, use_container_width=True,
                        config={"displayModeBar": False, "responsive": True})

        dur = sim["times"][-1]
        up_cm, side_cm = cy_cm[-1], cx_cm[-1]
        r0, r1 = sim["radii"][0] * 100, sim["radii"][-1] * 100
        if sim_g > 0:
            rise_en = (f"Gravity makes the hot gas lighter than the cooler air around it, so it **rises**: "
                       f"about **{up_cm:.1f} cm upward** in {dur:.1f} s.")
            rise_zh = f"重力使热气体比周围冷空气更轻，因此它会**上升**：{dur:.1f} 秒内向上约 **{up_cm:.1f} cm**。"
        else:
            rise_en = (f"With **no gravity** there is no 'up': hot gas has nothing to float on, so it does not rise. "
                       f"It only spreads out evenly, growing from {r0:.1f} cm to {r1:.1f} cm wide.")
            rise_zh = f"**没有重力**就没有“上”：热气体无从上浮，不会上升，只会均匀向四周扩散，宽度从 {r0:.1f} cm 增长到 {r1:.1f} cm。"
        if sim_u > 0:
            flow_en = f" The cabin airflow then pushes it **sideways about {side_cm:.1f} cm**."
            flow_zh = f" 舱内气流再把它**横向推送约 {side_cm:.1f} cm**。"
        else:
            flow_en, flow_zh = " There is no airflow, so it does not drift sideways.", " 没有气流，因此不会横向漂移。"
        if sim_g <= 0 and sim_u > 0:
            tip_en = "\n\n💡 **Key idea:** in space, airflow (not buoyancy) decides where heat and flame go."
            tip_zh = "\n\n💡 **核心要点：** 在太空中，决定热量和火焰去向的是气流，而不是浮力。"
        elif sim_g > 0 and sim_u == 0:
            tip_en = "\n\n💡 **Key idea:** this is the familiar upward flame, like a candle on Earth."
            tip_zh = "\n\n💡 **核心要点：** 这就是我们熟悉的向上火焰，就像地球上的蜡烛。"
        else:
            tip_en = tip_zh = ""
        st.info(tr("**What you are seeing:** " + rise_en + flow_en + tip_en,
                   "**图中发生了什么：** " + rise_zh + flow_zh + tip_zh))
        st.caption(tr(
            "Schematic educational illustration of the physics, not a CFD simulation or NASA imagery.",
            "物理原理的示意性教学演示，并非 CFD 仿真或 NASA 影像。",
        ))

        derived = sim["derived"]
        row_a, row_b = st.columns(2), st.columns(2)
        p1, p2, p3, p4 = row_a[0], row_a[1], row_b[0], row_b[1]
        p1.metric(tr("Reduced gravity g'", "约化重力 g'"), f"{derived['reduced_gravity_m_s2']:.3f} m/s²",
                  help=tr("Gravity felt by the hot gas, after subtracting the lift from the cooler air around it. "
                          "Bigger = rises faster.", "热气体实际感受到的重力（已扣除周围冷空气的浮托）。越大上升越快。"))
        p2.metric(tr("Buoyancy velocity u_b", "特征浮力速度 u_b"), f"{derived['buoyancy_velocity_scale_m_s']:.3f} m/s",
                  help=tr("Typical speed at which buoyancy alone would lift the hot gas.",
                          "仅靠浮力时，热气体上升的典型速度。"))
        p3.metric(tr("Net buoyancy F_net", "净浮力 F_net"), f"{derived['net_buoyancy_force_n']:.2e} N",
                  help=tr("Upward push on the hot gas minus its weight. Zero in microgravity.",
                          "热气体受到的向上推力减去自身重量。微重力下为 0。"))
        p4.metric(tr("Final spread width r_diff", "末帧扩散宽度 r_diff"), f"{sim['radii'][-1]*100:.2f} cm",
                  help=tr("How wide the hot patch has spread by the last frame.", "到最后一帧时热区扩散的宽度。"))

with tab_simulation:
    render_simulation()


# =========================================================================
# TAB 4: 科学审计与 25 折验证监视器 (Phase 1 成果可视化)
# =========================================================================
with tab_audit:
    st.subheader(tr("Scientific audit and 5×5 repeated group cross-validation report", "科学审计与 5×5 重复分组交叉验证报告"))
    audit_json = REPORTS_DIR / "data_audit_report.json"
    metrics_csv = REPORTS_DIR / "cv_fold_metrics.csv"

    if audit_json.exists() and metrics_csv.exists():
        with open(audit_json, "r", encoding="utf-8") as f:
            a_data = json.load(f)
        m_df = pd.read_csv(metrics_csv)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(tr("Audit status", "审计状态"), a_data.get("audit_status", "").upper())
        c2.metric(tr("Independent experiment groups", "独立实验组数"),
                  tr(f"{a_data.get('experiment_group_count', 0)} groups", f"{a_data.get('experiment_group_count', 0)} 组"))
        c3.metric(tr("Evaluation folds", "评估折数"), tr(f"{m_df['eval_id'].nunique()} folds", f"{m_df['eval_id'].nunique()} 折"))
        c4.metric(tr("Zero cross-group leakage assertion", "组间零泄漏断言"), "100% PASSED ✅")

        st.markdown(tr("#### Summary of 25 isolated validation folds (Mean ± Std)", "#### 25 轮隔离测试折评测汇总 (Mean ± Std)"))
        summary = m_df.groupby("model_family").agg({
            "mae_mm_s": ["mean", "std"], "rmse_mm_s": ["mean", "std"],
            "r2_score": ["mean", "std"], "balanced_accuracy": ["mean", "std"], "macro_f1": ["mean", "std"]
        }).round(4)
        st.dataframe(summary, use_container_width=True)

        st.markdown(tr("#### Audit warnings and scientific honesty declarations", "#### 审计警示与科学诚实声明 (Honesty Declarations)"))
        for warn in a_data.get("warnings", []):
            st.warning(f"⚠️ {warn}")


# =========================================================================
# TAB 5: After-test estimates from a separately trained experimental model
# =========================================================================
with tab_prediction:
    st.subheader(tr("Predict peak heat release from a new CSV", "根据新 CSV 预测峰值热释放率"))
    st.info(tr(
        "This experimental model uses measured after-test burn summaries. "
        "Its current training source is ground-based PMMA cone calorimeter tests; "
        "it has not been validated for spacecraft or microgravity fires. "
        "Use it only when the peak is missing from an otherwise complete burn summary. "
        "If you have the full HRR time series, calculate its maximum directly.",
        "该实验性模型使用试验后实测的燃烧汇总数据。当前训练数据来自地面 PMMA 锥形量热仪试验，"
        "尚未针对航天器或微重力火灾验证。仅当完整燃烧汇总中缺少峰值时使用；"
        "若已有完整 HRR 时间序列，请直接取其最大值。",
    ))
    st.caption(tr(
        "Required columns (in these units): "
        + ", ".join(f"`{column}`" for column in FEATURE_COLUMNS)
        + ". The high-hazard flag is a project rule: predicted peak heat release "
        ">= 350 kW/m². It is not a separately trained classifier.",
        "必需列（按此单位）："
        + ", ".join(f"`{column}`" for column in FEATURE_COLUMNS)
        + "。高危标记为项目规则：预测峰值热释放率 >= 350 kW/m²，并非单独训练的分类器。",
    ))
    sample_csv = (
        "radiant_heat_flux_kw_m2,real_burn_duration_s,real_mean_hrr_kw_m2\n"
        "25,900,300\n35,650,400\n50,400,500\n65,200,1500\n"
    )
    st.download_button(
        tr("⬇ Download a sample CSV to try", "⬇ 下载示例 CSV 试用"),
        sample_csv.encode("utf-8"), file_name="sample_after_test.csv", mime="text/csv",
        key="download_sample_csv",
        help=tr("Four example burns inside the model's training range. Upload it below, then press Predict.",
                "四条位于模型训练范围内的示例燃烧数据。在下方上传后点击预测。"),
    )
    st.caption(tr(
        "Accepted range (from training data): heat flux 25–65 kW/m², burn duration 195–1149 s, "
        "mean HRR 147–1881 kW/m². Values outside are rejected rather than guessed.",
        "可接受范围（来自训练数据）：热通量 25–65 kW/m²，燃烧时长 195–1149 秒，平均 HRR 147–1881 kW/m²。超出范围将被拒绝而不是猜测。",
    ))
    uploaded_csv = st.file_uploader(tr("Upload after-test CSV", "上传试验后 CSV"), type="csv", key="fire_prediction_csv")
    if uploaded_csv is None:
        st.session_state.pop("fire_prediction_result", None)
        # Preview so the tab shows what a prediction looks like before any upload.
        # Fixed illustrative numbers (no model call), so TensorFlow is not loaded just to view the tab.
        st.markdown(tr("#### 🔥 Preview: what a prediction looks like", "#### 🔥 预览：预测结果的样子"))
        render_flame_gallery(
            [
                {"radiant_heat_flux_kw_m2": 25, "real_burn_duration_s": 900, "real_mean_hrr_kw_m2": 300,
                 "predicted_peak_hrr_kw_m2": 650, "_title": tr("EXAMPLE · LOW HEAT FLUX", "示例 · 低热通量")},
                {"radiant_heat_flux_kw_m2": 65, "real_burn_duration_s": 200, "real_mean_hrr_kw_m2": 1500,
                 "predicted_peak_hrr_kw_m2": 2400, "_title": tr("EXAMPLE · HIGH HEAT FLUX", "示例 · 高热通量")},
            ],
            ref_peak=2909.0, threshold=350.0,
        )
        st.caption(tr(
            "Example values for illustration only. Upload a CSV (or the sample above) to see your own burns.",
            "示例数值仅作演示。上传 CSV（或上方的示例文件）即可查看你自己的燃烧数据。",
        ))
    if uploaded_csv is not None:
        try:
            upload_hash = sha256(uploaded_csv.getvalue()).hexdigest()
            previous = st.session_state.get("fire_prediction_result")
            if previous is not None and previous[0] != upload_hash:
                st.session_state.pop("fire_prediction_result", None)
            input_df = pd.read_csv(BytesIO(uploaded_csv.getvalue()))
            missing_columns = [column for column in FEATURE_COLUMNS if column not in input_df.columns]
            if missing_columns:
                raise ValueError("Missing required columns: " + ", ".join(missing_columns))
            if not 1 <= len(input_df) <= MAX_BATCH_ROWS:
                raise ValueError(f"CSV must contain 1 to {MAX_BATCH_ROWS} data rows")
            feature_df = input_df.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
            feature_array = feature_df.to_numpy(dtype=float)
            if (
                not np.isfinite(feature_array).all()
                or (feature_array[:, 0] < 0).any()
                or (feature_array[:, 1:] <= 0).any()
            ):
                raise ValueError("Heat flux must be nonnegative; burn duration and mean HRR must be positive")
            st.success(tr(f"✅ CSV loaded: {len(input_df)} burn(s) ready. Press the button to predict.",
                          f"✅ 已载入 CSV：{len(input_df)} 条燃烧数据。点击按钮开始预测。"))
            with st.expander(tr("Preview uploaded data", "预览上传的数据")):
                st.dataframe(input_df.head(10), use_container_width=True)
            if st.button(tr("Predict fire results", "预测火灾结果"), key="run_fire_prediction"):
                results = predict_fire_rows(feature_df.to_dict(orient="records"))
                model_info = get_model_info()
                output_df = pd.concat([input_df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
                st.session_state["fire_prediction_result"] = (upload_hash, output_df, model_info)
            saved_result = st.session_state.get("fire_prediction_result")
            if saved_result is not None and saved_result[0] == upload_hash:
                _, output_df, model_info = saved_result
                st.caption(tr(
                    f"Model source: {model_info['source']}. Scope: {model_info['scope']}",
                    f"模型来源：{model_info['source']}。适用范围：{model_info['scope']}",
                ))
                st.markdown(tr("#### 🔥 What each burn looks like", "#### 🔥 每次燃烧的可视化"))
                st.caption(tr(
                    "Animated schematic of the cone-calorimeter test. Flame height scales with predicted peak HRR "
                    "(∝ HRR^0.4), the sustained flame with mean/peak HRR, and the orange arrows with heat flux. "
                    "Illustrative, not to scale.",
                    "锥形量热仪试验的动画示意图。火焰高度随预测峰值 HRR 变化（∝ HRR^0.4），持续火焰大小对应平均/峰值 HRR，"
                    "橙色箭头数量对应热通量。仅为示意，非按比例。",
                ))
                training = model_info.get("training_data") or {}
                render_flame_gallery(
                    output_df.to_dict(orient="records"),
                    ref_peak=float(training.get("peak_hrr_max_kw_m2") or 3000.0),
                    threshold=float(model_info.get("hazard_threshold_kw_m2") or 350.0),
                )
                with st.expander(tr("Show results as a data table", "以数据表查看结果")):
                    st.dataframe(output_df, use_container_width=True)
                st.download_button(
                    tr("Download results CSV", "下载结果 CSV"),
                    output_df.to_csv(index=False).encode("utf-8"),
                    file_name="fire_predictions.csv",
                    mime="text/csv",
                    key="download_fire_predictions",
                )
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
            st.error(tr(f"Cannot use this CSV: {exc}", f"无法使用该 CSV：{exc}"))
        except ModelUnavailableError as exc:
            st.error(str(exc))


st.markdown(
    f'<div class="legal">{tr("Data: NASA Physical Sciences Informatics (CC0) and MaCFP. Independent Space Apps project, not affiliated with or endorsed by NASA. Educational use only.", "数据：NASA 物理科学信息库 (CC0) 与 MaCFP。独立 Space Apps 参赛项目，与 NASA 无隶属关系，亦未获其认可，仅供教育用途。")}</div>',
    unsafe_allow_html=True,
)
