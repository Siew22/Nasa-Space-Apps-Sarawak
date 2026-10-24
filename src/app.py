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
from plotly.subplots import make_subplots
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
from app.settings import read_json_config
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
        "equations": [dict(e) for e in res.equations],
        "status": res.scientific_status,
    }


@st.cache_data(show_spinner=False)
def load_sim_config() -> dict:
    """The solver's config, read for its bilingual assumption/limitation text."""
    return read_json_config()


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

/* Scientific audit: model picture (left) + results panel (right) */
img.model-art { width: 100%; height: auto; border-radius: 16px; background: #0f1530; border: 1px solid #2a3355; padding: 8px; }
.mpanel { display: grid; gap: 14px; }
.rings { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
.ring-wrap { text-align: center; }
.ring { width: 92px; height: 92px; border-radius: 50%; margin: 0 auto; display: grid; place-items: center; }
.ring-in { width: 72px; height: 72px; border-radius: 50%; background: #0f1426; display: grid; place-content: center; line-height: 1.1; }
.ring-in b { font-size: 1.15rem; color: #fff; }
.ring-in span { font-size: .7rem; color: #8f9ab8; }
.ring-lbl { font-size: .76rem; color: #aab3cc; margin-top: 6px; }
.eb-top { display: flex; justify-content: space-between; font-size: .82rem; color: #aab3cc; }
.eb-top b { color: #fff; }
.eb-row { height: 8px; border-radius: 99px; background: rgba(255,255,255,.06); margin-top: 5px; overflow: hidden; }
.eb-row i { display: block; height: 100%; border-radius: 99px; animation: grow 1s cubic-bezier(.2,.8,.2,1) both; transform-origin: left; }
.eb-this { background: #d95926; } .eb-base { background: #6f7ca8; }
.eb-key { font-size: .72rem; color: #8f9ab8; margin-top: 5px; display: flex; align-items: center; gap: 6px; }
.eb-key span { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
.eb-key .k1 { background: #d95926; } .eb-key .k2 { background: #6f7ca8; margin-left: 10px; }
.mnote { margin: 0; font-size: .9rem; color: #c3cbe0; line-height: 1.55; }
.mnote b { color: #FF9A4D; margin-right: 6px; }
@media (max-width: 640px) { .ring { width: 78px; height: 78px; } .ring-in { width: 60px; height: 60px; } }

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


def cfg_text(d: dict, key: str, default=""):
    """Text from a config/report dict in the UI language.

    Convention used by config/*.json: `key` holds English, `key_zh` holds Chinese.
    Falls back to English when no Chinese version exists.
    """
    if lang == "zh" and d.get(f"{key}_zh"):
        return d[f"{key}_zh"]
    return d.get(key, default)


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
@st.cache_data(show_spinner=False)
def load_provenance() -> dict:
    path = ROOT / "config" / "data_provenance.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"datasets": {}}


@st.cache_data(show_spinner=False)
def headline_numbers() -> dict:
    """Every number in the header is computed from files in the repo, never typed in."""
    out = {"datasets": 0, "records": 0, "cone_burns": 0, "folds": 0, "physics_tests": 0}
    if RECORDS_PATH.exists():
        rec = load_records(str(RECORDS_PATH))
        out["datasets"], out["records"] = int(rec["dataset_key"].nunique()), int(len(rec))
    meta_path = ROOT / "models" / "spacecraft_fire_peak_hrr_v1.metadata.json"
    if meta_path.exists():
        out["cone_burns"] = int(json.loads(meta_path.read_text(encoding="utf-8")).get("training_data", {}).get("rows", 0))
    folds_path = REPORTS_DIR / "cv_fold_metrics.csv"
    if folds_path.exists():
        out["folds"] = int(pd.read_csv(folds_path)["eval_id"].nunique())
    tests_path = ROOT / "tests" / "test_simulation_physics_limits.py"
    if tests_path.exists():
        out["physics_tests"] = tests_path.read_text(encoding="utf-8").count("\ndef test_")
    return out


PROVENANCE = load_provenance()
PLACEHOLDER_KEYS = sorted(k for k, v in PROVENANCE.get("datasets", {}).items() if v.get("status") == "synthetic_placeholder")
_nums = headline_numbers()
_telemetry = [
    (str(_nums["datasets"]), tr("NASA-format datasets", "NASA 格式数据集")
     + (tr(" · placeholder data", " · 占位数据") if PLACEHOLDER_KEYS else "")),
    (str(_nums["cone_burns"]), tr("Real MaCFP cone burns", "真实 MaCFP 锥形量热试验")),
    (str(_nums["folds"]), tr("CV folds (exploratory)", "交叉验证折（探索性）")),
    (str(_nums["physics_tests"]), tr("Physics sanity tests", "物理合理性测试")),
]
_briefing = [
    ("01 · " + tr("THE PROBLEM", "问题"),
     tr("In microgravity there is no buoyancy, so flames behave very differently. Too little airflow can smother "
        "a fire, while moderate ventilation can make it spread fast. Spacecraft fire safety depends on knowing which.",
        "微重力下没有浮力，火焰行为截然不同：气流太弱会使火焰窒息，适度通风却可能令其快速蔓延。航天器防火安全取决于能否分清这两种情况。")),
    ("02 · " + tr("OUR APPROACH", "我们的方法"),
     tr("We keep the solid-fuel (BASS-II) and droplet (FLEX) tracks separate and explain results with "
        "first-principles physics. Every dataset is labelled with where it came from and whether it is real.",
        "我们将固体燃料 (BASS-II) 与液滴 (FLEX) 两条轨道分开处理，并用第一性原理物理解释结果。每个数据集都标明来源以及是否为真实数据。")),
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
  <p>{tr("Physics-informed fire-safety learning built around NASA's microgravity combustion experiments, with transparent, auditable science for engineers and students.",
         "围绕 NASA 微重力燃烧实验构建的物理驱动防火安全学习平台，为工程师与学生提供透明、可审计的科学分析。")}</p>
  <div class="telemetry">{"".join(f'<div class="tele"><div class="tele-v">{v}</div><div class="tele-l">{l}</div></div>' for v, l in _telemetry)}</div>
</div>
<div class="briefing">{"".join(f'<div class="bcard"><div class="bk">{k}</div><p>{t}</p></div>' for k, t in _briefing)}</div>
""",
    unsafe_allow_html=True,
)
if PLACEHOLDER_KEYS:
    _names = ", ".join(cfg_text(PROVENANCE["datasets"][k], "display_name", k) for k in PLACEHOLDER_KEYS)
    st.warning(tr(
        f"**Data status in this build.** The {_names} tables are **synthetic placeholders** in NASA's table format, "
        "generated by `src/download_nasa_tables.py`. They are **not NASA measurements**, so the Evidence rankings and the "
        "Scientific Audit scores are demonstrations of the pipeline, not scientific results. The MaCFP cone data used by "
        "the prediction tab is real. Full details: Evidence tab → Data provenance.",
        f"**本版本数据状态。** {_names} 数据表是按 NASA 表格格式由 `src/download_nasa_tables.py` 生成的**合成占位数据**，"
        "**并非 NASA 实测数据**，因此证据排名与科学审计得分只是流程演示，不是科学结论。预测页使用的 MaCFP 锥形量热数据为真实数据。"
        "详情见：证据页 → 数据来源。",
    ), icon="⚠️")

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
    """Free-floating fuel droplet inside its spherical microgravity flame (FLEX).

    Proportions follow NASA's FLEX description: a 2-4 mm droplet inside a flame ball
    "about the size of a large olive", so the flame is drawn ~6x the droplet diameter.
    Alkanes show a luminous yellow-orange soot shell with a blue rim; methanol burns
    with a faint, soot-free blue flame. The droplet shrinks from its measured start
    diameter to the measured extinction diameter, where the flame goes out.
    Relative speeds only; time is compressed. Illustration, not footage.
    """
    is_methanol = "methanol" in fuel.lower()
    s = max(0.2, min(0.95, d_ext / d0)) if d0 > 0 else 0.5
    flame_r = 40 + 14 * (rate / rate_max if rate_max > 0 else 0.5)
    drop_r = 7.0
    if is_methanol:
        stops = ('<stop offset="0" stop-color="#3f6dff" stop-opacity="0"/>'
                 '<stop offset="0.72" stop-color="#3f6dff" stop-opacity="0.04"/>'
                 '<stop offset="0.9" stop-color="#5c8dff" stop-opacity="0.32"/>'
                 '<stop offset="0.97" stop-color="#8fb8ff" stop-opacity="0.55"/>'
                 '<stop offset="1" stop-color="#8fb8ff" stop-opacity="0"/>')
        soot = ""
        liquid_hi, liquid_lo = "#EEF6FF", "#9cc4f0"
    else:
        stops = ('<stop offset="0" stop-color="#ff9a3d" stop-opacity="0"/>'
                 '<stop offset="0.55" stop-color="#ff9a3d" stop-opacity="0.04"/>'
                 '<stop offset="0.76" stop-color="#ffb347" stop-opacity="0.55"/>'
                 '<stop offset="0.87" stop-color="#ffd27a" stop-opacity="0.9"/>'
                 '<stop offset="0.94" stop-color="#6fa0ff" stop-opacity="0.45"/>'
                 '<stop offset="1" stop-color="#6fa0ff" stop-opacity="0"/>')
        soot = (f'<circle cx="110" cy="100" r="{flame_r * 0.5:.1f}" fill="none" stroke="#7a3b12" '
                f'stroke-width="3" opacity="0.35" filter="url(#soft)"/>')
        liquid_hi, liquid_lo = "#FFF6DD", "#E2B86A"
    ext_txt = tr(f"EXTINCTION · {d_ext:.2f} mm", f"熄火 · {d_ext:.2f} mm")
    size_txt = tr(f"droplet {d0:.1f} mm · flame ~ olive-sized", f"液滴 {d0:.1f} mm · 火焰约橄榄大小")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 210" font-family="Inter, Segoe UI, Microsoft YaHei, Arial, sans-serif">
<defs>
  <radialGradient id="fl">{stops}</radialGradient>
  <radialGradient id="liq" cx="35%" cy="30%" r="75%"><stop offset="0" stop-color="{liquid_hi}"/><stop offset="1" stop-color="{liquid_lo}"/></radialGradient>
  <filter id="soft"><feGaussianBlur stdDeviation="2"/></filter>
</defs>
<style>
  .fl {{ transform-box: view-box; transform-origin: 110px 100px; animation: shell {cycle:.1f}s ease-in infinite; }}
  .br {{ transform-box: view-box; transform-origin: 110px 100px; animation: breathe 1.3s ease-in-out infinite alternate; }}
  .dr {{ transform-box: view-box; transform-origin: 110px 100px; animation: shrink {cycle:.1f}s ease-in infinite; }}
  .ext {{ opacity: 0; animation: ext {cycle:.1f}s steps(1) infinite; }}
  @keyframes shell {{ 0% {{ transform: scale(1); opacity: 1 }} 78% {{ transform: scale({0.6 + 0.4 * s:.3f}); opacity: .95 }} 84% {{ opacity: 0 }} 100% {{ opacity: 0 }} }}
  @keyframes breathe {{ from {{ transform: scale(0.98) }} to {{ transform: scale(1.025) }} }}
  @keyframes shrink {{ 0% {{ transform: scale(1) }} 80% {{ transform: scale({s:.3f}) }} 100% {{ transform: scale({s:.3f}) }} }}
  @keyframes ext {{ 0% {{ opacity: 0 }} 82% {{ opacity: 1 }} 98% {{ opacity: 0 }} }}
</style>
<g class="fl"><g class="br">
  <circle cx="110" cy="100" r="{flame_r:.1f}" fill="url(#fl)"/>
  {soot}
</g></g>
<g class="dr">
  <circle cx="110" cy="100" r="{drop_r}" fill="url(#liq)"/>
  <circle cx="108" cy="98" r="2" fill="#FFFFFF" opacity="0.8"/>
</g>
<text x="110" y="178" text-anchor="middle" fill="#8f9ab8" font-size="9.5">{size_txt}</text>
<text class="ext" x="110" y="198" text-anchor="middle" fill="#FC3D21" font-size="11" font-weight="800" letter-spacing="1">{ext_txt}</text>
</svg>"""


def solid_svg(material: str, co_mean: float, co_max_all: float) -> str:
    """Solid sample burning in the BASS-II flow duct (side view), modelled on NASA ISS photos.

    In microgravity the flame is a blue, bullet/lens-shaped envelope lying along the
    airflow with an orange glow where it is hottest; smoke is carried downstream by the
    flow (nothing rises). Rods burn at the rounded tip; thin films burn on both faces and
    vanish; the cotton-fiberglass fabric leaves its fiberglass mesh behind. Number of
    smoke wisps scales with the measured mean CO. Illustration, not footage.
    """
    name = material.lower()
    micron = next((int(t) for t in name.split() if t.isdigit()), 0)
    co_frac = co_mean / co_max_all if co_max_all > 0 else 0.5
    n_smoke = 2 + int(round(3 * co_frac))
    cy = 75.0
    flow = "".join(
        f'<line class="flow" x1="0" y1="{y}" x2="220" y2="{y}" style="animation-delay:{-0.5 * i:.1f}s"/>'
        for i, y in enumerate((36, 52, 98, 114))
    )
    defs_common = (
        '<linearGradient id="env" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#4f8dff" stop-opacity="0"/>'
        '<stop offset="0.35" stop-color="#4f8dff" stop-opacity="0.9"/><stop offset="1" stop-color="#7fb2ff" stop-opacity="0.95"/></linearGradient>'
        '<radialGradient id="glow"><stop offset="0" stop-color="#ffb070" stop-opacity="0.85"/><stop offset="1" stop-color="#ff7a3d" stop-opacity="0"/></radialGradient>'
        '<filter id="soft"><feGaussianBlur stdDeviation="1.6"/></filter>'
    )
    css_common = (
        ".flow { stroke: rgba(160,180,230,.22); stroke-width: 1; stroke-dasharray: 10 16; animation: flow 1.6s linear infinite; }"
        "@keyframes flow { to { stroke-dashoffset: -52; } }"
        ".breathe { transform-box: fill-box; transform-origin: 30% 50%; animation: breathe 1.2s ease-in-out infinite alternate; }"
        "@keyframes breathe { from { transform: scale(0.96, 0.94); } to { transform: scale(1.04, 1.06); } }"
        ".smoke { opacity: 0; transform-box: fill-box; transform-origin: center; animation: smoke 2.4s ease-out infinite; }"
        "@keyframes smoke { 0% { opacity: .45; transform: translate(0, 0) scale(.6); } 100% { opacity: 0; transform: translate(46px, 0) scale(1.8); } }"
    )

    if "rod" in name:
        # Delrin rod: white acetal rod with a rounded (hemispherical) tip, clamped at the base, burning at the tip.
        tip_x, rad = 150.0, 9.0
        smoke = "".join(
            f'<ellipse class="smoke" cx="{tip_x + 34:.0f}" cy="{cy + (k - n_smoke / 2) * 5:.1f}" rx="7" ry="4" fill="#7d8597" '
            f'style="animation-delay:{-0.6 * k:.1f}s"/>' for k in range(n_smoke)
        )
        body = f"""
  <rect x="6" y="{cy - 8}" width="18" height="16" rx="2" fill="#4a5270"/>
  <path d="M24 {cy - rad} H{tip_x} A{rad} {rad} 0 0 1 {tip_x} {cy + rad} H24 Z" fill="url(#delrin)"/>
  <path d="M{tip_x - 18} {cy - rad} H{tip_x} A{rad} {rad} 0 0 1 {tip_x} {cy + rad} H{tip_x - 18} Z" fill="#3a2a22" opacity="0.55"/>
  {smoke}
  <g class="breathe">
    <path d="M96 {cy - 22} Q {tip_x + 2} {cy - 30} {tip_x + 30} {cy} Q {tip_x + 2} {cy + 30} 96 {cy + 22}" fill="rgba(255,150,120,0.10)" stroke="url(#env)" stroke-width="3" filter="url(#soft)"/>
  </g>
  <ellipse cx="{tip_x + 3}" cy="{cy}" rx="11" ry="13" fill="url(#glow)"/>"""
        extra_defs = ('<linearGradient id="delrin" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffffff"/>'
                      '<stop offset="0.55" stop-color="#e6e7ec"/><stop offset="1" stop-color="#a9adbb"/></linearGradient>')
        extra_css = ""
    else:
        # Thin sheet held at one end; the flame front travels downstream along it, burning both faces.
        is_fabric = "fabric" in name
        thick = 4.0 if is_fabric else 1.2 + micron / 120
        x0, x1 = 20.0, 205.0
        flame_ry = 10 + thick * 1.6
        mesh = (f'<line x1="{x0}" y1="{cy}" x2="{x1}" y2="{cy}" stroke="#9aa0ad" stroke-width="{thick:.1f}" '
                f'stroke-dasharray="2 2" opacity="0.7"/>') if is_fabric else ""
        sheet_fill = "url(#weave)" if is_fabric else "#bfe0ff"
        sheet_op = "1" if is_fabric else "0.55"
        smoke = "".join(
            f'<ellipse class="smoke" cx="{x0 + 30}" cy="{cy + (k - n_smoke / 2) * 5:.1f}" rx="6" ry="3.5" fill="#7d8597" '
            f'style="animation-delay:{-0.55 * k:.1f}s"/>' for k in range(n_smoke)
        )
        body = f"""
  <rect x="6" y="{cy - 10}" width="14" height="20" rx="2" fill="#4a5270"/>
  {mesh}
  <rect class="unburnt" x="{x0}" y="{cy - thick / 2:.2f}" width="{x1 - x0}" height="{thick:.2f}" fill="{sheet_fill}" fill-opacity="{sheet_op}"/>
  <g class="front">
    {smoke}
    <g class="breathe">
      <ellipse cx="{x0 + 12}" cy="{cy}" rx="28" ry="{flame_ry:.1f}" fill="rgba(255,150,120,0.10)" stroke="url(#env)" stroke-width="2.5" filter="url(#soft)"/>
    </g>
    <ellipse cx="{x0 + 2}" cy="{cy - thick / 2 - 3:.1f}" rx="9" ry="5" fill="url(#glow)"/>
    <ellipse cx="{x0 + 2}" cy="{cy + thick / 2 + 3:.1f}" rx="9" ry="5" fill="url(#glow)"/>
  </g>"""
        extra_defs = ('<pattern id="weave" width="5" height="5" patternUnits="userSpaceOnUse"><rect width="5" height="5" fill="#d9c9a8"/>'
                      '<path d="M0 0 L5 5 M5 0 L0 5" stroke="#a8946c" stroke-width="0.8"/></pattern>')
        extra_css = (f".unburnt {{ transform-box: view-box; transform-origin: {x1}px {cy}px; animation: burn 7s linear infinite; }}"
                     "@keyframes burn { from { transform: scaleX(1); } to { transform: scaleX(0); } }"
                     ".front { animation: front 7s linear infinite; }"
                     f"@keyframes front {{ from {{ transform: translateX(0); }} to {{ transform: translateX({x1 - x0}px); }} }}")

    flow_lbl = tr("airflow →", "气流 →")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 150" font-family="Inter, Segoe UI, Microsoft YaHei, Arial, sans-serif">
<defs>{defs_common}{extra_defs}</defs>
<style>{css_common}{extra_css}</style>
<rect x="0" y="20" width="220" height="110" rx="6" fill="#070b18" stroke="#2a3355"/>
{flow}
<text x="8" y="32" fill="#8f9ab8" font-size="9">{flow_lbl}</text>
{body}
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
            f'<div><div class="rank-name">{name}</div><div class="rank-sub">{tr(f"{count} records in table", f"表中 {count} 条记录")}</div></div></div>'
            f'{obj}<div class="stats">{stats}</div></div>'
        )
    return f'<div class="rank-grid">{"".join(html)}</div>'


AUDIT_INPUTS = ("Re", "log Ri", "pO₂", "u_b", "Da")  # the 5 physics features in config/scientific_validation.json
_SVG_HEAD = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 220" font-family="Inter, Segoe UI, Microsoft YaHei, Arial, sans-serif">'
_SVG_CSS = (
    "<style>"
    ".lbl{fill:#c9d2ea;font-size:10px}.mut{fill:#8f9ab8;font-size:9px}"
    ".edge{stroke:rgba(160,180,230,.18);stroke-width:1}"
    ".pulse{stroke:#FF9A4D;stroke-width:2;stroke-dasharray:6 60;animation:pulse 1.8s linear infinite}"
    "@keyframes pulse{from{stroke-dashoffset:66}to{stroke-dashoffset:0}}"
    ".glow{animation:glow 1.8s ease-in-out infinite}"
    "@keyframes glow{0%,100%{fill:#2a3560}50%{fill:#FF9A4D}}"
    ".chip{fill:#151c3a;stroke:#3a4670}"
    "</style>"
)


def _input_chips(x: float, ys: list, dim: bool = False) -> str:
    out = []
    for y, name in zip(ys, AUDIT_INPUTS):
        op = ' opacity="0.35"' if dim else ""
        out.append(f'<g{op}><rect class="chip" x="{x}" y="{y - 9}" width="44" height="18" rx="9"/>'
                   f'<text class="lbl" x="{x + 22}" y="{y + 3.5}" text-anchor="middle">{name}</text></g>')
        if dim:
            out.append(f'<line x1="{x + 4}" y1="{y}" x2="{x + 40}" y2="{y}" stroke="#FC3D21" stroke-width="1.2" opacity="0.7"/>')
    return "".join(out)


def baseline_svg() -> str:
    """Dummy baseline: ignores every input and always predicts the training average."""
    ys = [40, 70, 100, 130, 160]
    truth = [(205, 62), (220, 140), (234, 88), (248, 158), (262, 74), (276, 120), (290, 98)]
    resid = "".join(
        f'<line x1="{x}" y1="{y}" x2="{x}" y2="110" stroke="#FC3D21" stroke-width="1.6" stroke-dasharray="3 2" '
        f'opacity="0.8"><animate attributeName="opacity" values="0.2;0.9;0.2" dur="2.4s" begin="{i * 0.25:.2f}s" repeatCount="indefinite"/></line>'
        for i, (x, y) in enumerate(truth))
    dots = "".join(f'<circle cx="{x}" cy="{y}" r="4" fill="#7CC4FF"/>' for x, y in truth)
    return (_SVG_HEAD + _SVG_CSS + _input_chips(8, ys, dim=True)
            + '<path d="M58 100 H100" stroke="#3a4670" stroke-width="1.5" stroke-dasharray="4 3"/>'
            + '<rect x="100" y="80" width="70" height="40" rx="10" fill="#151c3a" stroke="#8f9ab8"/>'
            + f'<text class="lbl" x="135" y="98" text-anchor="middle" font-weight="700">{tr("AVERAGE", "平均值")}</text>'
            + f'<text class="mut" x="135" y="111" text-anchor="middle">{tr("same answer", "答案不变")}</text>'
            + '<path d="M170 110 H300" stroke="#FF9A4D" stroke-width="2.2" stroke-dasharray="6 4"/>'
            + resid + dots
            + f'<text class="mut" x="8" y="200">{tr("Inputs ignored ✕", "忽略全部输入 ✕")}</text>'
            + f'<text class="mut" x="196" y="200">{tr("● real   — guess   ┆ error", "● 真实  — 猜测  ┆ 误差")}</text>'
            + "</svg>")


def forest_svg() -> str:
    """Random forest: many decision trees answer yes/no questions, then their answers are averaged."""
    ys = [40, 70, 100, 130, 160]
    parts = [_SVG_HEAD, _SVG_CSS, _input_chips(8, ys)]
    leaf_pts = []
    for t, cx in enumerate((110, 178, 246)):
        root, kids = (cx, 40), [(cx - 16, 74), (cx + 16, 74)]
        leaves = [(cx - 24, 108), (cx - 8, 108), (cx + 8, 108), (cx + 24, 108)]
        for k, (kx, ky) in enumerate(kids):
            parts.append(f'<line class="edge" x1="{root[0]}" y1="{root[1]}" x2="{kx}" y2="{ky}"/>')
            for lx, ly in leaves[2 * k:2 * k + 2]:
                parts.append(f'<line class="edge" x1="{kx}" y1="{ky}" x2="{lx}" y2="{ly}"/>')
        # the decision path this sample takes through tree t (lights up in turn)
        kid = kids[t % 2]
        leaf = leaves[(t % 2) * 2 + (1 if t == 1 else 0)]
        parts.append(f'<polyline points="{root[0]},{root[1]} {kid[0]},{kid[1]} {leaf[0]},{leaf[1]}" fill="none" '
                     f'stroke="#FF9A4D" stroke-width="2.4" stroke-linecap="round" opacity="0">'
                     f'<animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.2;0.7;1" dur="3s" begin="{t * 0.6:.1f}s" repeatCount="indefinite"/></polyline>')
        for (nx, ny) in [root, *kids, *leaves]:
            parts.append(f'<circle cx="{nx}" cy="{ny}" r="4.5" fill="#2a3560" stroke="#6f7ca8"/>')
        leaf_pts.append(leaf)
    for lx, ly in leaf_pts:
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="178" y2="150" stroke="rgba(255,154,77,.45)" stroke-width="1.2"/>')
    parts.append(f'<text class="mut" x="178" y="20" text-anchor="middle">{tr("3 of the 100 trees shown", "显示 100 棵树中的 3 棵")}</text>')
    parts.append('<circle class="glow" cx="178" cy="150" r="10" stroke="#FF9A4D"/>')
    parts.append(f'<text class="lbl" x="194" y="154">{tr("average answers", "汇总答案")}</text>')
    parts.append('<path d="M178 160 L130 186 M178 160 L226 186" stroke="#FF9A4D" stroke-width="1.5"/>')
    parts.append(f'<rect class="chip" x="88" y="186" width="84" height="20" rx="10"/><text class="lbl" x="130" y="200" text-anchor="middle">{tr("burn rate", "燃烧速率")}</text>')
    parts.append(f'<rect class="chip" x="184" y="186" width="84" height="20" rx="10"/><text class="lbl" x="226" y="200" text-anchor="middle">{tr("flame out?", "是否熄火？")}</text>')
    parts.append("</svg>")
    return "".join(parts)


def network_svg() -> str:
    """Multitask neural network: 5 inputs -> 32 -> 16 shared neurons -> two output heads."""
    layers_x = (66, 140, 205, 262)
    ys_in = [40, 70, 100, 130, 160]
    ys_h1 = [30 + i * 20 for i in range(8)]   # 8 drawn of 32
    ys_h2 = [45 + i * 22 for i in range(6)]   # 6 drawn of 16
    ys_out = [78, 132]
    parts = [_SVG_HEAD, _SVG_CSS]
    cols = [(layers_x[0], ys_in), (layers_x[1], ys_h1), (layers_x[2], ys_h2), (layers_x[3], ys_out)]
    for (xa, ya), (xb, yb) in zip(cols, cols[1:]):
        for y1 in ya:
            for y2 in yb:
                parts.append(f'<line class="edge" x1="{xa}" y1="{y1}" x2="{xb}" y2="{y2}"/>')
    # a few signals travelling forward through the network
    pulse_edges = [(0, 0, 2), (0, 2, 5), (0, 4, 7), (1, 1, 1), (1, 4, 3), (1, 6, 5), (2, 0, 0), (2, 3, 1), (2, 5, 1), (2, 2, 0)]
    for i, (layer, a, b) in enumerate(pulse_edges):
        (xa, ya), (xb, yb) = cols[layer], cols[layer + 1]
        parts.append(f'<line class="pulse" x1="{xa}" y1="{ya[a]}" x2="{xb}" y2="{yb[b]}" style="animation-delay:{-0.17 * i:.2f}s"/>')
    for layer, (x, ys) in enumerate(cols):
        for y in ys:
            delay = f' style="animation-delay:{0.45 * layer:.2f}s"' if layer else ""
            parts.append(f'<circle class="glow" cx="{x}" cy="{y}" r="6" stroke="#6f7ca8"{delay}/>')
    for y, name in zip(ys_in, AUDIT_INPUTS):
        parts.append(f'<text class="lbl" x="{layers_x[0] - 10}" y="{y + 3.5}" text-anchor="end">{name}</text>')
    parts.append(f'<text class="mut" x="{layers_x[1]}" y="200" text-anchor="middle">{tr("32 neurons", "32 个神经元")}</text>')
    parts.append(f'<text class="mut" x="{layers_x[2]}" y="200" text-anchor="middle">{tr("16 neurons", "16 个神经元")}</text>')
    parts.append(f'<text class="lbl" x="{layers_x[3]}" y="{ys_out[0] - 12}" text-anchor="middle">{tr("burn rate", "燃烧速率")}</text>')
    parts.append(f'<text class="lbl" x="{layers_x[3]}" y="{ys_out[1] + 20}" text-anchor="middle">{tr("flame out?", "是否熄火？")}</text>')
    parts.append(f'<text class="mut" x="{(layers_x[1] + layers_x[2]) / 2}" y="16" text-anchor="middle">{tr("shared layers", "共享层")}</text>')
    parts.append("</svg>")
    return "".join(parts)


def model_panel_html(stats: dict, base: dict, how: str) -> str:
    """Right-hand panel: ring gauges, error-vs-baseline bars, and plain-language notes."""
    def ring(label: str, mean: float, std: float) -> str:
        pct = max(0.0, min(1.0, mean)) * 100
        return (f'<div class="ring-wrap"><div class="ring" style="background:conic-gradient(#d95926 {pct:.1f}%, rgba(255,255,255,.08) 0)">'
                f'<div class="ring-in"><b>{mean:.2f}</b><span>± {std:.2f}</span></div></div><div class="ring-lbl">{label}</div></div>')

    def err_bar(label: str, key: str) -> str:
        m, b = stats[key]["mean"], base[key]["mean"]
        scale = max(m, b)
        return (f'<div class="eb"><div class="eb-top"><span>{label}</span><b>{m:.3f} ± {stats[key]["std"]:.3f} mm/s</b></div>'
                f'<div class="eb-row"><i class="eb-this" style="width:{100 * m / scale:.0f}%"></i></div>'
                f'<div class="eb-row"><i class="eb-base" style="width:{100 * b / scale:.0f}%"></i></div>'
                f'<div class="eb-key"><span class="k1"></span>{tr("this model", "本模型")}<span class="k2"></span>{tr("baseline", "基线")}</div></div>')

    is_base = stats is base
    mae_drop = (1 - stats["mae_mm_s"]["mean"] / base["mae_mm_s"]["mean"]) * 100
    did = (tr("This is the yardstick: R² ≈ 0 means no skill beyond guessing the average.",
              "这是参照标准：R² ≈ 0 表示除了猜平均值外没有任何预测能力。") if is_base else
           tr(f"Across 25 held-out folds it explains about {stats['r2_score']['mean'] * 100:.0f}% of the variation in burn rate, "
              f"with {mae_drop:.0f}% lower error than the baseline.",
              f"在 25 个独立测试折中，它解释了约 {stats['r2_score']['mean'] * 100:.0f}% 的燃烧速率变化，误差比基线低 {mae_drop:.0f}%。"))
    if PLACEHOLDER_KEYS and not is_base:
        did += tr(" These scores come from placeholder data with leaked BASS-II labels, so they show the pipeline works, "
                  "not how well the model would really predict.",
                  " 这些得分来自含 BASS-II 标签泄漏的占位数据，只能说明流程可运行，不能代表模型的真实预测能力。")
    return (
        '<div class="mpanel"><div class="rings">'
        + ring(tr("R² (burn rate)", "R²（燃烧速率）"), stats["r2_score"]["mean"], stats["r2_score"]["std"])
        + ring(tr("Balanced accuracy", "平衡准确率"), stats["balanced_accuracy"]["mean"], stats["balanced_accuracy"]["std"])
        + ring(tr("Macro-F1 (flame out?)", "Macro-F1（是否熄火）"), stats["macro_f1"]["mean"], stats["macro_f1"]["std"])
        + "</div>"
        + err_bar(tr("Mean absolute error", "平均绝对误差"), "mae_mm_s")
        + f'<p class="mnote"><b>{tr("How it works", "工作原理")}</b> {how}</p>'
        + f'<p class="mnote"><b>{tr("How it did", "表现如何")}</b> {did}</p></div>'
    )


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

        ds_key = "BASS_II" if track_key == "solid_fuel" else "FLEX"
        prov = PROVENANCE.get("datasets", {}).get(ds_key, {})
        is_placeholder = prov.get("status") == "synthetic_placeholder"
        st.info(tr(
            f"📌 **Current track**: {track_mode} | **Real dataset**: NASA DOI `{doi}` | [Open NASA data repository]({source_url})"
            + (" | **Data shown here: synthetic placeholder** (see Data provenance below)" if is_placeholder else ""),
            f"📌 **当前实验轨道**: {track_mode} | **真实数据集**: NASA DOI `{doi}` | [访问 NASA 官方实验库]({source_url})"
            + (" | **此处显示：合成占位数据**（见下方数据来源）" if is_placeholder else ""),
        ))

        st.markdown(tr(
            "#### Ranking computed live from the loaded table",
            "#### 由已载入数据表实时计算的排名",
        ))
        ranking_data = load_ranking(track_key)
        if ranking_data:
            st.caption(tr(
                "Animated illustration modelled on NASA flight photos (not real footage). Sizes, colours and speeds are "
                "driven by the averages computed from the loaded table"
                + (", which in this build is a synthetic placeholder, so the ranking itself is not a scientific result. "
                   if is_placeholder else ". ")
                + ("Each tiny droplet sits inside its olive-sized spherical flame and shrinks until it reaches the "
                   "table's extinction diameter, where the flame goes out." if track_key == "liquid_droplet" else
                   "Side view inside the flow duct: in microgravity the flame is a blue envelope lying along the airflow, "
                   "and smoke is carried downstream rather than rising. More smoke = more CO in the table."),
                "动画示意图，参照 NASA 在轨照片绘制（非真实影像）。大小、颜色与速度由已载入数据表计算的平均值驱动"
                + ("；本版本该表为合成占位数据，因此排名本身不是科学结论。" if is_placeholder else "。")
                + ("每个微小液滴位于橄榄大小的球形火焰中，逐渐缩小至表中的熄火直径时火焰熄灭。" if track_key == "liquid_droplet" else
                   "流道内侧视图：微重力下火焰是沿气流方向的蓝色包络，烟雾随气流向下游飘散而不会上升。烟越多表示表中 CO 越高。"),
            ))
            flex_df = records_df[records_df["dataset_key"] == "FLEX"]
            d0_by_fuel = (flex_df.groupby("fuel")["droplet_initial_diameter_mm"].mean().to_dict()
                          if "droplet_initial_diameter_mm" in flex_df else {})
            st.markdown(ranking_cards_html(track_key, ranking_data, d0_by_fuel), unsafe_allow_html=True)
            with st.expander(tr("Show ranking as a data table", "以数据表查看排名")):
                st.dataframe(pd.DataFrame(ranking_data), use_container_width=True, hide_index=True)

        st.markdown(tr("#### Data provenance", "#### 数据来源"))
        status_badge = {
            "synthetic_placeholder": tr("🟠 SYNTHETIC PLACEHOLDER: not NASA measurements", "🟠 合成占位数据：非 NASA 实测"),
            "nasa_measured": tr("🟢 NASA MEASURED DATA", "🟢 NASA 实测数据"),
        }.get(prov.get("status", ""), tr("⚪ status not declared", "⚪ 未声明状态"))
        full_track = records_df[records_df["dataset_key"] == ds_key]
        st.markdown(
            f"**{cfg_text(prov, 'display_name', ds_key)}** · {status_badge}\n\n"
            + tr(f"- **Rows loaded:** {len(full_track)} (computed from `data/processed/dashboard_records.csv`)\n",
                 f"- **已载入行数：** {len(full_track)}（由 `data/processed/dashboard_records.csv` 计算）\n")
            + tr(f"- **Produced by:** `{prov.get('generated_by', 'unknown')}`\n", f"- **生成方式：** `{prov.get('generated_by', 'unknown')}`\n")
            + tr(f"- **Real NASA source:** [{prov.get('real_source_page', '')}]({prov.get('real_source_page', '')}) · DOI `{prov.get('real_source_doi', '')}` · {cfg_text(prov, 'real_source_license')}\n",
                 f"- **真实 NASA 来源：** [{prov.get('real_source_page', '')}]({prov.get('real_source_page', '')}) · DOI `{prov.get('real_source_doi', '')}` · {cfg_text(prov, 'real_source_license')}\n")
            + "".join(f"- ⚠️ {issue}\n" for issue in cfg_text(prov, "known_issues", []))
        )

        with st.expander(tr("🔎 Record inspector: trace one row back to its source", "🔎 记录溯源：追踪单条记录的来源")):
            if full_track.empty:
                st.caption(tr("No rows for this track.", "该轨道无数据。"))
            else:
                rec_id = st.selectbox(tr("Choose a record", "选择记录"), full_track["record_id"].tolist(), key=f"inspect_{ds_key}")
                row = full_track[full_track["record_id"] == rec_id].iloc[0]
                catalog_path = ROOT / "data" / "processed" / "field_catalog.csv"
                catalog = pd.read_csv(catalog_path) if catalog_path.exists() else pd.DataFrame()
                catalog = catalog[catalog.get("dataset_key", pd.Series(dtype=str)) == ds_key] if not catalog.empty else catalog
                st.markdown(
                    tr(f"**Source file:** `{row.get('source_file', '')}` · **row** {row.get('source_row_number', '')} · "
                       f"**license field:** {row.get('source_license_status', '')}",
                       f"**来源文件：** `{row.get('source_file', '')}` · **第** {row.get('source_row_number', '')} **行** · "
                       f"**许可字段：** {row.get('source_license_status', '')}")
                )
                trace = []
                for _, f in catalog.iterrows():
                    field = f["field_name"]
                    if field in row.index and pd.notna(row[field]):
                        original = str(f["original_field_name"])
                        unit = original.split(";", 1)[1].strip() if ";" in original else ""
                        trace.append({tr("original column", "原始列名"): original, tr("value", "数值"): str(row[field]),
                                      tr("unit", "单位"): unit, tr("column in app", "应用内列名"): field})
                st.dataframe(pd.DataFrame(trace), use_container_width=True, hide_index=True)
                st.caption(tr("Processing log (src/build_dashboard_data.py): column names normalised to snake_case; values copied "
                              "unchanged; source URL, DOI and licence columns attached from config/dataset_catalog.csv. Those "
                              "columns describe the real NASA dataset, so on placeholder rows they say where real data should "
                              "come from, not where these values came from.",
                              "处理记录（src/build_dashboard_data.py）：列名统一为 snake_case；数值原样复制；来源网址、DOI 与许可列取自 "
                              "config/dataset_catalog.csv。这些列描述的是真实 NASA 数据集，因此在占位数据行上仅表示真实数据应来自何处，"
                              "而非这些数值的来源。"))

        st.markdown(tr("#### Raw table as loaded", "#### 已载入的原始数据表"))
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

        # Two aligned panels instead of one dual-axis chart: each measure keeps its own honest scale.
        # Colours validated for the dark surface (dataviz validator: lightness, chroma, CVD, contrast all pass).
        ORANGE, BLUE, INK, MUTED = "#d95926", "#3987e5", "#c9d2ea", "#8f9ab8"
        xs = points_df["sweep_value"].tolist()
        fig_curve = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, row_heights=[0.62, 0.38],
            subplot_titles=(tr("Flame spread intensity (proxy index)", "火焰蔓延强度（代理指标）"),
                            tr("Flow regime · log₁₀ Richardson number", "流动模式 · log₁₀ Richardson 数")),
        )
        for ann in fig_curve.layout.annotations:  # left-align the two panel titles
            ann.update(x=0, xanchor="left", font=dict(size=13, color=INK))

        # Shaded bands for each contiguous flow regime, labelled directly (no colour-only meaning).
        def regime_key(text: str) -> str:
            t = str(text)
            return "quench" if ("Quench" in t or "窒息" in t) else ("forced" if ("Forced" in t or "强迫" in t) else "buoyant")

        regime_style = {
            "quench": (tr("Smothered (quenching)", "窒息熄灭"), "rgba(255,255,255,0.09)"),
            "forced": (tr("Forced-flow dominated", "强迫对流主导"), "rgba(255,255,255,0.03)"),
            "buoyant": (tr("Buoyancy dominated", "浮力主导"), "rgba(255,255,255,0.06)"),
        }
        keys = [regime_key(r) for r in points_df["flow_regime"]]
        edges = [xs[0]] + [(a + b) / 2 for a, b in zip(xs, xs[1:])] + [xs[-1]]
        run_start = 0
        for i in range(1, len(keys) + 1):
            if i == len(keys) or keys[i] != keys[run_start]:
                label, fill = regime_style[keys[run_start]]
                x0, x1 = edges[run_start], edges[i]
                fig_curve.add_vrect(x0=x0, x1=x1, fillcolor=fill, line_width=0, layer="below", row="all", col=1)
                if x1 - x0 > (xs[-1] - xs[0]) * 0.12:  # wide band: label inside it
                    fig_curve.add_annotation(x=(x0 + x1) / 2, y=0.97, xref="x", yref="y domain", text=label,
                                             showarrow=False, font=dict(size=11, color=MUTED))
                else:  # narrow band: label beside it with a pointer so it is not lost
                    fig_curve.add_annotation(x=(x0 + x1) / 2, y=0.78, xref="x", yref="y domain", text=label,
                                             showarrow=True, arrowhead=2, arrowsize=0.8, arrowcolor=MUTED,
                                             ax=46, ay=-22, font=dict(size=11, color=INK))
                run_start = i

        fig_curve.add_trace(go.Scatter(
            x=xs, y=points_df["predicted_flame_intensity_proxy"], mode="lines",
            line=dict(color=ORANGE, width=2),
            fill="tozeroy", fillcolor="rgba(217,89,38,0.18)",
            name=tr("Flame spread", "火焰蔓延"), hovertemplate="%{y:.3f}<extra></extra>",
        ), row=1, col=1)
        fig_curve.add_trace(go.Scatter(
            x=xs, y=points_df["log_richardson_number"], mode="lines",
            line=dict(color=BLUE, width=2), name="log₁₀ Ri", hovertemplate="%{y:.2f}<extra></extra>",
        ), row=2, col=1)
        fig_curve.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"), row=2, col=1)
        fig_curve.add_annotation(x=1, xref="x domain", y=0, yref="y2", xanchor="right", yanchor="bottom",
                                 text=tr("Ri = 1: buoyancy ≈ airflow", "Ri = 1：浮力 ≈ 气流"),
                                 showarrow=False, font=dict(size=10, color=MUTED))
        if points_df["log_richardson_number"].max() <= -3.99:  # g = 0: Ri is 0, drawn at the solver's floor
            fig_curve.add_annotation(x=0.5, xref="x domain", y=-4, yref="y2", yanchor="bottom",
                                     text=tr("No gravity → no buoyancy (Ri ≈ 0, drawn at the chart floor)",
                                             "无重力 → 无浮力（Ri ≈ 0，绘于图表下限）"),
                                     showarrow=False, font=dict(size=11, color=INK))

        axis_style = dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False, showline=False,
                          tickfont=dict(color=MUTED, size=11), showspikes=True, spikemode="across",
                          spikecolor="rgba(255,255,255,0.35)", spikethickness=1, spikedash="solid")
        fig_curve.update_xaxes(**axis_style)
        fig_curve.update_yaxes(**axis_style)
        fig_curve.update_yaxes(rangemode="tozero", row=1, col=1)
        fig_curve.update_xaxes(title=dict(text=sweep_target, font=dict(color=INK, size=12)), row=2, col=1)
        fig_curve.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=INK), showlegend=False, height=500, margin=dict(l=10, r=10, t=40, b=10),
            hovermode="x unified", hoverlabel=dict(bgcolor="#151B30", bordercolor="#2a3355", font=dict(color="#E8ECF4")),
        )
        st.plotly_chart(fig_curve, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    st.markdown(tr("##### 3. Scientific narration for STEM students", "##### 3. 面向 STEM 学生的物理因果叙述"))
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
    with st.expander(tr("🎓 What can students observe here?", "🎓 学生可以在这里观察到什么？"), expanded=True):
        o1, o2, o3 = st.columns(3)
        o1.markdown(tr(
            "**Effective gravity g**\n\nHot gas is lighter than the cooler air around it. Gravity turns that density "
            "difference into a buoyant push (reduced gravity g′ = g·Δρ/ρ). Lower g → weaker push → the plume stops rising.",
            "**有效重力 g**\n\n热气体比周围冷空气轻，重力把这一密度差转化为浮力（约化重力 g′ = g·Δρ/ρ）。g 越小，浮力越弱，羽流不再上升。"))
        o2.markdown(tr(
            "**Forced airflow u**\n\nCabin ventilation carries heat and combustion products sideways. In microgravity this "
            "forced flow, not buoyancy, decides where the hot gas goes.",
            "**强制气流 u**\n\n舱内通风把热量与燃烧产物横向输运。在微重力下，决定热气体去向的是这股强制气流，而不是浮力。"))
        o3.markdown(tr(
            "**Gas temperature T**\n\nHotter gas means a bigger density difference, so more buoyancy on Earth. Watch how the "
            "hot patch spreads and cools over time through diffusion.",
            "**气体温度 T**\n\n温度越高，密度差越大，在地球上浮力越强。观察热区如何随时间通过扩散而扩展、降温。"))
        st.caption(tr(
            "Microgravity does not switch every force off. Buoyancy-driven natural convection almost disappears, but "
            "diffusion, pressure gradients and forced airflow still act, which is why ventilation matters so much in spacecraft.",
            "微重力并不意味着所有作用力都消失。浮力驱动的自然对流几乎消失，但扩散、压力梯度与强制气流依然存在，这正是航天器中通风如此关键的原因。"))

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
                    colorbar=dict(title=dict(text=tr("Temperature (K)<br>dark = cool<br>bright = hot",
                                                      "温度 (K)<br>暗 = 冷<br>亮 = 热"), side="top"),
                                  thickness=12),
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

    with st.expander(tr("🧮 What is actually computed (equations, assumptions, limits)", "🧮 实际计算了什么（方程、假设与局限）")):
        st.markdown(tr(
            "**Computed from physics equations every time you move a slider:** gas densities, buoyancy force, "
            "reduced gravity g′, buoyancy velocity u_b, diffusion width, and the hot-gas centre's path.\n\n"
            "**Not solved:** the temperature map itself. It is a Gaussian 'hot patch' placed at the computed centre with "
            "the computed width, an illustration, not a solution of the heat and flow (Navier-Stokes) equations, and not "
            "a flame model.",
            "**每次移动滑块都会由物理方程实时计算：** 气体密度、浮力、约化重力 g′、浮力速度 u_b、扩散宽度以及热气体中心的运动轨迹。\n\n"
            "**未求解：** 温度图本身。它是放置在计算出的中心、具有计算宽度的高斯“热区”，属于示意图，并非热量与流动（Navier-Stokes）方程的数值解，也不是火焰模型。",
        ))
        st.dataframe(
            pd.DataFrame([{tr("equation", "方程"): e.get("equation", ""), tr("what it does", "作用"): cfg_text(e, "description")}
                          for e in sim["equations"]]),
            use_container_width=True, hide_index=True,
        )
        sim_cfg = load_sim_config()
        assumptions = cfg_text(sim_cfg, "model_assumptions", [])
        limitations = cfg_text(sim_cfg, "science_notes", [])
        if assumptions:
            st.markdown(tr("**Assumptions**", "**假设**") + "\n" + "\n".join(f"- {a}" for a in assumptions))
        if limitations:
            st.markdown(tr("**Limitations**", "**局限**") + "\n" + "\n".join(f"- {a}" for a in limitations))
        st.caption(tr(f"Solver status flag: `{sim['status']}`", f"求解器状态标记：`{sim['status']}`"))

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

        # Group-leakage check recomputed here from the saved split manifest, not read from a hard-coded label.
        manifest_path = REPORTS_DIR / "experiment_split_manifest.csv"
        overlaps, n_splits = None, 0
        if manifest_path.exists():
            man = pd.read_csv(manifest_path)
            overlaps = 0
            for _, split in man.groupby("eval_id"):
                train_g = set(split.loc[split["partition"].str.contains("train"), "group_id"])
                hold_g = set(split.loc[~split["partition"].str.contains("train"), "group_id"])
                overlaps += len(train_g & hold_g)
                n_splits += 1

        eval_status = str(a_data.get("model_evaluation_status", "unknown"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(tr("Evaluation status", "评估状态"),
                  tr("Exploratory", "探索性") if "exploratory" in eval_status else eval_status,
                  help=tr(f"From reports/data_audit_report.json: {eval_status}. Exploratory means the provenance checks were "
                          "overridden, so scores must not be presented as validated.",
                          f"来自 reports/data_audit_report.json：{eval_status}。探索性表示来源检查被跳过，得分不能作为已验证结果。"))
        c2.metric(tr("Experiment groups", "实验组数"),
                  tr(f"{a_data.get('experiment_group_count', 0)} groups", f"{a_data.get('experiment_group_count', 0)} 组"))
        c3.metric(tr("Evaluation splits", "评估划分数"), tr(f"{m_df['eval_id'].nunique()} (5 seeds × 5 folds)", f"{m_df['eval_id'].nunique()}（5 种子 × 5 折）"))
        c4.metric(tr("Group overlap train ↔ test", "训练/测试组重叠"),
                  tr(f"{overlaps} in {n_splits} splits", f"{n_splits} 次划分中 {overlaps} 个") if overlaps is not None else tr("manifest missing", "缺少清单"),
                  help=tr("Recomputed live from reports/experiment_split_manifest.csv. 0 means no experiment group appears in both "
                          "training and testing. This checks the split only; it cannot detect bad labels or fake data.",
                          "由 reports/experiment_split_manifest.csv 实时重新计算。0 表示没有实验组同时出现在训练与测试中。"
                          "这只检查划分，无法发现错误标签或虚假数据。"))

        cv_issues = cfg_text(PROVENANCE.get("cross_validation_dataset", {}), "known_issues", [])
        if cv_issues:
            st.error(
                tr("**Why these scores are not scientific evidence (read before quoting any number below):**\n",
                   "**为什么以下得分不是科学证据（引用任何数字前请先阅读）：**\n")
                + "\n".join(f"- {issue}" for issue in cv_issues)
                + tr("\n\nWhat this tab does show: the evaluation pipeline works (grouped splits, no group overlap, every seed and "
                     "fold saved, uncertainty reported). Re-running it on the real NASA tables is what would turn it into evidence.",
                     "\n\n本页真正能说明的是：评估流程本身可用（分组划分、无组重叠、保存每个种子与折、报告不确定性）。"
                     "只有在真实 NASA 数据表上重新运行，才能使其成为证据。"),
                icon="🚫",
            )
            dl1, dl2 = st.columns(2)
            if manifest_path.exists():
                dl1.download_button(tr("⬇ Split manifest (every seed, fold, record)", "⬇ 划分清单（每个种子、折、记录）"),
                                    manifest_path.read_bytes(), file_name="experiment_split_manifest.csv", mime="text/csv")
            dl2.download_button(tr("⬇ Per-fold metrics", "⬇ 每折指标"), metrics_csv.read_bytes(),
                                file_name="cv_fold_metrics.csv", mime="text/csv")

        st.markdown(tr("#### Meet the three models", "#### 认识三个模型"))
        st.caption(tr(
            "All three get the same 5 physics inputs and are tested on experiment groups they never saw "
            "(5 seeds × 5 folds = 25 tests). Pick a model to see how it works and how it scored.",
            "三个模型使用相同的 5 个物理输入，并在从未见过的实验组上测试（5 个随机种子 × 5 折 = 25 次测试）。选择一个模型查看其原理与得分。",
        ))
        metric_cols = ["mae_mm_s", "rmse_mm_s", "r2_score", "balanced_accuracy", "macro_f1"]
        stats_by_model = {
            fam: {c: {"mean": float(g[c].mean()), "std": float(g[c].std())} for c in metric_cols}
            for fam, g in m_df.groupby("model_family")
        }
        model_specs = [
            ("Dummy_Baseline", tr("🎯 Baseline", "🎯 基线模型"), baseline_svg,
             tr("Ignores all five physics inputs. It always predicts the average burn rate seen in training and the most "
                "common outcome for extinction. Any useful model has to beat it.",
                "忽略全部 5 个物理输入，始终预测训练集的平均燃烧速率和最常见的熄火结果。任何有用的模型都必须胜过它。")),
            ("Random_Forest", tr("🌳 Random forest", "🌳 随机森林"), forest_svg,
             tr("100 decision trees each ask yes/no questions about the inputs (for example, \"is log Ri below −1?\") and give "
                "an answer; the forest averages them. One forest predicts burn rate, another whether the flame goes out.",
                "100 棵决策树分别对输入提出是/否问题（例如“log Ri 是否小于 −1？”）并给出答案，森林再取平均。一个森林预测燃烧速率，另一个预测火焰是否熄灭。")),
            ("TensorFlow_Multitask_NN", tr("🧠 Neural network", "🧠 神经网络"), network_svg,
             tr("The five inputs flow through two shared layers (32 and 16 neurons). Two output heads then predict both the "
                "burn rate and the chance the flame goes out: one network, two tasks.",
                "5 个输入经过两层共享神经元（32 和 16 个），再由两个输出头同时预测燃烧速率与火焰熄灭概率：一个网络，两个任务。")),
        ]
        base_stats = stats_by_model.get("Dummy_Baseline")
        present = [spec for spec in model_specs if spec[0] in stats_by_model]
        if base_stats and present:
            model_tabs = st.tabs([spec[1] for spec in present])
            for mtab, (fam, _label, svg_fn, how) in zip(model_tabs, present):
                with mtab:
                    left, right = st.columns([1, 1.15], gap="large")
                    left.markdown(_svg_img(svg_fn(), "model-art"), unsafe_allow_html=True)
                    right.markdown(model_panel_html(stats_by_model[fam], base_stats, how), unsafe_allow_html=True)

                    folds = m_df[m_df["model_family"] == fam].sort_values("eval_id")
                    fig_f = go.Figure(go.Scatter(
                        x=folds["eval_id"], y=folds["r2_score"], mode="markers",
                        marker=dict(size=9, color="#3987e5", line=dict(width=2, color="#0f1426")),
                        customdata=folds[["seed", "fold"]],
                        hovertemplate=tr("Seed %{customdata[0]}, fold %{customdata[1]}<br>R² = %{y:.3f}<extra></extra>",
                                         "种子 %{customdata[0]}，第 %{customdata[1]} 折<br>R² = %{y:.3f}<extra></extra>"),
                    ))
                    fig_f.add_hline(y=float(folds["r2_score"].mean()), line=dict(color="#d95926", width=2, dash="dot"),
                                    annotation_text=tr("mean", "均值"), annotation_position="top right",
                                    annotation_font=dict(color="#c9d2ea", size=11))
                    fig_f.update_layout(
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        height=210, margin=dict(l=10, r=10, t=36, b=10), showlegend=False,
                        title=dict(text=tr("R² on each of the 25 held-out tests (one dot = one test)",
                                           "25 次独立测试的 R²（每个点代表一次测试）"), font=dict(size=13, color="#c9d2ea"), x=0),
                        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, title=None),
                        yaxis=dict(range=[-0.35, 1.05], gridcolor="rgba(255,255,255,0.06)", zeroline=False,
                                   tickfont=dict(color="#8f9ab8", size=11)),
                    )
                    st.plotly_chart(fig_f, use_container_width=True, config={"displayModeBar": False})

        with st.expander(tr("Show the full summary table (mean ± std)", "查看完整汇总表（均值 ± 标准差）")):
            summary = m_df.groupby("model_family").agg({
                "mae_mm_s": ["mean", "std"], "rmse_mm_s": ["mean", "std"],
                "r2_score": ["mean", "std"], "balanced_accuracy": ["mean", "std"], "macro_f1": ["mean", "std"]
            }).round(4)
            st.dataframe(summary, use_container_width=True)

        st.markdown(tr("#### Audit warnings and scientific honesty declarations", "#### 审计警示与科学诚实声明 (Honesty Declarations)"))
        # The audit script writes these warnings in English; known messages get a Chinese version here.
        # An unknown (new) message is shown in English rather than hidden.
        report_warnings_zh = {
            "At least one exact feature+target pattern occurs across distinct group IDs. Inspect whether this is a "
            "legitimate repeated condition or accidental row duplication; group separation alone cannot fix mislabeled group IDs.":
                "至少有一组完全相同的特征+目标组合出现在不同的实验组 ID 中。请检查这是合理的重复工况还是意外的重复行；"
                "仅靠分组划分无法修正标错的组 ID。",
            "Exploratory mode: provenance/target review blockers were overridden. Do not present resulting metrics as "
            "scientifically validated.":
                "探索模式：来源/目标审查的阻断项已被跳过。不得将所得指标作为已科学验证的结果呈现。",
        }
        for warn in a_data.get("warnings", []):
            st.warning(f"⚠️ {report_warnings_zh.get(warn, warn) if lang == 'zh' else warn}")


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
        "🧪 The sample CSV is **made-up demo input**, not measured burns. Uploading any CSV only runs predictions with the "
        "already-trained model; it does not retrain or validate it. The model itself was trained and checked on the real "
        "MaCFP measurements described above.",
        "🧪 示例 CSV 是**虚构的演示输入**，并非实测燃烧数据。上传任何 CSV 只会用已训练好的模型进行预测，不会重新训练或验证模型。"
        "模型本身是在上文所述的真实 MaCFP 实测数据上训练与检验的。",
    ))
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
                # The scope sentence comes from the model metadata in English; the known one has a Chinese version.
                scope_zh = {
                    "Post-test PMMA cone-calorimeter measurements; spaceflight performance unverified.":
                        "试验后的 PMMA 锥形量热仪测量数据；在航天环境下的表现未经验证。",
                }.get(model_info["scope"], model_info["scope"])
                st.caption(tr(
                    f"Model source: {model_info['source']}. Scope: {model_info['scope']}",
                    f"模型来源：{model_info['source']}。适用范围：{scope_zh}",
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
