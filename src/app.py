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
import html
import json
import math
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
from app.services import flex_inference as flex
from app.services.insights import crew_insights
from app.services.simulation_service import run_simulation
from app.settings import read_json_config
from app.services.provenance import dataset_status, load_report as load_provenance_report
from app.services.research_engine import (
    get_empirical_material_ranking,
    ranking_insights,
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
/* Metric cards in one row all take the height of the tallest card */
[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) { align-items: stretch; }
[data-testid="stColumn"]:has([data-testid="stMetric"]) > [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stColumn"]:has([data-testid="stMetric"]) > [data-testid="stVerticalBlockBorderWrapper"] > div,
[data-testid="stColumn"]:has([data-testid="stMetric"]) [data-testid="stVerticalBlock"],
[data-testid="stElementContainer"]:has(> [data-testid="stMetric"]) { height: 100%; }
[data-testid="stMetric"] { height: 100%; box-sizing: border-box; }
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
.telemetry.compact .tele-v { font-size: 1.15rem; }  /* AI prediction results */
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
/* Top three shown; the rest unfold below when the reader opens "show more" */
.rank-extra { display: none; }
.rank-wrap:has(> .rank-more[open]) > .rank-extra { display: grid; animation: rankIn .45s ease both; }
@keyframes rankIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
.rank-more { margin: 0 0 16px; }
.rank-more summary { list-style: none; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px;
  padding: 11px 16px; border-radius: 14px; border: 1px dashed rgba(255,154,77,.45); background: rgba(255,106,43,.06);
  color: #FFD9B8; font-weight: 600; font-size: .92rem; transition: background .2s ease, border-color .2s ease; }
.rank-more summary::-webkit-details-marker { display: none; }
.rank-more summary:hover { background: rgba(255,106,43,.13); border-color: #FF9A4D; }
.rank-more summary:focus-visible { outline: 2px solid #FF9A4D; outline-offset: 2px; }
.rank-more .less, .rank-more[open] .more { display: none; }
.rank-more[open] .less { display: inline; }
.rank-more .chev { display: inline-block; transition: transform .25s ease; }
.rank-more[open] .chev { transform: rotate(180deg); }

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

/* Front page: crew fire-safety takeaways */
.insights { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; margin: 4px 0 26px; }
.icard { display: flex; flex-direction: column; padding: 16px 18px; border-radius: 16px;
  background: linear-gradient(160deg, #1a2140, #121830); border: 1px solid rgba(255,255,255,.09);
  border-top: 3px solid #FF9A4D; transition: transform .25s ease, border-color .25s ease; }
.icard:hover { transform: translateY(-4px); border-color: rgba(252,61,33,.6); }
.ihead { display: flex; align-items: center; gap: 10px; color: #fff; font-weight: 700; font-size: 1.02rem; line-height: 1.3; }
.iicon { font-size: 1.35rem; }
.icard p { margin: 10px 0 0; font-size: .88rem; color: #c3cbe0; line-height: 1.5; }
.icard p.iact { color: #FFC59A; font-weight: 600; }
.ifoot { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: auto; padding-top: 12px;
  font-size: .76rem; color: #8f9ab8; }

/* Audit tab: prediction-task overview table */
.tasks { overflow-x: auto; margin: 2px 0 20px; border-radius: 14px; border: 1px solid rgba(255,255,255,.09);
  background: linear-gradient(160deg, #1a2140, #121830); }
.tasks table { width: 100%; min-width: 660px; border-collapse: collapse; margin: 0; }
.tasks th { text-align: left; padding: 12px 16px; font-size: .68rem; font-weight: 700; letter-spacing: .12em;
  text-transform: uppercase; color: #8f9ab8; border: none; border-bottom: 1px solid rgba(255,255,255,.09); }
.tasks td { padding: 14px 16px; vertical-align: top; font-size: .88rem; color: #c3cbe0; border: none;
  border-bottom: 1px solid rgba(255,255,255,.06); }
.tasks tr:last-child td { border-bottom: none; }
.tasks tbody tr:hover { background: rgba(255,255,255,.03); }
.tasks .t { color: #fff; font-weight: 600; line-height: 1.35; }
.tasks .sub { color: #8f9ab8; font-size: .78rem; line-height: 1.45; margin-top: 5px; }
.tasks .sub code { font-size: .74rem; }
.tasks .size { white-space: nowrap; }
.pill { display: inline-block; white-space: nowrap; padding: 3px 10px; border-radius: 999px; font-size: .7rem;
  font-weight: 700; letter-spacing: .06em; border: 1px solid; }
.pill.ok { color: #7dffa8; border-color: rgba(125,255,168,.4); background: rgba(125,255,168,.08); }
.pill.no { color: #aab3cc; border-color: rgba(170,179,204,.35); background: rgba(170,179,204,.08); }
.pill.warn { color: #FFB27A; border-color: rgba(255,178,122,.45); background: rgba(255,178,122,.1); }
.pill.info { color: #7CC4FF; border-color: rgba(124,196,255,.45); background: rgba(124,196,255,.1); }
.pill.env { font-weight: 600; letter-spacing: .02em; margin-top: 6px; }
/* Audit tab: report downloads as equal-size tiles */
.st-key-report_files .stDownloadButton > button { height: 3.4rem; padding: 0 .8rem; line-height: 1.25; }
.st-key-report_files .stDownloadButton p { font-size: .85rem; }
@media (max-width: 1280px) { .st-key-report_files .stDownloadButton > button { height: 4.3rem; } }  /* names wrap to 3 lines */
@media (max-width: 640px) { .st-key-report_files .stDownloadButton > button { height: 3rem; } }    /* columns stack: 1 line */

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
        "Educational and research use only. The prediction models are experimental: the FLEX models cover ISS "
        "droplet tests only, the PMMA model ground tests only, and none is validated for spacecraft "
        "fire-safety decisions.",
        "仅供教育与科研使用。预测模型均为实验性质：FLEX 模型仅涵盖国际空间站液滴试验，PMMA 模型仅涵盖地面试验，"
        "均未经航天器消防安全决策验证。",
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
def load_json_file(path_str: str) -> dict | None:
    try:
        return json.loads(Path(path_str).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# Static facts (config) vs computed results (reports). Statuses always come from the reports.
SOURCES = load_json_file(str(ROOT / "config" / "data_sources.json")) or {"datasets": {}}
REGISTRY = load_json_file(str(ROOT / "config" / "target_registry.json")) or {"tasks": []}
PROV_REPORT = load_provenance_report() if (REPORTS_DIR / "data_provenance_audit.json").exists() else None
CV_SUMMARY = load_json_file(str(REPORTS_DIR / "cv_summary.json"))
NASA_KEYS = ("BASS_II", "FLEX")


def ds_status(key: str) -> dict:
    """Computed status of a dataset; explicit UNVERIFIED (with a warning) when the report is missing."""
    return dataset_status(key, PROV_REPORT)


STATUS_LABELS = {
    "VERIFIED_SOURCE": ("🟢", "VERIFIED SOURCE", "已核实来源"),
    "REAL_MEASUREMENT": ("🟢", "REAL MEASUREMENT", "真实测量"),
    "UNVERIFIED_SOURCE": ("🟠", "UNVERIFIED SOURCE", "来源未核实"),
    "SYNTHETIC_DEMO": ("🟠", "SYNTHETIC DEMO DATA", "合成演示数据"),
    "MODEL_PREDICTION": ("🔵", "MODEL PREDICTION", "模型预测"),
    "SIMULATED_RESULT": ("🟡", "SIMULATED RESULT", "模拟结果"),
    "NOT_EVALUATED": ("⚪", "NOT EVALUATED", "未评估"),
    "EVALUATED": ("🟢", "EVALUATED", "已评估"),
}


def status_badge(code: str) -> str:
    icon, en, zh = STATUS_LABELS.get(code, ("⚪", code, code))
    return f"{icon} **{tr(en, zh)}**"


STATUS_TONE = {"VERIFIED_SOURCE": "ok", "REAL_MEASUREMENT": "ok", "EVALUATED": "ok", "NOT_EVALUATED": "no",
               "UNVERIFIED_SOURCE": "warn", "SYNTHETIC_DEMO": "warn"}


def status_pill(code: str) -> str:
    """HTML version of status_badge, for tables built with HTML."""
    _, en, zh = STATUS_LABELS.get(code, ("", code, code))
    return f'<span class="pill {STATUS_TONE.get(code, "info")}">{html.escape(tr(en, zh))}</span>'


# Plain-language text for the machine-readable codes written by src/provenance_audit.py and
# src/run_cv_evaluation.py. Unknown codes are shown as-is rather than hidden.
CODE_TEXT = {
    "GENERATOR_REPRODUCES_ALL_ROWS": ("All {rows} rows are reproduced exactly by `{generator}`, so they are generated, not measured.",
                                      "全部 {rows} 行都能由 `{generator}` 完全复现，因此是生成的数据而非测量值。"),
    "GENERATOR_REPRODUCES_SOME_ROWS": ("{rows} of {of} rows are reproduced by `{generator}`.", "{of} 行中有 {rows} 行可由 `{generator}` 复现。"),
    "NO_OFFICIAL_FILE_REGISTERED": ("No official {investigation} file with a checksum is registered in `config/data_sources.json`.",
                                    "`config/data_sources.json` 中尚未登记带校验值的 {investigation} 官方文件。"),
    "SOURCE_FILE_MISSING": ("Source file `{path}` is not in the repository.", "来源文件 `{path}` 不在仓库中。"),
    "OFFICIAL_CHECKSUM_MATCH": ("`{path}` matches its registered official checksum.", "`{path}` 与登记的官方校验值一致。"),
    "DUPLICATE_ROWS": ("{count} duplicate rows found.", "发现 {count} 条重复行。"),
    "CLONE_AT_PINNED_COMMIT": ("Source repository clone is at the pinned commit `{pinned}`.", "来源仓库克隆位于固定提交 `{pinned}`。"),
    "CLONE_COMMIT_MISMATCH": ("Clone is at `{commit}` but `{pinned}` is pinned.", "克隆位于 `{commit}`，但固定提交为 `{pinned}`。"),
    "CLONE_CLEAN": ("No local edits in the source files.", "来源文件没有本地修改。"),
    "CLONE_HAS_LOCAL_EDITS": ("Source files have local edits.", "来源文件存在本地修改。"),
    "REBUILD_BYTE_IDENTICAL": ("Rebuilding from the source files gives a byte-identical file (SHA-256 `{rebuilt_sha256}`).",
                               "由来源文件重新构建得到逐字节相同的文件（SHA-256 `{rebuilt_sha256}`）。"),
    "REBUILD_DIFFERS": ("Rebuilding from the source files gives a different file.", "由来源文件重新构建得到的文件与现有文件不同。"),
    "SOURCE_CLONE_MISSING": ("Source clone `{path}` is not available, so the data cannot be re-checked here.",
                             "来源克隆 `{path}` 不可用，因此无法在此重新核验。"),
    "MODEL_TRAINED_ON_THIS_FILE": ("The deployed model's metadata records this exact file as its training data.",
                                   "已部署模型的元数据记录的训练数据正是此文件。"),
    "MODEL_TRAINED_ON_OTHER_FILE": ("The deployed model was trained on a different file.", "已部署模型使用的是另一个训练文件。"),
    "PROCESSED_FILE_MISSING": ("Processed file `{path}` is missing.", "缺少处理后的文件 `{path}`。"),
    "NO_RECORDS_LOADED": ("No records are loaded for this dataset.", "该数据集未载入任何记录。"),
    "DATASET_NOT_VERIFIED": ("The dataset is not a verified source (status: {dataset_status}).", "数据集不是已核实来源（状态：{dataset_status}）。"),
    "TARGET_UNIT_UNVERIFIED": ("The target's unit has not been confirmed from the source documentation.", "目标量的单位尚未依据来源文档确认。"),
    "TARGET_NOT_MEASURED": ("The table contains no measured value for this target.", "数据表中没有该目标的测量值。"),
    "LABEL_MAPPING_UNVERIFIED": ("How outcomes map to classes has not been confirmed from the source documentation.",
                                 "结果类别的映射方式尚未依据来源文档确认。"),
    "LABEL_NOT_OBSERVED": ("No observed outcome exists for this label.", "该标签没有观测到的结果。"),
    "DATA_FILE_MISSING": ("Data file `{path}` is missing.", "缺少数据文件 `{path}`。"),
    "INSUFFICIENT_GROUPS": ("Too few independent groups ({groups}; need {required}).", "独立组数不足（{groups}，需要 {required}）。"),
    "DERIVED_FROM_SYNTHETIC_TABLES": ("Built from the formula-generated BASS-II/FLEX tables ({rows} rows).",
                                      "由公式生成的 BASS-II/FLEX 数据表构建（{rows} 行）。"),
    "TARGET_COMPUTED_FROM_INPUT": ("{rows} of {of} BASS-II targets equal the formula {formula}: the label was computed from an input (leakage).",
                                   "{of} 个 BASS-II 目标值中有 {rows} 个等于公式 {formula}：标签由输入计算得到（泄漏）。"),
    "INCOMPATIBLE_TARGETS_MERGED": ("A BASS-II proxy value and the FLEX burning rate were merged into one target column despite different meanings and units.",
                                    "BASS-II 代理值与 FLEX 燃烧速率被合并为同一目标列，尽管含义与单位不同。"),
    "CONSTANT_FALLBACK_VALUES": ("All {rows} rows hold the same fallback constants (distinct values per column: {distinct_values}).",
                                 "全部 {rows} 行都是相同的回退常数（各列不同值数量：{distinct_values}）。"),
    "IDENTICAL_TO_TEMPLATE": ("Byte-identical to its template, with {placeholders} unfilled placeholders.",
                              "与模板逐字节相同，含 {placeholders} 个未填写的占位符。"),
    "REFERENCED_MODEL_MISSING": ("Refers to `{model}`, which does not exist in the repository.", "引用的 `{model}` 在仓库中不存在。"),
    "CHECKSUM_MISMATCH": ("`{path}` does not match its registered checksum.", "`{path}` 与登记的校验值不一致。"),
}


class _Blank(dict):
    def __missing__(self, key: str) -> str:
        return "?"


# Chinese names for English item labels that appear inside evidence details (so 中文 mode stays Chinese).
DETAIL_ZH = {"Apex: S-IC Stage / Range Time (sec)": "第一级残骸到达最高点的时间（秒）"}


def code_text(item: dict) -> str:
    """Render one evidence/reason item: {'code': ..., 'detail': {...}} or {'code': ..., **detail}."""
    detail = {**{k: v for k, v in item.items() if k not in ("code", "detail")}, **(item.get("detail") or {})}
    detail = {k: (str(v)[:12] + "…" if isinstance(v, str) and len(v) == 64 else v) for k, v in detail.items()}
    detail = {k: tr(v, DETAIL_ZH.get(v, v)) if isinstance(v, str) else v for k, v in detail.items()}
    en, zh = CODE_TEXT.get(item.get("code", ""), (item.get("code", ""), item.get("code", "")))
    return tr(en, zh).format_map(_Blank(detail))


@st.cache_data(show_spinner=False)
def load_provenance_mapping(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def headline_numbers() -> dict:
    """Every number in the header is computed from data files and generated reports, never typed in."""
    out = {"nasa_verified": sum(ds_status(k)["status"] == "VERIFIED_SOURCE" for k in NASA_KEYS),
           "nasa_total": len(NASA_KEYS), "cone_burns": 0, "cone_status": ds_status("MACFP_PMMA_CONE")["status"],
           "folds": 0, "tasks_evaluated": 0, "physics_tests": 0}
    if PROV_REPORT:
        out["cone_burns"] = int(PROV_REPORT["datasets"].get("MACFP_PMMA_CONE", {}).get("rows") or 0)
    if CV_SUMMARY:
        evaluated = [t for t in CV_SUMMARY.get("tasks", []) if t.get("evaluation_status") == "EVALUATED"]
        out["tasks_evaluated"] = len(evaluated)
        out["folds"] = sum(int(t.get("outer_folds_total", 0)) for t in evaluated)
    tests_path = ROOT / "tests" / "test_simulation_physics_limits.py"
    if tests_path.exists():
        out["physics_tests"] = tests_path.read_text(encoding="utf-8").count("\ndef test_")
    return out


_nums = headline_numbers()
_telemetry = [
    (f"{_nums['nasa_verified']}/{_nums['nasa_total']}", tr("NASA flight datasets verified", "NASA 飞行数据集已核实")),
    (str(_nums["cone_burns"]), tr("Real MaCFP cone burns", "真实 MaCFP 锥形量热试验")
     + (tr(" · verified", " · 已核实") if _nums["cone_status"] == "VERIFIED_SOURCE" else tr(" · unverified", " · 未核实"))),
    (str(_nums["folds"]), tr(f"Leakage-audited CV folds · {_nums['tasks_evaluated']} "
                             + ("task" if _nums["tasks_evaluated"] == 1 else "tasks"),
                             f"无泄漏审计的交叉验证折 · {_nums['tasks_evaluated']} 个任务")),
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
if PROV_REPORT is None:
    st.error(tr(
        "**Provenance report missing.** `reports/data_provenance_audit.json` could not be read, so every dataset is "
        "treated as UNVERIFIED. Run `python -m src.provenance_audit` to compute statuses.",
        "**缺少数据来源报告。** 无法读取 `reports/data_provenance_audit.json`，因此所有数据集都按“未核实”处理。"
        "请运行 `python -m src.provenance_audit` 计算状态。",
    ), icon="🚫")
else:
    _not_verified = [k for k in NASA_KEYS if ds_status(k)["status"] != "VERIFIED_SOURCE"]
    if _not_verified:
        _parts = [f"{cfg_text(SOURCES['datasets'].get(k, {}), 'display_name', k)}: {status_badge(ds_status(k)['status'])}"
                  for k in _not_verified]
        st.warning(tr(
            "**Data status (computed by `src/provenance_audit.py`).** " + " · ".join(_parts) + ". These tables are not "
            "NASA measurements, so their rankings are demonstrations and no model is scored on them. The MaCFP cone "
            "data behind the prediction tab is verified real measurement. Details: Evidence tab → Data provenance.",
            "**数据状态（由 `src/provenance_audit.py` 计算）。** " + " · ".join(_parts) + "。这些数据表并非 NASA 实测，"
            "因此其排名仅为演示，也不会用它们给任何模型打分。预测页使用的 MaCFP 锥形量热数据是已核实的真实测量。详情见：证据页 → 数据来源。",
        ), icon="⚠️")

def _flame_path(cx: float, base: float, w: float, h: float) -> str:
    return (f"M {cx - w:.1f} {base:.1f} "
            f"C {cx - w:.1f} {base - 0.45 * h:.1f}, {cx - 0.35 * w:.1f} {base - 0.55 * h:.1f}, {cx:.1f} {base - h:.1f} "
            f"C {cx + 0.35 * w:.1f} {base - 0.55 * h:.1f}, {cx + w:.1f} {base - 0.45 * h:.1f}, {cx + w:.1f} {base:.1f} "
            f"Q {cx:.1f} {base + 0.18 * w:.1f}, {cx - w:.1f} {base:.1f} Z")


def flame_diagram_svg(flux: float, duration: float, mean_hrr: float, peak: float,
                      threshold: float, ref_peak: float, title: str, measured: float | None = None) -> str:
    """Animated, labelled cone-calorimeter schematic for one burn.

    Flame height scales with peak HRR^0.4 (Heskestad-style flame-height scaling), the
    sustained flame size with mean/peak HRR, and the number of radiant-heat arrows with
    the applied heat flux. Schematic only; not to scale and not CFD.
    `peak` is the model prediction. When the real `measured` peak is known it drives the flame,
    and the bottom bar shows measured vs predicted instead of the hazard flag.
    """
    predicted = peak
    if measured is not None:
        peak = measured
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
{_result_bar(lbl, predicted, measured, badge_col)}
</svg>"""


def _result_bar(lbl: dict, predicted: float, measured: float | None, badge_col: str) -> str:
    """Bottom bar: predicted peak + hazard flag, or (when known) measured vs out-of-fold prediction + error."""
    if measured is None:
        return (f'<rect x="14" y="366" width="532" height="34" rx="10" fill="#151c3a" stroke="#2a3355"/>'
                f'<text x="28" y="388" fill="#aab3cc" font-size="12">{lbl["peak"]}</text>'
                f'<text x="{170 if lang == "en" else 150}" y="389" fill="#FFFFFF" font-size="17" font-weight="800">{predicted:,.0f} kW/m²</text>'
                f'<rect x="370" y="372" width="168" height="22" rx="11" fill="{badge_col}" fill-opacity="0.18" stroke="{badge_col}"/>'
                f'<text x="454" y="387" fill="{badge_col}" font-size="11" font-weight="800" text-anchor="middle">{lbl["badge"]}</text>')
    err = (predicted - measured) / measured * 100 if measured else 0.0
    return (f'<rect x="14" y="364" width="532" height="40" rx="10" fill="#151c3a" stroke="#2a3355"/>'
            f'<text x="28" y="379" fill="#2EBD6B" font-size="9.5" font-weight="700" letter-spacing="1">{tr("MEASURED (REAL)", "实测（真实）")}</text>'
            f'<text x="28" y="397" fill="#FFFFFF" font-size="15" font-weight="800">{measured:,.0f} kW/m²</text>'
            f'<text x="200" y="379" fill="#7CB6FF" font-size="9.5" font-weight="700" letter-spacing="1">{tr("MODEL PREDICTION (OUT-OF-FOLD)", "模型预测（折外）")}</text>'
            f'<text x="200" y="397" fill="#7CB6FF" font-size="15" font-weight="800">{predicted:,.0f} kW/m²</text>'
            f'<rect x="420" y="372" width="118" height="24" rx="12" fill="#8f9ab8" fill-opacity="0.15" stroke="#8f9ab8"/>'
            f'<text x="479" y="388" fill="#c9d2ea" font-size="11" font-weight="800" text-anchor="middle">{tr("error", "误差")} {err:+.0f}%</text>')


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
                measured=row.get("measured_peak_hrr_kw_m2"),
            )
            b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
            col.markdown(f'<img class="flame-card" alt="" src="data:image/svg+xml;base64,{b64}">',
                         unsafe_allow_html=True)
    if len(rows) > max_cards:
        st.caption(tr(f"Showing the first {max_cards} of {len(rows)} burns. All results are in the table below.",
                      f"显示前 {max_cards} 条（共 {len(rows)} 条）。完整结果见下表。"))


def preview_burns() -> list[dict]:
    """Two real MaCFP burns chosen by a fixed rule (the median-peak run at the lowest and at the highest heat
    flux), each paired with the network's out-of-fold prediction from reports/cv_predictions.csv, i.e. a model
    that never saw that laboratory. Nothing is typed in; returns [] if either file is missing."""
    data = load_report_csv(str(ROOT / "data" / "processed" / "macfp_measured_cone_experiments.csv"))
    preds = load_report_csv(str(REPORTS_DIR / "cv_predictions.csv"))
    if data.empty or preds.empty:
        return []
    p = preds[(preds["task_id"] == "macfp_pmma_peak_hrr") & (preds["model"] == "tf_nn_deployed_config")]
    if p.empty:
        return []
    oof = p.groupby("record_id")["y_pred"].mean()
    rows = []
    for flux in (data["radiant_heat_flux_kw_m2"].min(), data["radiant_heat_flux_kw_m2"].max()):
        sub = data[data["radiant_heat_flux_kw_m2"] == flux].sort_values("target_peak_hrr_kw_m2").reset_index(drop=True)
        r = sub.iloc[(len(sub) - 1) // 2]
        if r["experiment_id"] in oof.index:
            rows.append({"radiant_heat_flux_kw_m2": r["radiant_heat_flux_kw_m2"], "real_burn_duration_s": r["real_burn_duration_s"],
                         "real_mean_hrr_kw_m2": r["real_mean_hrr_kw_m2"], "predicted_peak_hrr_kw_m2": float(oof[r["experiment_id"]]),
                         "measured_peak_hrr_kw_m2": float(r["target_peak_hrr_kw_m2"]),
                         "_title": tr(f"REAL BURN · {r['laboratory']} · {r['experiment_id']}",
                                      f"真实燃烧试验 · {r['laboratory']} · {r['experiment_id']}")})
    return rows


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


def solid_svg(material_class: str, co_mean: float | None, co_max_all: float) -> str:
    """One BASS-II material class burning in the flow duct (side view), modelled on NASA ISS photos.

    In microgravity the flame is a blue, bullet/lens-shaped envelope lying along the airflow with an
    orange glow where it is hottest; smoke is carried downstream (nothing rises). Rods and the candle burn
    at the tip; films and slabs burn on both faces as the front moves along; SIBAL leaves its fiberglass
    mesh; the sphere section carries a dome-shaped flame. Smoke wisps scale with the class's mean final CO.
    Illustration, not footage.
    """
    co_frac = (co_mean / co_max_all) if (co_mean and co_max_all) else 0.2
    n_smoke = 1 + int(round(4 * co_frac))
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
    extra_defs, extra_css = "", ""

    def smoke_at(x: float, spread: float = 5.0) -> str:
        return "".join(
            f'<ellipse class="smoke" cx="{x:.0f}" cy="{cy + (k - n_smoke / 2) * spread:.1f}" rx="7" ry="4" fill="#7d8597" '
            f'style="animation-delay:{-0.6 * k:.1f}s"/>' for k in range(n_smoke))

    if material_class in ("pmma_rod_clear", "pmma_rod_black", "wax_candle"):
        tip_x, rad, start = (150.0, 9.0, 24.0) if material_class != "wax_candle" else (140.0, 11.0, 70.0)
        stops = {"pmma_rod_clear": ("#e9f4ff", "#a8c8e8", "0.75"), "pmma_rod_black": ("#5a5f6e", "#15171d", "1"),
                 "wax_candle": ("#fffaf0", "#e0d2b4", "1")}[material_class]
        extra_defs = (f'<linearGradient id="rodfill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{stops[0]}"/>'
                      f'<stop offset="1" stop-color="{stops[1]}"/></linearGradient>')
        body = f"""
  <rect x="6" y="{cy - 8}" width="18" height="16" rx="2" fill="#4a5270"/>
  <rect x="24" y="{cy - 3}" width="{start - 24}" height="6" fill="#4a5270"/>
  <path d="M{start} {cy - rad} H{tip_x} A{rad} {rad} 0 0 1 {tip_x} {cy + rad} H{start} Z" fill="url(#rodfill)" fill-opacity="{stops[2]}" stroke="#ffffff" stroke-opacity="0.25"/>
  <path d="M{tip_x - 16} {cy - rad} H{tip_x} A{rad} {rad} 0 0 1 {tip_x} {cy + rad} H{tip_x - 16} Z" fill="#3a2a22" opacity="0.5"/>
  {smoke_at(tip_x + 34)}
  <g class="breathe">
    <path d="M{tip_x - 54} {cy - 22} Q {tip_x + 2} {cy - 30} {tip_x + 30} {cy} Q {tip_x + 2} {cy + 30} {tip_x - 54} {cy + 22}" fill="rgba(255,150,120,0.10)" stroke="url(#env)" stroke-width="3" filter="url(#soft)"/>
  </g>
  <ellipse cx="{tip_x + 3}" cy="{cy}" rx="11" ry="13" fill="url(#glow)"/>"""
    elif material_class == "pmma_sphere_section":
        body = f"""
  <rect x="96" y="{cy + 16}" width="28" height="12" rx="2" fill="#4a5270"/>
  <path d="M84 {cy + 16} A26 26 0 0 1 136 {cy + 16} Z" fill="#cfe6ff" fill-opacity="0.6" stroke="#ffffff" stroke-opacity="0.3"/>
  {smoke_at(160, 4)}
  <g class="breathe">
    <path d="M74 {cy + 16} A36 40 0 0 1 150 {cy + 16}" fill="rgba(255,150,120,0.08)" stroke="url(#env)" stroke-width="3" filter="url(#soft)"/>
  </g>
  <ellipse cx="122" cy="{cy - 4}" rx="10" ry="12" fill="url(#glow)"/>"""
    else:
        # Sheets: thin film, slab, SIBAL or Nomex fabric. Held at one end; the flame front travels downstream.
        thick = {"pmma_thin_film": 1.8, "pmma_slab": 7.0, "sibal_fabric": 4.0, "nomex_fabric": 4.0}.get(material_class, 2.0)
        fabric = material_class in ("sibal_fabric", "nomex_fabric")
        x0, x1 = 20.0, 205.0
        flame_ry = 10 + thick * 1.4
        mesh = (f'<line x1="{x0}" y1="{cy}" x2="{x1}" y2="{cy}" stroke="#9aa0ad" stroke-width="{thick:.1f}" '
                f'stroke-dasharray="2 2" opacity="0.7"/>') if material_class == "sibal_fabric" else ""
        weave = "#d9c9a8" if material_class == "sibal_fabric" else "#e8dfa0"
        sheet_fill = "url(#weave)" if fabric else "#bfe0ff"
        sheet_op = "1" if fabric else ("0.55" if material_class == "pmma_thin_film" else "0.7")
        body = f"""
  <rect x="6" y="{cy - 10}" width="14" height="20" rx="2" fill="#4a5270"/>
  {mesh}
  <rect class="unburnt" x="{x0}" y="{cy - thick / 2:.2f}" width="{x1 - x0}" height="{thick:.2f}" fill="{sheet_fill}" fill-opacity="{sheet_op}"/>
  <g class="front">
    {smoke_at(x0 + 30, 5)}
    <g class="breathe">
      <ellipse cx="{x0 + 12}" cy="{cy}" rx="28" ry="{flame_ry:.1f}" fill="rgba(255,150,120,0.10)" stroke="url(#env)" stroke-width="2.5" filter="url(#soft)"/>
    </g>
    <ellipse cx="{x0 + 2}" cy="{cy - thick / 2 - 3:.1f}" rx="9" ry="5" fill="url(#glow)"/>
    <ellipse cx="{x0 + 2}" cy="{cy + thick / 2 + 3:.1f}" rx="9" ry="5" fill="url(#glow)"/>
  </g>"""
        extra_defs = (f'<pattern id="weave" width="5" height="5" patternUnits="userSpaceOnUse"><rect width="5" height="5" fill="{weave}"/>'
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


RANK_TOP_N = 3  # cards shown before "show more"


def ranking_cards_html(track_key: str, ranking: list[dict]) -> str:
    """Card per material class / fuel: animated object, medal, and stat bars, all from the ranking rows."""
    medals = ["🥇", "🥈", "🥉"]

    def bar(label: str, value: str, frac: float | None, color: str) -> str:
        pct = max(4.0, min(100.0, 100 * (frac or 0)))
        return (f'<div class="stat"><div class="stat-top"><span>{label}</span><b>{value}</b></div>'
                f'<div class="stat-bar"><i style="width:{pct:.0f}%;background:{color}"></i></div></div>')

    def fmt(x: float | None, pattern: str) -> str:
        return "—" if x is None else pattern.format(x)

    cards = []
    if track_key == "liquid_droplet":
        rate_max = max((r["mean_burning_rate_constant_mm2_s"] or 0) for r in ranking) or 1.0
        d_max = max(max((r["mean_initial_diameter_mm"] or 0), (r["mean_extinction_diameter_mm"] or 0)) for r in ranking) or 1.0
        for r in ranking:
            k = r["mean_burning_rate_constant_mm2_s"] or 0.0
            d0 = r["mean_initial_diameter_mm"] or 2.0
            d_ext = r["mean_extinction_diameter_mm"] or d0 * 0.5
            cycle = 4.5 + 3.0 * (1 - k / rate_max) + 1.5  # faster-burning fuel shrinks faster
            obj = _svg_img(droplet_svg(r["fuel_name"], k, rate_max, d0, d_ext, cycle), "rank-obj")
            oc = r["outcome_counts"]
            ext_n, comp_n, disr_n = oc["Extinction"], oc["Completion"], oc["Disruption"]
            outcome_line = tr(f"Outcomes: {ext_n} extinction · {comp_n} completion · {disr_n} disruption",
                              f"结果：熄灭 {ext_n} · 燃尽 {comp_n} · 破碎 {disr_n}")
            share = r["extinction_share_of_non_disrupted"]
            stats = (bar(tr("Burning-rate constant K", "燃烧速率常数 K"), fmt(r["mean_burning_rate_constant_mm2_s"], "{:.3f} mm²/s"), k / rate_max, "#FF8A3D")
                     + bar(tr("Initial droplet diameter", "初始液滴直径"), fmt(r["mean_initial_diameter_mm"], "{:.2f} mm"), d0 / d_max, "#9DA8C8")
                     + bar(tr("Diameter when the flame went out", "熄火时液滴直径"), fmt(r["mean_extinction_diameter_mm"], "{:.2f} mm"), d_ext / d_max, "#7CC4FF")
                     + bar(tr("Flame went out (of non-disrupted tests)", "火焰熄灭比例（不含破碎试验）"), fmt(share * 100 if share is not None else None, "{:.0f}%"), share, "#FC3D21")
                     + f'<div class="rank-sub" style="margin-top:2px">{outcome_line}</div>')
            cards.append((r["rank"], r["fuel_name"], r["empirical_test_count"], obj, stats))
    else:
        co_max_all = max((r["max_final_co_ppm"] or 0) for r in ranking) or 1.0
        for r in ranking:
            obj = _svg_img(solid_svg(r["material_class"], r["mean_final_co_ppm"], co_max_all), "rank-obj")
            over, n_used = r["co_over_range_readings"], r["co_numeric_readings"]
            co_note = tr(f"{n_used} CO readings used" + (f"; {over} over range, not averaged" if over else ""),
                         f"使用 {n_used} 个 CO 读数" + (f"；{over} 个超量程读数未计入平均" if over else ""))
            stats = (bar(tr("Mean final CO", "平均最终 CO"), fmt(r["mean_final_co_ppm"], "{:.0f} ppm"), (r["mean_final_co_ppm"] or 0) / co_max_all, "#FF8A3D")
                     + bar(tr("Highest final CO", "最高最终 CO"), fmt(r["max_final_co_ppm"], "{:.0f} ppm"), (r["max_final_co_ppm"] or 0) / co_max_all, "#FC3D21")
                     + bar(tr("O₂ consumed", "O₂ 消耗"), fmt(r["mean_o2_depletion_pct_vol"], "{:.2f} %vol"), min(1.0, (r["mean_o2_depletion_pct_vol"] or 0) / 1.5), "#7CC4FF")
                     + f'<div class="rank-sub" style="margin-top:2px">{co_note}</div>')
            cards.append((r["rank"], cfg_text(r, "material_name"), r["empirical_test_count"], obj, stats))

    def card_html(rank: int, name: str, count: int, obj: str, stats: str) -> str:
        medal = medals[rank - 1] if rank <= 3 else f"#{rank}"
        return (
            f'<div class="rank-card"><div class="rank-head"><span class="medal">{medal}</span>'
            f'<div><div class="rank-name">{name}</div><div class="rank-sub">{tr(f"{count} ISS tests", f"{count} 次空间站试验")}</div></div></div>'
            f'{obj}<div class="stats">{stats}</div></div>'
        )

    # Top three first; the rest stay folded behind a "show more" toggle (pure CSS, so no rerun).
    cards.sort(key=lambda c: c[0])
    top = "".join(card_html(*c) for c in cards[:RANK_TOP_N])
    rest = cards[RANK_TOP_N:]
    if not rest:
        return f'<div class="rank-grid">{top}</div>'
    noun_en, noun_zh = ("fuels", "种燃料") if track_key == "liquid_droplet" else ("materials", "种材料")
    more = tr(f"Show {len(rest)} more {noun_en}", f"展开其余 {len(rest)} {noun_zh}")
    less = tr(f"Show top {RANK_TOP_N} only", f"只看前 {RANK_TOP_N} 名")
    return (
        f'<div class="rank-wrap"><div class="rank-grid">{top}</div>'
        f'<div class="rank-grid rank-extra">{"".join(card_html(*c) for c in rest)}</div>'
        f'<details class="rank-more"><summary><span class="more">{more}</span><span class="less">{less}</span>'
        f'<span class="chev">▾</span></summary></details></div>'
    )


FEATURE_SHORT = {  # short chip labels for model diagrams, keyed by registry feature/target column
    "radiant_heat_flux_kw_m2": ("heat flux", "热通量"),
    "real_burn_duration_s": ("burn time", "燃烧时间"),
    "real_mean_hrr_kw_m2": ("mean HRR", "平均 HRR"),
    "target_peak_hrr_kw_m2": ("peak HRR", "峰值 HRR"),
    "is_heptane": ("fuel", "燃料"),
    "pressure_mmhg": ("pressure", "压力"),
    "x_o2": ("O₂", "O₂"),
    "x_co2": ("CO₂", "CO₂"),
    "x_he": ("He", "He"),
    "d0_mm": ("drop size", "液滴直径"),
    "burning_rate_constant_mm2_s": ("burn rate K", "燃烧速率 K"),
    "extinction_label": ("flame out?", "是否熄火？"),
}
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


# Readable headers for the raw NASA tables (Evidence tab). NASA's own header stays in each column's tooltip.
# Species and the burning-rate unit follow NASA/TP-2015-216046: the FLEX CSV prints 'O', 'N', 'CO' and 'mm'.
RAW_COLUMN_LABELS = {
    "pi": ("Investigator", "研究员"),
    "test_number": ("Test #", "试验编号"),
    "date": ("Date", "日期"),
    "gmt": ("Time (GMT)", "时间 (GMT)"),
    "sample_number": ("Sample #", "样品编号"),
    "fuel_sample_material": ("Material", "材料"),
    "flow_restrictor": ("Flow restrictor", "流量限制器"),
    "fan_display": ("Fan display", "风扇显示值"),
    "air_display": ("Air display", "气流显示值"),
    "radiometer_gain_1000_250_50_10": ("Radiometer gain", "辐射计增益"),
    "total_frames_shot": ("Frames shot", "拍摄帧数"),
    "calibrated_initial_o2_pct_by_vol": ("O₂ start (% vol)", "初始 O₂ (% vol)"),
    "calibrated_final_o2_pct_by_vol": ("O₂ end (% vol)", "结束 O₂ (% vol)"),
    "initial_co2_pct_by_vol": ("CO₂ start (% vol)", "初始 CO₂ (% vol)"),
    "final_co2_pct_by_vol": ("CO₂ end (% vol)", "结束 CO₂ (% vol)"),
    "initial_co_ppm": ("CO start (ppm)", "初始 CO (ppm)"),
    "final_co_ppm": ("CO end (ppm)", "结束 CO (ppm)"),
    "flex_test_number": ("Test #", "试验编号"),
    "flex_identifier": ("FLEX ID", "FLEX 编号"),
    "test_date": ("Date", "日期"),
    "test_gmt": ("Time (GMT)", "时间 (GMT)"),
    "fuel": ("Fuel", "燃料"),
    "ambient_pressure_mmhg": ("Pressure (mmHg)", "压力 (mmHg)"),
    "o_initial_ambient_composition_mole_fraction": ("O₂ (mole fraction)", "O₂（摩尔分数）"),
    "n_initial_ambient_composition_mole_fraction": ("N₂ (mole fraction)", "N₂（摩尔分数）"),
    "co_initial_ambient_composition_mole_fraction": ("CO₂ (mole fraction)", "CO₂（摩尔分数）"),
    "he_initial_ambient_composition_mole_fraction": ("He (mole fraction)", "He（摩尔分数）"),
    "droplet_initial_diameter_mm": ("Droplet Ø start (mm)", "初始液滴直径 (mm)"),
    "visible_flame_extinction_diameter_mm": ("Droplet Ø at flame-out (mm)", "熄火时直径 (mm)"),
    "burning_rate_mm": ("Burning rate K (mm²/s)", "燃烧速率 K (mm²/s)"),
    "burn_time_s": ("Burn time (s)", "燃烧时间 (s)"),
    "test_end": ("How the test ended", "试验结束方式"),
}
# Provenance columns repeat the same file, DOI and licence on every row, so the raw table hides them unless asked.
RAW_SOURCE_LABELS = {
    "record_id": ("Record ID", "记录 ID"),
    "source_row_number": ("Row in NASA file", "NASA 文件中的行号"),
    "source_file": ("Source file", "来源文件"),
    "source_doi": ("DOI", "DOI"),
    "source_license_status": ("Licence", "许可"),
    "source_url": ("Source URL", "来源链接"),
    "dataset_name": ("Dataset", "数据集"),
    "dataset_key": ("Dataset key", "数据集代码"),
}


def short_label(column: str) -> str:
    en, zh = FEATURE_SHORT.get(column, (column, column))
    return tr(en, zh)


def _chips(x: float, ys: list, labels: list[str], dim: bool = False, w: int = 64) -> str:
    out = []
    for y, name in zip(ys, labels):
        op = ' opacity="0.35"' if dim else ""
        out.append(f'<g{op}><rect class="chip" x="{x}" y="{y - 9}" width="{w}" height="18" rx="9"/>'
                   f'<text class="lbl" x="{x + w / 2}" y="{y + 3.5}" text-anchor="middle">{name}</text></g>')
        if dim:
            out.append(f'<line x1="{x + 5}" y1="{y}" x2="{x + w - 5}" y2="{y}" stroke="#FC3D21" stroke-width="1.2" opacity="0.7"/>')
    return "".join(out)


def _in_ys(n: int) -> list[float]:
    step = 32 if n <= 4 else 27
    return [108 + (i - (n - 1) / 2) * step for i in range(n)]


def baseline_svg(inputs: list[str], output: str, classification: bool = False) -> str:
    """Baseline: ignores every input and always predicts the training average (or most common class)."""
    if classification:
        ys = _in_ys(len(inputs))
        return (_SVG_HEAD + _SVG_CSS + _chips(6, ys, inputs, dim=True)
                + '<path d="M72 110 H104" stroke="#3a4670" stroke-width="1.5" stroke-dasharray="4 3"/>'
                + '<rect x="104" y="88" width="88" height="44" rx="10" fill="#151c3a" stroke="#8f9ab8"/>'
                + f'<text class="lbl" x="148" y="107" text-anchor="middle" font-weight="700">{tr("MOST COMMON", "最常见结果")}</text>'
                + f'<text class="mut" x="148" y="121" text-anchor="middle">{tr("same answer", "答案不变")}</text>'
                + '<path d="M192 110 H222" stroke="#FF9A4D" stroke-width="2"/>'
                + f'<rect class="chip" x="222" y="100" width="72" height="20" rx="10"/><text class="lbl" x="258" y="114" text-anchor="middle">{output}</text>'
                + f'<text class="mut" x="6" y="208">{tr("Inputs ignored ✕ · never predicts the rare outcome", "忽略全部输入 ✕ · 从不预测少见结果")}</text>'
                + "</svg>")
    ys = _in_ys(len(inputs))
    truth = [(205, 62), (220, 140), (234, 88), (248, 158), (262, 74), (276, 120), (290, 98)]
    resid = "".join(
        f'<line x1="{x}" y1="{y}" x2="{x}" y2="110" stroke="#FC3D21" stroke-width="1.6" stroke-dasharray="3 2" '
        f'opacity="0.8"><animate attributeName="opacity" values="0.2;0.9;0.2" dur="2.4s" begin="{i * 0.25:.2f}s" repeatCount="indefinite"/></line>'
        for i, (x, y) in enumerate(truth))
    dots = "".join(f'<circle cx="{x}" cy="{y}" r="4" fill="#7CC4FF"/>' for x, y in truth)
    return (_SVG_HEAD + _SVG_CSS + _chips(6, ys, inputs, dim=True)
            + '<path d="M72 110 H104" stroke="#3a4670" stroke-width="1.5" stroke-dasharray="4 3"/>'
            + '<rect x="104" y="90" width="70" height="40" rx="10" fill="#151c3a" stroke="#8f9ab8"/>'
            + f'<text class="lbl" x="139" y="108" text-anchor="middle" font-weight="700">{tr("AVERAGE", "平均值")}</text>'
            + f'<text class="mut" x="139" y="121" text-anchor="middle">{tr("same answer", "答案不变")}</text>'
            + '<path d="M174 110 H300" stroke="#FF9A4D" stroke-width="2.2" stroke-dasharray="6 4"/>'
            + resid + dots
            + f'<text class="mut" x="6" y="204">{tr("Inputs ignored ✕", "忽略全部输入 ✕")}</text>'
            + f'<text class="mut" x="190" y="204">{tr("● measured  — guess  ┆ error", "● 实测  — 猜测  ┆ 误差")}</text>'
            + "</svg>")


def ridge_svg(inputs: list[str], output: str, logistic: bool = False) -> str:
    """Ridge (or logistic) regression: one weight per input, summed; logistic squashes it into a probability."""
    ys = _in_ys(len(inputs))
    subs = "₁₂₃₄₅₆₇₈₉"
    parts = [_SVG_HEAD, _SVG_CSS, _chips(6, ys, inputs)]
    for i, y in enumerate(ys):
        parts.append(f'<line class="edge" x1="70" y1="{y}" x2="114" y2="110" style="stroke-width:1.6"/>')
        parts.append(f'<line class="pulse" x1="70" y1="{y}" x2="114" y2="110" style="animation-delay:{-0.4 * i:.1f}s"/>')
    # Dark box with a pulsing outline (not a pulsing fill), so the text inside always stays readable.
    parts.append('<rect x="114" y="86" width="98" height="48" rx="10" fill="#151c3a" stroke="#FF9A4D" stroke-width="2">'
                 '<animate attributeName="stroke-opacity" values="0.35;1;0.35" dur="1.8s" repeatCount="indefinite"/></rect>')
    formula = "σ(Σ wᵢxᵢ + b)" if logistic else "Σ wᵢxᵢ + b"
    parts.append(f'<text class="lbl" x="163" y="107" text-anchor="middle" font-weight="700">{formula}</text>')
    parts.append(f'<text class="mut" x="163" y="122" text-anchor="middle">'
                 f'{tr("probability", "概率") if logistic else tr("weighted sum", "加权求和")}</text>')
    parts.append('<path d="M212 110 H226" stroke="#FF9A4D" stroke-width="2"/>')
    parts.append(f'<rect class="chip" x="226" y="100" width="70" height="20" rx="10"/><text class="lbl" x="261" y="114" text-anchor="middle">{output}</text>')
    terms = " + ".join(f"w{subs[i]}x{subs[i]}" for i in range(min(len(inputs), 4))) + (" + …" if len(inputs) > 4 else "")
    parts.append(f'<text class="mut" x="150" y="196" text-anchor="middle">{terms} + b</text>')
    note = (tr("C picked on inner validation; classes weighted equally", "C 由内层验证选出；两类权重均衡") if logistic
            else tr("penalty keeps weights small (alpha picked on inner validation)", "惩罚项使权重保持较小（alpha 由内层验证选出）"))
    parts.append(f'<text class="mut" x="150" y="210" text-anchor="middle">{note}</text>')
    parts.append("</svg>")
    return "".join(parts)


def forest_svg(inputs: list[str], output: str, classification: bool = False) -> str:
    """Random forest: many decision trees answer yes/no questions, then their answers are averaged (or voted)."""
    ys = _in_ys(len(inputs))
    parts = [_SVG_HEAD, _SVG_CSS, _chips(6, ys, inputs)]
    leaf_pts = []
    for t, cx in enumerate((124, 186, 248)):
        root, kids = (cx, 40), [(cx - 16, 74), (cx + 16, 74)]
        leaves = [(cx - 24, 108), (cx - 8, 108), (cx + 8, 108), (cx + 24, 108)]
        for k, (kx, ky) in enumerate(kids):
            parts.append(f'<line class="edge" x1="{root[0]}" y1="{root[1]}" x2="{kx}" y2="{ky}"/>')
            for lx, ly in leaves[2 * k:2 * k + 2]:
                parts.append(f'<line class="edge" x1="{kx}" y1="{ky}" x2="{lx}" y2="{ly}"/>')
        kid = kids[t % 2]
        leaf = leaves[(t % 2) * 2 + (1 if t == 1 else 0)]
        parts.append(f'<polyline points="{root[0]},{root[1]} {kid[0]},{kid[1]} {leaf[0]},{leaf[1]}" fill="none" '
                     f'stroke="#FF9A4D" stroke-width="2.4" stroke-linecap="round" opacity="0">'
                     f'<animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.2;0.7;1" dur="3s" begin="{t * 0.6:.1f}s" repeatCount="indefinite"/></polyline>')
        for (nx, ny) in [root, *kids, *leaves]:
            parts.append(f'<circle cx="{nx}" cy="{ny}" r="4.5" fill="#2a3560" stroke="#6f7ca8"/>')
        leaf_pts.append(leaf)
    for lx, ly in leaf_pts:
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="186" y2="150" stroke="rgba(255,154,77,.45)" stroke-width="1.2"/>')
    parts.append(f'<text class="mut" x="186" y="20" text-anchor="middle">{tr("3 of the 200 trees shown", "显示 200 棵树中的 3 棵")}</text>')
    parts.append('<circle class="glow" cx="186" cy="150" r="10" stroke="#FF9A4D"/>')
    parts.append(f'<text class="lbl" x="202" y="154">'
                 f'{tr("trees vote", "树木投票") if classification else tr("average answers", "汇总答案")}</text>')
    parts.append('<path d="M186 160 L186 184" stroke="#FF9A4D" stroke-width="1.5"/>')
    parts.append(f'<rect class="chip" x="151" y="184" width="70" height="20" rx="10"/><text class="lbl" x="186" y="198" text-anchor="middle">{output}</text>')
    parts.append("</svg>")
    return "".join(parts)


def network_svg(inputs: list[str], output: str) -> str:
    """The deployed network: inputs -> 32 -> 16 -> 1 (drawn with fewer neurons than it has)."""
    layers_x = (96, 156, 214, 268)
    ys_in = _in_ys(len(inputs))
    ys_h1 = [30 + i * 22 for i in range(8)]   # 8 drawn of 32
    ys_h2 = [55 + i * 22 for i in range(6)]   # 6 drawn of 16
    ys_out = [110]
    parts = [_SVG_HEAD, _SVG_CSS]
    cols = [(layers_x[0], ys_in), (layers_x[1], ys_h1), (layers_x[2], ys_h2), (layers_x[3], ys_out)]
    for (xa, ya), (xb, yb) in zip(cols, cols[1:]):
        for y1 in ya:
            for y2 in yb:
                parts.append(f'<line class="edge" x1="{xa}" y1="{y1}" x2="{xb}" y2="{y2}"/>')
    pulse_edges = [(0, 0, 2), (0, 1, 5), (0, len(ys_in) - 1, 7), (1, 1, 1), (1, 4, 3), (1, 6, 5), (2, 0, 0), (2, 3, 0), (2, 5, 0)]
    for i, (layer, a, b) in enumerate(pulse_edges):
        (xa, ya), (xb, yb) = cols[layer], cols[layer + 1]
        if a < len(ya) and b < len(yb):
            parts.append(f'<line class="pulse" x1="{xa}" y1="{ya[a]}" x2="{xb}" y2="{yb[b]}" style="animation-delay:{-0.17 * i:.2f}s"/>')
    for layer, (x, ys) in enumerate(cols):
        for y in ys:
            delay = f' style="animation-delay:{0.45 * layer:.2f}s"' if layer else ""
            parts.append(f'<circle class="glow" cx="{x}" cy="{y}" r="6" stroke="#6f7ca8"{delay}/>')
    for y, name in zip(ys_in, inputs):
        parts.append(f'<text class="lbl" x="{layers_x[0] - 10}" y="{y + 3.5}" text-anchor="end">{name}</text>')
    parts.append(f'<text class="mut" x="{layers_x[1]}" y="208" text-anchor="middle">{tr("32 neurons", "32 个神经元")}</text>')
    parts.append(f'<text class="mut" x="{layers_x[2]}" y="208" text-anchor="middle">{tr("16 neurons", "16 个神经元")}</text>')
    parts.append(f'<text class="lbl" x="{layers_x[3]}" y="{ys_out[0] - 14}" text-anchor="middle">{output}</text>')
    parts.append("</svg>")
    return "".join(parts)


MODEL_ART = {
    "mean_baseline": baseline_svg, "ridge": ridge_svg, "random_forest": forest_svg, "tf_nn_deployed_config": network_svg,
    "majority_baseline": lambda i, o: baseline_svg(i, o, classification=True),
    "logistic_regression": lambda i, o: ridge_svg(i, o, logistic=True),
    "random_forest_classifier": lambda i, o: forest_svg(i, o, classification=True),
}
MODEL_LABELS = {
    "mean_baseline": ("🎯 Baseline", "🎯 基线模型"),
    "ridge": ("📏 Ridge (linear)", "📏 岭回归（线性）"),
    "random_forest": ("🌳 Random forest", "🌳 随机森林"),
    "tf_nn_deployed_config": ("🧠 Neural network (deployed)", "🧠 神经网络（已部署）"),
    "majority_baseline": ("🎯 Baseline", "🎯 基线模型"),
    "logistic_regression": ("📈 Logistic regression", "📈 逻辑回归"),
    "random_forest_classifier": ("🌳 Random forest", "🌳 随机森林"),
}
MODEL_HOW = {
    "mean_baseline": ("Ignores every input and always predicts the average peak seen in training. Any useful model has to beat it.",
                      "忽略全部输入，始终预测训练集中的平均峰值。任何有用的模型都必须胜过它。"),
    "ridge": ("Multiplies each input by a weight and adds them up, with a penalty that keeps the weights small. The penalty "
              "strength is chosen on inner-validation labs only.",
              "将每个输入乘以权重后相加，并用惩罚项使权重保持较小。惩罚强度只在内层验证实验室上选择。"),
    "random_forest": ("200 decision trees each ask yes/no questions about the inputs and give an answer; the forest averages "
                      "them. The minimum leaf size is chosen on inner-validation labs only.",
                      "200 棵决策树分别对输入提出是/否问题并给出答案，森林再取平均。最小叶节点大小只在内层验证实验室上选择。"),
    "tf_nn_deployed_config": ("The exact configuration of the deployed model (32 and 16 neurons, 60 epochs), retrained inside "
                              "every training partition. It is not tuned, so this measures the deployed recipe as it is.",
                              "与已部署模型完全相同的配置（32 与 16 个神经元，60 轮），在每个训练分区内重新训练。未做调参，"
                              "因此衡量的就是已部署方案本身。"),
    "majority_baseline": ("Ignores every input and always predicts the outcome that was most common in training "
                          "(here: the flame goes out). Any useful model has to beat it.",
                          "忽略全部输入，始终预测训练中最常见的结果（此处为“火焰熄灭”）。任何有用的模型都必须胜过它。"),
    "logistic_regression": ("Adds up weighted inputs and turns the total into a probability that the flame goes out. "
                            "Rare outcomes get equal weight, and the penalty strength is chosen on inner-validation days only.",
                            "将加权输入求和并转换为火焰熄灭的概率。少见结果获得同等权重，惩罚强度只在内层验证日期上选择。"),
    "random_forest_classifier": ("200 decision trees vote on whether the flame goes out, with rare outcomes weighted up. "
                                 "The minimum leaf size is chosen on inner-validation days only.",
                                 "200 棵决策树对火焰是否熄灭进行投票，少见结果的权重被提高。最小叶节点大小只在内层验证日期上选择。"),
}


def _ring(label: str, stats: dict) -> str:
    mean, std = stats.get("mean"), stats.get("std")
    if mean is None:
        return f'<div class="ring-wrap"><div class="ring"><div class="ring-in"><b>—</b></div></div><div class="ring-lbl">{label}</div></div>'
    pct = max(0.0, min(1.0, mean)) * 100
    return (f'<div class="ring-wrap"><div class="ring" style="background:conic-gradient(#d95926 {pct:.1f}%, rgba(255,255,255,.08) 0)">'
            f'<div class="ring-in"><b>{mean:.2f}</b><span>± {std:.2f}</span></div></div><div class="ring-lbl">{label}</div></div>')


def classification_panel_html(m: dict, base: dict, how: str, is_base: bool, minority: str) -> str:
    """Panel for classification tasks: fold-level balanced accuracy, ROC-AUC and minority-class recall."""
    fl = m["fold_level"]
    ba, bba, auc, rec = fl["balanced_accuracy"], base["fold_level"]["balanced_accuracy"], fl["roc_auc"], fl["recall_neg"]
    bar = (f'<div class="eb"><div class="eb-top"><span>{tr("Balanced accuracy (25 folds)", "平衡准确率（25 折）")}</span>'
           f'<b>{ba["mean"]:.2f} ± {ba["std"]:.2f}</b></div>'
           f'<div class="eb-row"><i class="eb-this" style="width:{100 * ba["mean"]:.0f}%"></i></div>'
           f'<div class="eb-row"><i class="eb-base" style="width:{100 * bba["mean"]:.0f}%"></i></div>'
           f'<div class="eb-key"><span class="k1"></span>{tr("this model", "本模型")}<span class="k2"></span>{tr("baseline", "基线")}'
           f'&nbsp;·&nbsp;{tr("0.5 = guessing", "0.5 = 瞎猜")}</div></div>')
    if is_base:
        did = tr("The yardstick: it is right about the common outcome and always wrong about the rare one, so balanced accuracy is 0.5.",
                 "参照标准：它对常见结果总是正确，对少见结果总是错误，因此平衡准确率为 0.5。")
    else:
        did = tr(f"On test days it never saw: balanced accuracy {ba['mean']:.2f} ± {ba['std']:.2f}, ROC-AUC {auc['mean']:.2f} ± {auc['std']:.2f}. "
                 f"It caught {rec['mean'] * 100:.0f}% of the rare '{minority}' outcomes on average.",
                 f"在从未见过的试验日上：平衡准确率 {ba['mean']:.2f} ± {ba['std']:.2f}，ROC-AUC {auc['mean']:.2f} ± {auc['std']:.2f}。"
                 f"平均识别出 {rec['mean'] * 100:.0f}% 的少见“{minority}”结果。")
    return ('<div class="mpanel"><div class="rings" style="grid-template-columns:repeat(2,1fr)">'
            + _ring(tr("Balanced accuracy", "平衡准确率"), ba) + _ring("ROC-AUC", auc)
            + "</div>" + bar
            + f'<p class="mnote"><b>{tr("How it works", "工作原理")}</b> {how}</p>'
            + f'<p class="mnote"><b>{tr("How it did", "表现如何")}</b> {did}</p></div>')


def model_panel_html(m: dict, base: dict, how: str, unit: str, is_base: bool) -> str:
    """Right-hand panel: R² rings, MAE versus the baseline, and plain-language notes, all from cv_summary.json."""
    fl, ps = m["fold_level"], m["per_seed_pooled_summary"]

    def ring(label: str, stats: dict) -> str:
        mean, std = stats.get("mean"), stats.get("std")
        if mean is None:
            return f'<div class="ring-wrap"><div class="ring"><div class="ring-in"><b>—</b></div></div><div class="ring-lbl">{label}</div></div>'
        pct = max(0.0, min(1.0, mean)) * 100
        return (f'<div class="ring-wrap"><div class="ring" style="background:conic-gradient(#d95926 {pct:.1f}%, rgba(255,255,255,.08) 0)">'
                f'<div class="ring-in"><b>{mean:.2f}</b><span>± {std:.2f}</span></div></div><div class="ring-lbl">{label}</div></div>')

    mae, bmae = fl["mae"], base["fold_level"]["mae"]
    scale = max(mae["mean"], bmae["mean"])
    digits = 0 if bmae["mean"] >= 10 else 3
    num = lambda x: f"{x:.{digits}f}"  # noqa: E731 - keeps the panel readable for kW/m2 and mm2/s alike
    err = (f'<div class="eb"><div class="eb-top"><span>{tr("Mean absolute error (25 folds)", "平均绝对误差（25 折）")}</span>'
           f'<b>{num(mae["mean"])} ± {num(mae["std"])} {unit}</b></div>'
           f'<div class="eb-row"><i class="eb-this" style="width:{100 * mae["mean"] / scale:.0f}%"></i></div>'
           f'<div class="eb-row"><i class="eb-base" style="width:{100 * bmae["mean"] / scale:.0f}%"></i></div>'
           f'<div class="eb-key"><span class="k1"></span>{tr("this model", "本模型")}<span class="k2"></span>{tr("baseline", "基线")}'
           f'&nbsp;·&nbsp;{tr("range", "范围")} {num(mae["min"])}–{num(mae["max"])} · RMSE {num(fl["rmse"]["mean"])}</div></div>')
    if is_base:
        did = tr(f"The yardstick: MAE {num(mae['mean'])} {unit} and R² near or below 0 mean no skill beyond the average.",
                 f"参照标准：MAE {num(mae['mean'])} {unit}，R² 接近或低于 0，表示除平均值外没有预测能力。")
    else:
        drop = (1 - mae["mean"] / bmae["mean"]) * 100
        did = tr(f"On groups it never saw, MAE was {num(mae['mean'])} ± {num(mae['std'])} {unit}, {drop:.0f}% lower than the "
                 f"baseline. Pooled over each repeat, R² = {ps['r2']['mean']:.2f} ± {ps['r2']['std']:.2f}.",
                 f"在从未见过的分组上，MAE 为 {num(mae['mean'])} ± {num(mae['std'])} {unit}，比基线低 {drop:.0f}%。"
                 f"按每次重复汇总，R² = {ps['r2']['mean']:.2f} ± {ps['r2']['std']:.2f}。")
    return ('<div class="mpanel"><div class="rings" style="grid-template-columns:repeat(2,1fr)">'
            + ring(tr("R² per fold", "每折 R²"), fl["r2"])
            + ring(tr("R² pooled per repeat", "按重复汇总的 R²"), ps["r2"])
            + "</div>" + err
            + f'<p class="mnote"><b>{tr("How it works", "工作原理")}</b> {how}</p>'
            + f'<p class="mnote"><b>{tr("How it did", "表现如何")}</b> {did}</p></div>')


TAB_LABELS = {
    "evidence": ("🔬 Evidence Explorer", "🔬 航天实验证据库 (Evidence)"),
    "stem": ("📈 STEM Analysis", "📈 科研变量分析与物理叙事 (STEM Analysis)"),
    "sandbox": ("🧪 2D Physics Sandbox", "🧪 2D 物理时空模拟沙盒 (Physics Sandbox)"),
    "audit": ("📊 Scientific Audit", "📊 科学审计与 25 折验证 (Scientific Audit)"),
    "ai": ("🤖 AI Predictions", "🤖 AI 预测"),
}


@st.cache_data(show_spinner=False)
def cached_insights(lang_code: str) -> list:
    return crew_insights(lang_code)


def insights_html(items: list) -> str:
    """One card per takeaway: what it means, the computed evidence, what to do, and where to check it."""
    cards = []
    for it in items:
        evidence = it["evidence"]
        if it["dataset"] and ds_status(it["dataset"])["status"] != "VERIFIED_SOURCE":
            evidence = ds_status(it["dataset"])["status"]  # never call unverified data a real measurement
        cards.append(
            f'<div class="icard"><div class="ihead"><span class="iicon">{it["icon"]}</span>{html.escape(it["title"])}</div>'
            f'<p>{html.escape(it["text"])}</p><p class="iact">➜ {html.escape(it["action"])}</p>'
            f'<div class="ifoot">{status_pill(evidence)}<span>{html.escape(tr("Evidence:", "证据："))} '
            f'{html.escape(tr(*TAB_LABELS[it["tab"]]))}</span></div></div>')
    return f'<div class="insights">{"".join(cards)}</div>'


st.markdown(tr("### 🧑‍🚀 What this means for crew fire safety", "### 🧑‍🚀 这对乘员防火安全意味着什么"))
st.caption(tr("Six takeaways. Every number is computed live from NASA's data, the AI models or the physics model; the badge "
              "says which kind of evidence it is, and the tab name says where to check it.",
              "六条要点。每个数字都由 NASA 数据、AI 模型或物理模型实时计算；标签说明证据类型，页面名称说明在哪里核查。"))
st.markdown(insights_html(cached_insights(lang)), unsafe_allow_html=True)

tab_evidence, tab_sweep, tab_simulation, tab_audit, tab_prediction = st.tabs(
    [tr(*TAB_LABELS[k]) for k in ("evidence", "stem", "sandbox", "audit", "ai")])

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
        prov = SOURCES.get("datasets", {}).get(ds_key, {})  # static facts only
        computed = ds_status(ds_key)                          # status computed by src/provenance_audit.py
        is_placeholder = computed["status"] != "VERIFIED_SOURCE"
        st.info(tr(
            f"📌 **Current track**: {track_mode} | **NASA dataset**: DOI `{doi}` | [Open NASA data repository]({source_url})"
            f" | **Data shown here:** {status_badge(computed['status'])} (see Data provenance below)",
            f"📌 **当前实验轨道**: {track_mode} | **NASA 数据集**: DOI `{doi}` | [访问 NASA 官方实验库]({source_url})"
            f" | **此处显示：** {status_badge(computed['status'])}（见下方数据来源）",
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
                + (", which is not verified NASA data in this build, so the ranking itself is not a scientific result. "
                   if is_placeholder else ". ")
                + ("Each tiny droplet sits inside its olive-sized spherical flame and shrinks until it reaches the "
                   "average diameter at which the flame went out. K is the d²-law burning-rate constant (mm²/s, "
                   "NASA/TP-2015-216046); 'flame went out' counts only tests that ended by extinction or completion, "
                   "because disrupted tests ended for another reason." if track_key == "liquid_droplet" else
                   "Side view inside the flow duct: in microgravity the flame is a blue envelope lying along the airflow, "
                   "and smoke is carried downstream rather than rising. Final CO is the chamber reading after each test; "
                   "it depends on sample size, airflow and burn time, so this ranks the tests as they were run, not each "
                   "material's intrinsic toxicity. Over-range readings are counted but not averaged."),
                "动画示意图，参照 NASA 在轨照片绘制（非真实影像）。大小、颜色与速度由已载入数据表计算的平均值驱动"
                + ("；本版本该表并非已核实的 NASA 数据，因此排名本身不是科学结论。" if is_placeholder else "。")
                + ("每个微小液滴位于橄榄大小的球形火焰中，逐渐缩小至火焰熄灭时的平均直径。K 为 d² 定律燃烧速率常数（mm²/s，"
                   "NASA/TP-2015-216046）；“火焰熄灭比例”只统计以熄灭或燃尽结束的试验，因为液滴破碎的试验是由其他原因结束的。"
                   if track_key == "liquid_droplet" else
                   "流道内侧视图：微重力下火焰是沿气流方向的蓝色包络，烟雾随气流向下游飘散而不会上升。最终 CO 是每次试验后舱内的读数，"
                   "受样品尺寸、气流与燃烧时长影响，因此该排名反映的是试验实际情况，而非材料本身的毒性。超量程读数会被计数，但不计入平均。"),
            ))
            st.markdown(ranking_cards_html(track_key, ranking_data), unsafe_allow_html=True)
            with st.expander(tr("Show ranking as a data table", "以数据表查看排名")):
                st.dataframe(pd.DataFrame(ranking_data), use_container_width=True, hide_index=True)

            insight = ranking_insights(track_key, ranking_data, lang)
            if insight["points"]:
                st.markdown(tr("#### What the ranking means", "#### 排名说明了什么"))
                st.markdown("\n".join(f"- {p}" for p in insight["points"]))
                if insight["safety"]:
                    st.info("💡 " + insight["safety"])
                st.caption(tr("Every number above is computed from the loaded table. ", "以上数字均由已载入的数据表计算。")
                           + (tr("That table is not verified NASA data in this build, so these are not scientific findings. ",
                                 "本版本该表并非已核实的 NASA 数据，因此这些并非科学结论。") if is_placeholder else "")
                           + insight["caveat"])

        st.markdown(tr("#### Data provenance", "#### 数据来源"))
        full_track = records_df[records_df["dataset_key"] == ds_key]
        report_entry = (PROV_REPORT or {}).get("datasets", {}).get(ds_key, {})
        st.markdown(
            f"**{cfg_text(prov, 'display_name', ds_key)}** · {status_badge(computed['status'])}\n\n"
            + tr(f"- **Rows loaded:** {len(full_track)} (computed from `data/processed/dashboard_records.csv`)\n",
                 f"- **已载入行数：** {len(full_track)}（由 `data/processed/dashboard_records.csv` 计算）\n")
            + (tr("- **Source:** ", "- **来源：** ") if not is_placeholder
               else tr("- **Where real data should come from:** ", "- **真实数据应来自：** "))
            + f"[{prov.get('real_source_page', '')}]({prov.get('real_source_page', '')}) · DOI `{prov.get('real_source_doi', '')}` · {cfg_text(prov, 'real_source_license')}\n"
            + "\n"
            + tr("**Evidence** (computed by `src/provenance_audit.py`"
                 + (f", {PROV_REPORT.get('generated_at_utc')}" if PROV_REPORT else "") + "):\n",
                 "**证据**（由 `src/provenance_audit.py` 计算"
                 + (f"，{PROV_REPORT.get('generated_at_utc')}" if PROV_REPORT else "") + "）：\n")
            + ("".join(f"- {code_text(e)}\n" for e in report_entry.get("evidence", []))
               or f"- {computed.get('warning') or ''}\n")
        )
        if report_entry.get("required_to_verify"):
            with st.expander(tr("What is needed to verify this dataset", "核实该数据集还需要什么")):
                st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(cfg_text(report_entry, "required_to_verify", []), 1)))

        catalog_path = ROOT / "data" / "processed" / "field_catalog.csv"
        catalog = pd.read_csv(catalog_path) if catalog_path.exists() else pd.DataFrame()
        catalog = catalog[catalog.get("dataset_key", pd.Series(dtype=str)) == ds_key] if not catalog.empty else catalog
        with st.expander(tr("🔎 Record inspector: trace one row back to its source", "🔎 记录溯源：追踪单条记录的来源")):
            if full_track.empty:
                st.caption(tr("No rows for this track.", "该轨道无数据。"))
            else:
                rec_id = st.selectbox(tr("Choose a record", "选择记录"), full_track["record_id"].tolist(), key=f"inspect_{ds_key}")
                row = full_track[full_track["record_id"] == rec_id].iloc[0]
                mapping = load_provenance_mapping(str(REPORTS_DIR / "provenance_mapping.csv"))
                mrow = mapping[mapping["internal_record_id"] == rec_id] if not mapping.empty else mapping
                rec_status = mrow["verification_status"].iloc[0] if len(mrow) else "UNVERIFIED_SOURCE"
                st.markdown(
                    tr(f"**Verification:** {status_badge(rec_status)} · **Source file:** `{row.get('source_file', '')}` · "
                       f"**row** {row.get('source_row_number', '')} · **licence field:** {row.get('source_license_status', '')}",
                       f"**核实状态：** {status_badge(rec_status)} · **来源文件：** `{row.get('source_file', '')}` · "
                       f"**第** {row.get('source_row_number', '')} **行** · **许可字段：** {row.get('source_license_status', '')}")
                )
                if len(mrow):
                    st.caption(str(cfg_text(mrow.iloc[0].to_dict(), "explanation")))
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
                              "unchanged. Files under data/demo/ never receive the real investigation's DOI, page or licence. "
                              "Full record-by-record mapping: reports/provenance_mapping.csv.",
                              "处理记录（src/build_dashboard_data.py）：列名统一为 snake_case；数值原样复制。data/demo/ 下的文件不会被附上"
                              "真实实验的 DOI、页面或许可。逐条记录映射见 reports/provenance_mapping.csv。"))

        st.markdown(tr("#### Raw table as loaded", "#### 已载入的原始数据表"))
        # The shared CSV holds both tracks, so drop the columns that are empty for this one.
        filled = [c for c in full_track.columns if full_track[c].notna().any()]
        data_cols = [c for c in filled if c not in RAW_SOURCE_LABELS]
        label = (tr(f"📋 {len(track_df)} of {len(full_track)} rows match “{search_query}”",
                    f"📋 {len(full_track)} 行中有 {len(track_df)} 行匹配“{search_query}”") if search_query
                 else tr(f"📋 Show all {len(full_track)} rows · {len(data_cols)} measured columns",
                         f"📋 查看全部 {len(full_track)} 行 · {len(data_cols)} 个测量列"))
        with st.expander(label, expanded=bool(search_query)):  # the search box above filters this table, so open it then
            show_source = st.toggle(tr("Show source columns (record ID, file row, DOI, licence)",
                                       "显示来源列（记录 ID、文件行号、DOI、许可）"), key=f"raw_source_{ds_key}")
            shown = data_cols + ([c for c in RAW_SOURCE_LABELS if c in filled] if show_source else [])
            nasa_header = dict(zip(catalog.get("field_name", []), catalog.get("original_field_name", [])))
            config = {c: st.column_config.Column(
                tr(*RAW_COLUMN_LABELS.get(c, RAW_SOURCE_LABELS.get(c, (c, c)))),
                help=tr(f"NASA column: “{' '.join(str(nasa_header[c]).split())}”",
                        f"NASA 原始列名：“{' '.join(str(nasa_header[c]).split())}”") if c in nasa_header else None,
            ) for c in shown}
            if "source_url" in shown:
                config["source_url"] = st.column_config.LinkColumn(tr(*RAW_SOURCE_LABELS["source_url"]))
            if track_df.empty:
                st.caption(tr("No rows match the search.", "没有匹配搜索的行。"))
            else:
                st.dataframe(track_df, use_container_width=True, hide_index=True, column_order=shown, column_config=config)
            st.caption(tr("Values are exactly as in NASA's file. Hover a header to see NASA's original column name.",
                          "数值与 NASA 文件完全一致。将鼠标悬停在列名上可查看 NASA 原始列名。")
                       + (tr(" Gas names and the burning-rate unit follow NASA/TP-2015-216046: the file prints 'O', 'N', "
                             "'CO' and 'mm' for O₂, N₂, CO₂ and mm²/s.",
                             " 气体名称与燃烧速率单位依据 NASA/TP-2015-216046：文件中的 'O'、'N'、'CO' 与 'mm' 分别指 O₂、N₂、CO₂ 与 mm²/s。")
                          if ds_key == "FLEX" else ""))


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
    st.caption(status_badge("SIMULATED_RESULT") + tr(
        " · computed from scaling formulas in app/services/research_engine.py; not measured data and not a trained model. "
        "The flame-spread index is a relative proxy without physical units.",
        " · 由 app/services/research_engine.py 中的尺度公式计算；不是测量数据，也不是训练出的模型。火焰蔓延指数是没有物理单位的相对代理量。"))

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

    st.markdown(tr("##### 3. What the curves show", "##### 3. 曲线说明了什么"))
    st.success("📘 " + tr("**Read from this sweep** (the numbers change when you move the sliders):",
                          "**从本次扫描读出**（移动滑块后数字会随之改变）：")
               + "\n" + "\n".join(f"- {text}" for text in sweep_results["scientific_narration"]))
    if sweep_results.get("safety_note"):
        st.info("💡 " + sweep_results["safety_note"])

    st.markdown(tr("##### 4. Solved sweep points", "##### 4. 连续参数求解点明细"))
    with st.expander(tr(f"📋 Show the {len(points_df)} solved points as a table", f"📋 以表格查看全部 {len(points_df)} 个求解点")):
        spread_max = float(points_df["predicted_flame_intensity_proxy"].max())
        st.dataframe(
            points_df, use_container_width=True, hide_index=True,
            column_order=["sweep_value", "flow_regime", "predicted_flame_intensity_proxy", "log_richardson_number",
                          "buoyancy_velocity_scale_m_s", "oxygen_flux_index", "reynolds_number"],
            column_config={
                "sweep_value": st.column_config.NumberColumn(sweep_target, format="%.3f", help=tr(
                    "The swept variable. The other two are held at the slider values on the left.",
                    "被扫描的自变量；另外两个变量固定为左侧滑块的取值。")),
                "flow_regime": st.column_config.TextColumn(tr("Flow regime", "流动模式")),
                "predicted_flame_intensity_proxy": st.column_config.ProgressColumn(
                    tr("Flame spread index", "火焰蔓延指数"), format="%.3f", min_value=0.0,
                    max_value=spread_max if spread_max > 0 else 1.0, help=tr(
                        "Relative proxy without physical units. The bar is scaled to the largest value in this sweep.",
                        "没有物理单位的相对代理量；条形长度以本次扫描的最大值为满格。")),
                "log_richardson_number": st.column_config.NumberColumn("log₁₀ Ri", format="%.2f", help=tr(
                    "Buoyancy compared with the forced airflow. Below 0 the airflow wins, above 0 buoyancy wins.",
                    "浮力与强迫气流之比。小于 0 时气流主导，大于 0 时浮力主导。")),
                "buoyancy_velocity_scale_m_s": st.column_config.NumberColumn(
                    tr("Buoyancy velocity u_b (m/s)", "浮力速度 u_b (m/s)"), format="%.3f", help=tr(
                        "How fast buoyancy alone would lift the hot gas. Zero without gravity.",
                        "仅靠浮力时热气体上升的速度；没有重力时为 0。")),
                "oxygen_flux_index": st.column_config.NumberColumn(
                    tr("O₂ supply index", "供氧通量指数"), format="%.2f", help=tr(
                        "Relative oxygen delivery to the flame: O₂ % × (airflow + ½ buoyancy velocity). No physical units.",
                        "到达火焰的相对供氧量：O₂ 百分比 ×（气流速度 + ½ 浮力速度），没有物理单位。")),
                "reynolds_number": st.column_config.NumberColumn(
                    tr("Reynolds number Re", "雷诺数 Re"), format="%.0f", help=tr(
                        "Airflow inertia compared with viscosity, over a 2 cm sample.",
                        "以 2 cm 试样为尺度，气流惯性与黏性之比。")),
            },
        )


with tab_sweep:
    render_sweep()


# =========================================================================
# TAB 3: STEM 2D 物理时空模拟沙盒 (动态场)
# =========================================================================
@st.fragment
def render_simulation():
    st.subheader(tr("2D microgravity combustion physics sandbox", "2D 空间微重力燃烧物理沙盒"))
    st.caption(status_badge("SIMULATED_RESULT") + tr(
        " · educational physics model (app/services/simulation_service.py); not NASA imagery, not measured data, not CFD.",
        " · 教学用物理模型（app/services/simulation_service.py）；不是 NASA 影像、不是测量数据，也不是 CFD。"))
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
@st.cache_data(show_spinner=False)
def load_report_csv(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def recheck_manifest(man: pd.DataFrame) -> dict[str, int]:
    """Recompute the leakage checks from the saved split manifest, independently of the pipeline's own audit."""
    out = {"splits": 0, "group_overlaps": 0, "record_conflicts": 0, "groups_not_once_per_repeat": 0}
    if man.empty:
        return out
    for _, s in man.groupby(["seed", "fold"]):
        out["splits"] += 1
        parts = {p: set(s.loc[s["partition"] == p, "group_id"]) for p in ("inner_train", "inner_validation", "outer_test")}
        out["group_overlaps"] += (len(parts["inner_train"] & parts["inner_validation"])
                                  + len(parts["inner_train"] & parts["outer_test"])
                                  + len(parts["inner_validation"] & parts["outer_test"]))
        out["record_conflicts"] += int(s["record_id"].duplicated().sum())
    tests = man[man["partition"] == "outer_test"].drop_duplicates(["seed", "fold", "group_id"])
    per_group = tests.groupby(["seed", "group_id"]).size()
    out["groups_not_once_per_repeat"] = int((per_group != 1).sum())
    n_groups = man["group_id"].nunique()
    for _, n in tests.drop_duplicates(["seed", "group_id"]).groupby("seed")["group_id"].nunique().items():
        out["groups_not_once_per_repeat"] += int(n_groups - n)
    return out


def _unit_display(unit: str) -> str:
    return unit.replace("^2", "²").replace("kW/m2", "kW/m²")


GROUP_NAMES = {"laboratory": ("labs", "个实验室"), "group_test_date": ("test days", "个试验日")}
TASK_DATA = {  # dataset_key -> short name, where the flames burned, pill colour
    "MACFP_PMMA_CONE": (("MaCFP PMMA", "MaCFP PMMA"), ("Ground lab · 1 g", "地面实验室 · 1 g"), "warn"),
    "FLEX": (("NASA FLEX", "NASA FLEX"), ("Microgravity · ISS", "微重力 · 国际空间站"), "info"),
    "BASS_II": (("NASA BASS-II", "NASA BASS-II"), ("Microgravity · ISS", "微重力 · 国际空间站"), "info"),
}


def _md_code(text: str) -> str:
    """Escape text for HTML and turn `code` spans into <code> tags."""
    parts = html.escape(text).split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def deployed_models() -> dict[str, str]:
    """task_id -> the model family the AI Predictions tab runs (only when its saved model file is present)."""
    out = {}
    if (ROOT / "models" / "spacecraft_fire_peak_hrr_v1.keras").exists():
        out["macfp_pmma_peak_hrr"] = "tf_nn_deployed_config"
    for path in (flex.BURNING_RATE_PATH, flex.EXTINCTION_PATH):
        try:
            model = flex.load_model(path)
        except flex.FlexModelUnavailable:
            continue
        out[model["task_id"]] = model["model_family"]
    return out


def task_overview_html(tasks: list[dict], reg_by_id: dict) -> str:
    """One row per prediction task: what it predicts, which data, how much, and both computed statuses."""
    deployed = deployed_models()
    rows = []
    for t in tasks:
        reg = reg_by_id.get(t["task_id"], {})
        ds_key = t.get("dataset_key") or reg.get("dataset_key", "")
        name, env, tone = TASK_DATA.get(ds_key, ((ds_key, ds_key), ("", ""), "info"))
        if t["task_type"] == "regression":
            unit = _unit_display((reg.get("target") or {}).get("unit", ""))
            kind = tr(f"Predicts a number ({unit})", f"预测数值（{unit}）") if unit else tr("Predicts a number", "预测数值")
        else:
            kind = tr("Predicts yes / no", "预测是 / 否")
        if t.get("rows_used") is not None:
            g_en, g_zh = GROUP_NAMES.get(t.get("group_column"), (t.get("group_column"), t.get("group_column")))
            size = tr(f"{t['rows_used']} rows · {t['independent_groups']} {g_en}",
                      f"{t['rows_used']} 行 · {t['independent_groups']} {g_zh}")
        else:
            size = "—"
        reasons = "<br>".join(_md_code(code_text(r)) for r in t.get("reasons", []))
        if t["task_id"] in deployed:
            reasons = (reasons + "<br>" if reasons else "") + html.escape(
                tr("▶ Runs in 🤖 AI Predictions: ", "▶ 用于 🤖 AI 预测：") + tr(*MODEL_LABELS.get(deployed[t["task_id"]], ("", ""))))
        rows.append(
            f'<tr><td><div class="t">{html.escape(cfg_text(reg, "title", t["task_id"]))}</div><div class="sub">{kind}</div></td>'
            f'<td><div class="t">{html.escape(tr(*name))}</div><span class="pill env {tone}">{html.escape(tr(*env))}</span>'
            f'<div class="sub size">{size}</div></td>'
            f'<td>{status_pill(t["evaluation_status"])}' + (f'<div class="sub">{reasons}</div>' if reasons else "") + "</td>"
            f'<td>{status_pill(ds_status(ds_key)["status"])}</td></tr>'
        )
    head = "".join(f"<th>{h}</th>" for h in (tr("Task", "任务"), tr("Data", "数据"), tr("Model evaluation", "模型评估"),
                                              tr("Data source", "数据来源")))
    return f'<div class="tasks"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def render_task_results(t: dict, reg: dict) -> None:
    ttype = t["task_type"]
    target = reg.get("target") or {}
    label = reg.get("label") or {}
    unit = _unit_display(target.get("unit", "")) if ttype == "regression" else ""
    feats = [f["column"] for f in reg.get("features", [])]
    out_col = target.get("column") if ttype == "regression" else label.get("column")
    st.markdown(f"#### {cfg_text(reg, 'title', t['task_id'])}")
    if ttype == "regression":
        evidence = cfg_text(target, "unit_evidence") or cfg_text(target, "definition")
        st.caption(tr(f"Target: `{out_col}` in {unit}. {evidence}", f"目标：`{out_col}`，单位 {unit}。{evidence}"))
    else:
        mapping = " · ".join(f"**{k}** → {v}" for k, v in cfg_text(label, "mapping", {}).items())
        note = cfg_text(label, "mapping_note")
        st.caption(tr(f"Label from `{label.get('source_column')}`: {mapping}. {note}",
                      f"标签来自 `{label.get('source_column')}`：{mapping}。{note}"))
    st.caption(tr("Independent group: ", "独立分组：") + cfg_text(reg, "group_definition", "")
               + (" " + cfg_text(reg, "prediction_stage_note") if reg.get("prediction_stage_note") else ""))

    manifest = load_report_csv(str(REPORTS_DIR / "experiment_split_manifest.csv"))
    man = manifest[manifest["task_id"] == t["task_id"]] if not manifest.empty else manifest
    live = recheck_manifest(man)
    issues = live["group_overlaps"] + live["record_conflicts"] + live["groups_not_once_per_repeat"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr("Rows used", "使用行数"), str(t["rows_used"]),
              help=tr("Rows with a valid target/label after the documented exclusions.", "按记录的排除规则处理后，目标/标签有效的行数。"))
    g_en, g_zh = GROUP_NAMES.get(t["group_column"], (t["group_column"], t["group_column"]))
    c2.metric(tr("Independent groups", "独立分组"), tr(f"{t['independent_groups']} {g_en}", f"{t['independent_groups']} {g_zh}"))
    c3.metric(tr("Outer test folds (5 seeds × 5)", "外层测试折（5 种子 × 5）"), str(t["outer_folds_total"]))
    c4.metric(tr("Leakage re-check", "泄漏复查"), tr(f"{issues} issues in {live['splits']} splits", f"{live['splits']} 次划分中 {issues} 个问题"),
              help=tr("Recomputed now from reports/experiment_split_manifest.csv: no group shared between inner-train, inner-validation "
                      "and test; no record in two partitions; every group tested exactly once per repeat. It checks the splits, "
                      "not the labels.",
                      "现在由 reports/experiment_split_manifest.csv 重新计算：内层训练、内层验证与测试之间没有共享分组；没有记录出现在两个分区；"
                      "每个分组在每次重复中恰好被测试一次。这只检查划分，不检查标签。"))

    base_key = "mean_baseline" if ttype == "regression" else "majority_baseline"
    base = t["models"].get(base_key)
    families = list(t["models"].keys())
    inputs, output = [short_label(c) for c in feats], short_label(out_col)
    folds_df = load_report_csv(str(REPORTS_DIR / "cv_fold_metrics.csv"))
    preds_df = load_report_csv(str(REPORTS_DIR / "cv_predictions.csv"))
    folds_df = folds_df[folds_df["task_id"] == t["task_id"]] if not folds_df.empty else folds_df
    preds_df = preds_df[preds_df["task_id"] == t["task_id"]] if not preds_df.empty else preds_df
    metric = "mae" if ttype == "regression" else "balanced_accuracy"
    metric_name = tr("MAE", "MAE") if ttype == "regression" else tr("balanced accuracy", "平衡准确率")
    axis = dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False, tickfont=dict(color="#8f9ab8", size=11))

    for mtab, fam in zip(st.tabs([tr(*MODEL_LABELS.get(f, (f, f))) for f in families]), families):
        with mtab:
            left, right = st.columns([1, 1.15], gap="large")
            if fam in MODEL_ART:
                left.markdown(_svg_img(MODEL_ART[fam](inputs, output), "model-art"), unsafe_allow_html=True)
            how = tr(*MODEL_HOW.get(fam, ("", "")))
            if base is None:
                right.info(tr("Baseline missing from the summary.", "汇总中缺少基线。"))
            elif ttype == "regression":
                right.markdown(model_panel_html(t["models"][fam], base, how, unit, fam == base_key), unsafe_allow_html=True)
            else:
                right.markdown(classification_panel_html(t["models"][fam], base, how, fam == base_key,
                                                         tr("burned completely", "完全燃尽")), unsafe_allow_html=True)

            f = folds_df[folds_df["model"] == fam].sort_values(["seed", "fold"]).reset_index(drop=True)
            if not f.empty:
                fig_f = go.Figure(go.Scatter(
                    x=list(range(1, len(f) + 1)), y=f[metric], mode="markers",
                    marker=dict(size=9, color="#3987e5", line=dict(width=2, color="#0f1426")),
                    customdata=f[["seed", "fold"]],
                    hovertemplate=tr("Seed %{customdata[0]}, fold %{customdata[1]}<br>%{y:.3f}<extra></extra>",
                                     "种子 %{customdata[0]}，第 %{customdata[1]} 折<br>%{y:.3f}<extra></extra>")))
                fig_f.add_hline(y=float(f[metric].mean()), line=dict(color="#d95926", width=2, dash="dot"),
                                annotation_text=tr("mean", "均值"), annotation_position="top right",
                                annotation_font=dict(color="#c9d2ea", size=11))
                fig_f.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    height=210, margin=dict(l=10, r=10, t=36, b=10), showlegend=False,
                                    title=dict(text=tr(f"{metric_name} on each of the 25 held-out tests (one dot = one test)",
                                                       f"25 次独立测试的{metric_name}（每个点代表一次测试）"),
                                               font=dict(size=13, color="#c9d2ea"), x=0),
                                    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False), yaxis=axis)
                st.plotly_chart(fig_f, use_container_width=True, config={"displayModeBar": False})

            p = preds_df[preds_df["model"] == fam]
            if not p.empty:
                agg = p.groupby("record_id").agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
                                                 group=("group_id", "first")).reset_index()
                if ttype == "regression":
                    lo, hi = float(min(agg.y_true.min(), agg.y_pred.min())), float(max(agg.y_true.max(), agg.y_pred.max()))
                    fig_p = go.Figure()
                    fig_p.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", hoverinfo="skip",
                                               line=dict(color="#8f9ab8", width=1, dash="dash")))
                    fig_p.add_trace(go.Scatter(x=agg.y_true, y=agg.y_pred, mode="markers",
                                               marker=dict(size=8, color="#3987e5", opacity=0.85, line=dict(width=1, color="#0f1426")),
                                               customdata=agg[["record_id", "group"]],
                                               hovertemplate=tr("%{customdata[0]} (%{customdata[1]})<br>measured %{x:.3g}<br>predicted %{y:.3g}<extra></extra>",
                                                                "%{customdata[0]}（%{customdata[1]}）<br>实测 %{x:.3g}<br>预测 %{y:.3g}<extra></extra>")))
                    fig_p.update_layout(xaxis_title=tr(f"Measured ({unit})", f"实测（{unit}）"),
                                        yaxis_title=tr(f"Predicted ({unit})", f"预测（{unit}）"))
                    title = tr("Measured vs predicted, out-of-fold (mean of 5 repeats). On the dashed line = perfect.",
                               "实测与预测（折外预测，5 次重复取均值）。落在虚线上表示完全准确。")
                else:
                    outcome = agg.y_true.map({1.0: tr("flame went out", "火焰熄灭"), 0.0: tr("burned completely", "完全燃尽")})
                    fig_p = go.Figure(go.Box(x=agg.y_pred, y=outcome, orientation="h", boxpoints="all", jitter=0.4,
                                             pointpos=0, marker=dict(size=6, color="#3987e5", opacity=0.75),
                                             line=dict(color="#8f9ab8", width=1), fillcolor="rgba(57,135,229,0.10)",
                                             hovertemplate="%{x:.2f}<extra></extra>"))
                    fig_p.add_vline(x=float(t.get("decision_threshold") or 0.5), line=dict(color="#d95926", width=1.5, dash="dot"))
                    fig_p.update_layout(xaxis_title=tr("Predicted chance the flame goes out", "预测火焰熄灭的概率"),
                                        xaxis_range=[0, 1])
                    title = tr("Predicted chance the flame goes out, by what actually happened (out-of-fold). Dotted line = decision threshold.",
                               "按实际结果分组的火焰熄灭预测概率（折外预测）。点线为判定阈值。")
                fig_p.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    height=320, margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
                                    title=dict(text=title, font=dict(size=13, color="#c9d2ea"), x=0), xaxis=axis, yaxis=axis)
                st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar": False})

    notes = list(cfg_text(t, "warnings", []))
    if t.get("derived_classification") and t["derived_classification"].get("evaluation_status") == "NOT_EVALUATED":
        notes.append(tr("High-hazard flag (≥ 350 kW/m²) not evaluated: every measured peak is on the same side of the "
                        "threshold, so there is nothing to discriminate.",
                        "高危标记（≥ 350 kW/m²）未评估：所有实测峰值都位于阈值同一侧，因此无从区分。"))
    if t["task_id"] == "macfp_pmma_peak_hrr" and "tf_nn_deployed_config" in t["models"]:
        meta = load_json_file(str(ROOT / "models" / "spacecraft_fire_peak_hrr_v1.metadata.json")) or {}
        stored = (meta.get("validation", {}).get("metrics", {}).get("TensorFlow_NN") or {}).get("mae_kw_m2")
        rep = t["models"]["tf_nn_deployed_config"]["fold_level"]["mae"]["mean"]
        best = min(t["models"], key=lambda k: t["models"][k]["fold_level"]["mae"]["mean"])
        if stored:
            notes.append(tr(f"The deployed network's metadata reports MAE {stored:.0f} kW/m² from one 5-fold run. Under this "
                            f"repeated protocol its configuration averages {rep:.0f} kW/m², and the best model here is "
                            f"'{tr(*MODEL_LABELS.get(best, (best, best)))}'.",
                            f"已部署网络的元数据报告的 MAE 为 {stored:.0f} kW/m²（单次 5 折）。在本重复协议下，该配置平均为 "
                            f"{rep:.0f} kW/m²，此处最佳模型为“{tr(*MODEL_LABELS.get(best, (best, best)))}”。"))
    if t.get("dependence_note"):  # the report states it in English; shown here in the UI language
        notes.append(tr("Dependence: the 25 outer folds come from 5 repeats over the same rows, so they are correlated "
                        "re-splits, not 25 independent experiments. Per-repeat pooled scores show repeat-to-repeat "
                        "variation; neither spread is a confidence interval for new experiments.",
                        "相关性：25 个外层折来自对同一批数据的 5 次重复划分，彼此相关，并非 25 次独立实验。按重复汇总的得分展示了重复之间的"
                        "波动；两种波动范围都不是针对新实验的置信区间。"))
    if t.get("pooled_note"):
        notes.append(tr("Pooled classification scores combine probabilities from separately fitted models, which are not "
                        "on a common scale (the constant baseline's pooled ROC-AUC even falls below 0.5), so fold-level "
                        "scores are the primary result.",
                        "汇总的分类得分混合了分别训练的模型给出的概率，它们不在同一尺度上（常数基线的汇总 ROC-AUC 甚至低于 0.5），"
                        "因此以逐折得分为主要结果。"))
    st.markdown("\n".join(f"- {n}" for n in notes))


@st.fragment
def render_audit():
    st.subheader(tr("Scientific audit: data provenance and leakage-audited evaluation", "科学审计：数据来源与无泄漏评估"))
    if CV_SUMMARY is None:
        st.error(tr("reports/cv_summary.json is missing. Run `python -m src.provenance_audit` and then "
                    "`python -m src.run_cv_evaluation`.",
                    "缺少 reports/cv_summary.json。请先运行 `python -m src.provenance_audit`，再运行 `python -m src.run_cv_evaluation`。"),
                 icon="🚫")
        return
    st.caption(tr(f"Every number on this tab is read from reports/ (generated by src/run_cv_evaluation.py at "
                  f"{CV_SUMMARY.get('generated_at_utc')}). Statuses come from src/provenance_audit.py.",
                  f"本页所有数字均读取自 reports/（由 src/run_cv_evaluation.py 于 {CV_SUMMARY.get('generated_at_utc')} 生成）。"
                  "状态来自 src/provenance_audit.py。"))
    reg_by_id = {t["task_id"]: t for t in REGISTRY.get("tasks", [])}
    tasks = CV_SUMMARY.get("tasks", [])
    st.markdown(tr("#### Prediction tasks", "#### 预测任务"))
    n_done = sum(t["evaluation_status"] == "EVALUATED" for t in tasks)
    st.caption(tr(f"{n_done} of {len(tasks)} tasks could be evaluated; the others show why not.",
                  f"{len(tasks)} 个任务中有 {n_done} 个可以评估；其余任务注明了无法评估的原因。"))
    st.markdown(task_overview_html(tasks, reg_by_id), unsafe_allow_html=True)

    evaluated = [t for t in tasks if t["evaluation_status"] == "EVALUATED"]
    if evaluated:
        names = [cfg_text(reg_by_id.get(t["task_id"], {}), "title", t["task_id"]) for t in evaluated]
        choice = st.radio(tr("Show results for", "查看结果"), names, key="audit_task")
        task = evaluated[names.index(choice)]
        render_task_results(task, reg_by_id.get(task["task_id"], {}))

    st.markdown(tr("#### Reports", "#### 报告文件"))
    files = ["experiment_split_manifest.csv", "cv_fold_metrics.csv", "cv_predictions.csv", "cv_summary.json",
             "cv_protocol.json", "data_audit_report.json", "data_provenance_audit.json", "provenance_mapping.csv"]
    with st.container(key="report_files"):  # equal-size tiles, styled by .st-key-report_files
        for start in range(0, len(files), 4):  # one st.columns per row, so phones (stacked columns) keep the list order
            for col, name in zip(st.columns(4), files[start:start + 4]):
                path = REPORTS_DIR / name
                if path.exists():
                    wrappable = name.replace("_", "_\u200b")  # zero-width space: long names break at underscores, not mid-word
                    col.download_button(f"⬇ {wrappable}", path.read_bytes(), file_name=name, key=f"dl_{name}",
                                        mime="text/csv" if name.endswith(".csv") else "application/json",
                                        use_container_width=True)

    with st.expander(tr("Retired evaluation on synthetic data: why it was removed", "已退役的合成数据评估：为何移除")):
        st.markdown(tr("Earlier versions scored models on 80 formula-generated rows labelled as NASA data, with a BASS-II target "
                       "computed from an input. Those files are kept unchanged in `data/legacy/` and `reports/legacy/`; the audit "
                       "found:",
                       "早期版本在 80 行由公式生成、却被标为 NASA 数据的记录上为模型打分，且 BASS-II 目标值由输入计算得到。这些文件原样保存在 "
                       "`data/legacy/` 与 `reports/legacy/`；审计发现："))
        for path, d in (PROV_REPORT or {}).get("legacy_files", {}).items():
            st.markdown(f"- `{path}` · **{d['status']}**\n" + "".join(f"  - {code_text(e)}\n" for e in d["evidence"]))


with tab_audit:
    render_audit()


# =========================================================================
# TAB 5, part 1: the deployed FLEX models (microgravity droplet flames, ISS)
# =========================================================================
FLEX_PRESETS = [  # real FLEX test atmospheres, each run several times (pressure mmHg, O2 %, CO2 %, He %)
    (("Normal air · 1 atm", "正常空气 · 1 atm"), (760, 21, 0, 0)),
    (("Low oxygen · 1 atm", "低氧 · 1 atm"), (760, 15, 0, 0)),
    (("CO₂-diluted · 0.71 atm", "CO₂ 稀释 · 0.71 atm"), (540, 23, 25, 0)),
    (("Helium-diluted · 1 atm", "氦气稀释 · 1 atm"), (760, 17, 0, 20)),
    (("Oxygen-rich · 0.71 atm", "富氧 · 0.71 atm"), (540, 34, 0, 0)),
    (("High pressure, heavy CO₂ · 2.7 atm", "高压、高 CO₂ · 2.7 atm"), (2070, 21, 70, 0)),
]
FLEX_INPUT_LABELS = {
    "pressure_mmhg": ("Pressure", "压力"), "x_o2": ("Oxygen", "氧气"), "x_co2": ("CO₂", "CO₂"),
    "x_he": ("Helium", "氦气"), "d0_mm": ("Droplet size", "液滴大小"),
}


def _apply_flex_preset(lang_key: str) -> None:
    vals = st.session_state["flex_vals"]
    vals["preset"] = st.session_state[f"flex_preset_{lang_key}"]
    p, o2, co2, he = FLEX_PRESETS[vals["preset"]][1]
    vals.update(p_atm=round(p / 760, 2), o2=o2, co2=co2, he=he)
    for name in ("p_atm", "o2", "co2", "he"):  # the sliders re-read flex_vals on this rerun
        st.session_state.pop(f"flex_{name}_{lang_key}", None)


def _flex_input_label(column: str, value: float, average: float) -> str:
    """'O₂ 21% (avg 19%)': the bar is this test's value measured against the average training test."""
    if column == "is_heptane":
        return tr("Fuel: heptane", "燃料：正庚烷") if value == 1 else tr("Fuel: methanol", "燃料：甲醇")
    shown = {"pressure_mmhg": lambda v: f"{v / 760:.2f} atm", "d0_mm": lambda v: f"{v:.1f} mm"}.get(
        column, lambda v: f"{100 * v:.0f}%")
    return tr(f"{FLEX_INPUT_LABELS[column][0]} {shown(value)} (avg {shown(average)})",
              f"{FLEX_INPUT_LABELS[column][1]} {shown(value)}（平均 {shown(average)}）")


def _contribution_chart(contrib: dict, inputs: dict, means: list, title: str, unit: str,
                        more_fire_positive: bool) -> go.Figure:
    """Signed bars, one per input. Orange = more fire (faster burning / keeps burning), blue = less fire."""
    names = [_flex_input_label(c, inputs[c], m) for c, m in zip(contrib, means)]
    vals = list(contrib.values())
    colors = ["#d95926" if (v > 0) == more_fire_positive else "#3987e5" for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors,
                           text=[f"{v:+.3f}" for v in vals], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:+.3f} " + unit + "<extra></extra>"))
    span = max(abs(v) for v in vals) * 1.35 or 1.0
    fig.add_vline(x=0, line=dict(color="#8f9ab8", width=1))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=290,
        title=dict(text=title, font=dict(size=13, color="#c9d2ea"), x=0, xanchor="left"),
        margin=dict(l=10, r=20, t=44, b=30), showlegend=False, font=dict(color="#c9d2ea"),
        xaxis=dict(range=[-span, span], zeroline=False, gridcolor="rgba(255,255,255,0.06)",
                   title=dict(text=unit, font=dict(size=11, color="#8f9ab8")), tickfont=dict(color="#8f9ab8", size=10)),
        yaxis=dict(autorange="reversed", automargin=True, tickfont=dict(color="#c9d2ea", size=12)),
    )
    return fig


@st.fragment
def render_flex_predictor() -> None:
    st.subheader(tr("Microgravity droplet flame: how fast does it burn, and does it go out?",
                    "微重力液滴火焰：燃烧有多快？会不会熄灭？"))
    st.caption(status_badge("MODEL_PREDICTION") + tr(
        " · two models trained on NASA FLEX droplet tests aboard the International Space Station (PSI-69, verified "
        "source). Pick a test condition; the AI answers, shows why, and shows the closest real NASA tests.",
        " · 两个模型基于国际空间站上的 NASA FLEX 液滴试验训练（PSI-69，已核实来源）。选择试验条件后，AI 给出预测、"
        "说明原因，并列出最接近的真实 NASA 试验。"))
    try:
        ranges = flex.input_ranges()
        k_meta, e_meta = flex.load_model(flex.BURNING_RATE_PATH), flex.load_model(flex.EXTINCTION_PATH)
    except flex.FlexModelUnavailable as exc:
        st.error(str(exc))
        return

    # The chosen values live in flex_vals; widget keys carry the language, because a widget whose label changes is a
    # new widget to Streamlit and would otherwise reset to its minimum when the language is switched.
    p0, o20, co20, he0 = FLEX_PRESETS[0][1]
    vals = st.session_state.setdefault("flex_vals", {"preset": 0, "heptane": True, "p_atm": round(p0 / 760, 2),
                                                      "o2": o20, "co2": co20, "he": he0, "d0": 3.0})

    c_in, c_out = st.columns([1, 1.4], gap="large")
    with c_in:
        st.markdown(tr("##### 1. Test condition", "##### 1. 试验条件"))
        fuel = st.radio(tr("Fuel droplet", "燃料液滴"), [tr("n-Heptane", "正庚烷"), tr("Methanol", "甲醇")],
                        index=0 if vals["heptane"] else 1, horizontal=True, key=f"flex_fuel_{lang}")
        heptane = vals["heptane"] = fuel == tr("n-Heptane", "正庚烷")
        st.selectbox(tr("Load a tested NASA atmosphere", "载入 NASA 测试过的气氛"), range(len(FLEX_PRESETS)),
                     index=vals["preset"], format_func=lambda i: tr(*FLEX_PRESETS[i][0]), key=f"flex_preset_{lang}",
                     on_change=_apply_flex_preset, args=(lang,))
        p_lo, p_hi = ranges["pressure_mmhg"]
        p_atm = vals["p_atm"] = st.slider(tr("Pressure (atm)", "压力 (atm)"), math.ceil(p_lo / 7.6) / 100,
                                          math.floor(p_hi / 7.6) / 100, value=vals["p_atm"], step=0.01,
                                          key=f"flex_p_atm_{lang}")
        o2 = vals["o2"] = st.slider(tr("Oxygen O₂ (%)", "氧气 O₂ (%)"), int(round(ranges["x_o2"][0] * 100)),
                                    int(round(ranges["x_o2"][1] * 100)), value=vals["o2"], step=1, key=f"flex_o2_{lang}")
        co2 = vals["co2"] = st.slider(tr("Carbon dioxide CO₂ (%)", "二氧化碳 CO₂ (%)"), 0, int(round(ranges["x_co2"][1] * 100)),
                                      value=vals["co2"], step=1, key=f"flex_co2_{lang}")
        he = vals["he"] = st.slider(tr("Helium He (%)", "氦气 He (%)"), 0, int(round(ranges["x_he"][1] * 100)),
                                    value=vals["he"], step=1, key=f"flex_he_{lang}")
        d0 = vals["d0"] = st.slider(tr("Droplet diameter at ignition (mm)", "点火时液滴直径 (mm)"),
                                    math.ceil(ranges["d0_mm"][0] * 10) / 10, math.floor(ranges["d0_mm"][1] * 10) / 10,
                                    value=vals["d0"], step=0.1, key=f"flex_d0_{lang}")
        n2 = 100 - o2 - co2 - he
        if n2 < 0:
            st.error(tr(f"O₂ + CO₂ + He add up to {o2 + co2 + he}%. Lower one of them to 100% or less.",
                        f"O₂ + CO₂ + He 合计 {o2 + co2 + he}%，请降低其中一项，使总和不超过 100%。"))
            return
        st.caption(tr(f"Nitrogen fills the rest: **{n2}%** · {p_atm * 760:.0f} mmHg",
                      f"其余为氮气：**{n2}%** · {p_atm * 760:.0f} mmHg"))
    inputs = {"is_heptane": 1.0 if heptane else 0.0, "pressure_mmhg": p_atm * 760, "x_o2": o2 / 100,
              "x_co2": co2 / 100, "x_he": he / 100, "d0_mm": float(d0)}
    try:
        pred, near = flex.predict(inputs), flex.nearest_tests(inputs)
    except ValueError as exc:
        c_out.error(str(exc))
        return

    cv_k, cv_e = k_meta["cross_validation"], e_meta["cross_validation"]
    with c_out:
        st.markdown(tr("##### 2. What the AI predicts", "##### 2. AI 的预测"))
        goes_out = pred["flame_goes_out"]
        cards = [
            (f"{pred['burning_rate_mm2_s']:.3f} mm²/s", tr("Burning rate K", "燃烧速率 K")),
            (tr("Goes out early", "提前熄灭") if goes_out else tr("Burns to the end", "燃烧至耗尽"),
             tr(f"Flame-out call · score {pred['extinction_score']:.2f}", f"熄火判断 · 得分 {pred['extinction_score']:.2f}")),
            (f"≈ {pred['burnout_time_s']:.0f} s", tr("To burn out, if it stays lit", "若不熄灭，烧尽所需时间")),
        ]
        st.markdown('<div class="telemetry compact">' + "".join(
            f'<div class="tele"><div class="tele-v">{html.escape(v)}</div><div class="tele-l">{html.escape(l)}</div></div>'
            for v, l in cards) + "</div>", unsafe_allow_html=True)
        st.caption(tr(
            "K is how fast the droplet's surface area shrinks (bigger = burns faster). The flame-out call says whether the "
            f"flame goes out while fuel is left (score ≥ {pred['extinction_threshold']:.2f}) or burns until the droplet is "
            f"used up. Burn-out time uses the d²-law: d₀² / K for this {d0:.1f} mm droplet.",
            "K 表示液滴表面积缩小的速度（越大燃烧越快）。熄火判断表示火焰是在仍有燃料时熄灭"
            f"（得分 ≥ {pred['extinction_threshold']:.2f}），还是一直燃烧到液滴耗尽。烧尽时间按 d² 定律计算：d₀² / K，"
            f"液滴直径 {d0:.1f} mm。"))

        fuel_name = tr("n-heptane", "正庚烷") if heptane else tr("methanol", "甲醇")
        if near["tests"]:
            k_txt = (tr(f"measured K **{near['k_mean']:.3f}** mm²/s (range {near['k_min']:.2f}–{near['k_max']:.2f})",
                        f"实测 K **{near['k_mean']:.3f}** mm²/s（范围 {near['k_min']:.2f}–{near['k_max']:.2f}）")
                     if near["k_readings"] else tr("no K readings", "无 K 读数"))
            st.info("🛰 " + tr(
                f"**Real NASA tests close to this:** {near['tests']} {fuel_name} tests at a similar pressure and gas mix "
                f"(droplets {near['d0_min']:.1f}–{near['d0_max']:.1f} mm): {k_txt}. Flame went out in **{near['went_out']}**, "
                f"burned to the end in **{near['burned_out']}**, disrupted in {near['disrupted']}.",
                f"**与此相近的真实 NASA 试验：** {near['tests']} 次{fuel_name}试验，压力与气体组成相近"
                f"（液滴 {near['d0_min']:.1f}–{near['d0_max']:.1f} mm）：{k_txt}。火焰熄灭 **{near['went_out']}** 次，"
                f"燃烧至耗尽 **{near['burned_out']}** 次，中断 {near['disrupted']} 次。"))
        else:
            st.warning(tr(
                "**No NASA test was run close to this combination.** The AI is combining what it learned from other "
                "conditions, so treat this prediction with extra caution.",
                "**NASA 没有在接近此组合的条件下做过试验。** AI 是在综合其他条件下学到的规律，请对该预测格外谨慎。"))
        st.caption(tr(
            f"**How good is it?** Scored on test days it never saw ({cv_k['model']['r2']['n']} grouped folds): the "
            f"burning-rate model explains about **{cv_k['model']['r2']['mean']:.0%}** of the variation, typical error "
            f"±{cv_k['model']['mae']['mean']:.3f} mm²/s (guessing the average: ±{cv_k['baseline']['mae']['mean']:.3f}). "
            f"The flame-out model's balanced accuracy is **{cv_e['model']['balanced_accuracy']['mean']:.2f}** "
            f"(coin flip: 0.50), ROC-AUC {cv_e['model']['roc_auc']['mean']:.2f}. Details: Scientific Audit tab.",
            f"**准确度如何？** 在模型从未见过的试验日上评分（{cv_k['model']['r2']['n']} 个分组折）：燃烧速率模型约解释 "
            f"**{cv_k['model']['r2']['mean']:.0%}** 的变化，典型误差 ±{cv_k['model']['mae']['mean']:.3f} mm²/s"
            f"（只猜平均值：±{cv_k['baseline']['mae']['mean']:.3f}）。熄火模型的平衡准确率为 "
            f"**{cv_e['model']['balanced_accuracy']['mean']:.2f}**（抛硬币：0.50），ROC-AUC {cv_e['model']['roc_auc']['mean']:.2f}。"
            "详见科学审计页。"))

    st.markdown(tr("##### 3. Why the AI says this", "##### 3. AI 为什么这样判断"))
    st.caption(tr("Both models are linear, so each prediction splits exactly into one push per input, measured from a test "
                  "with average conditions. Orange = more fire (faster burning, or keeps burning); blue = less fire.",
                  "两个模型都是线性的，因此每个预测都能精确分解为各输入的推动量（以平均条件的试验为起点）。"
                  "橙色 = 火更强（燃烧更快或持续燃烧）；蓝色 = 火更弱。"))
    ch1, ch2 = st.columns(2)
    ch1.plotly_chart(_contribution_chart(
        pred["burning_rate_contributions"], inputs, k_meta["scaler_mean"],
        tr(f"Burning rate: change from the average test ({pred['burning_rate_at_average_test']:.3f} mm²/s)",
           f"燃烧速率：相对平均试验（{pred['burning_rate_at_average_test']:.3f} mm²/s）的变化"),
        "mm²/s", more_fire_positive=True), use_container_width=True, config={"displayModeBar": False})
    ch2.plotly_chart(_contribution_chart(
        pred["extinction_contributions"], inputs, e_meta["scaler_mean"],
        tr("Flame-out: push toward going out (+) or burning to the end (−)", "熄火：推向熄灭 (+) 或燃烧至耗尽 (−)"),
        tr("log-odds", "对数几率"), more_fire_positive=False), use_container_width=True, config={"displayModeBar": False})

    td = e_meta["training_data"]["class_counts"]
    st.caption(tr(
        f"The flame-out score comes from a class-balanced model (NASA's tests are {td['1']} flame-outs to {td['0']} burn-outs), "
        "so it ranks the risk and 0.5 is the tested decision line, but it is not a probability. "
        + str(k_meta["scope"]) + " Not for other fuels, solid materials or whole-cabin fires. "
        f"Models: `{flex.BURNING_RATE_PATH.name}` (Ridge, alpha = {k_meta['hyperparameters']['alpha']:g}) and "
        f"`{flex.EXTINCTION_PATH.name}` (logistic regression, C = {e_meta['hyperparameters']['C']:g}), "
        "built by `src/train_flex_models.py` with the same recipe the cross-validation tested.",
        f"熄火得分来自类别平衡模型（NASA 试验中熄灭 {td['1']} 次、燃尽 {td['0']} 次），因此它用于排序风险，0.5 是经过测试的判断线，"
        "但并不是概率。" + str(k_meta["scope_zh"]) + "不适用于其他燃料、固体材料或整个舱室的火灾。"
        f"模型：`{flex.BURNING_RATE_PATH.name}`（岭回归，alpha = {k_meta['hyperparameters']['alpha']:g}）与 "
        f"`{flex.EXTINCTION_PATH.name}`（逻辑回归，C = {e_meta['hyperparameters']['C']:g}），"
        "由 `src/train_flex_models.py` 按交叉验证测试过的同一方案构建。"))


# =========================================================================
# TAB 5, part 2: After-test estimates from a separately trained experimental model
# =========================================================================
with tab_prediction:
    sub_flex, sub_pmma = st.tabs([tr("🛰 Microgravity droplet flame (FLEX · ISS)", "🛰 微重力液滴火焰（FLEX · 国际空间站）"),
                                  tr("🔥 Ground PMMA burn (CSV upload)", "🔥 地面 PMMA 燃烧（上传 CSV）")])
    with sub_flex:
        render_flex_predictor()
    with sub_pmma:
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
            # Preview with two REAL MaCFP burns and their out-of-fold predictions (read from reports/, no model call,
            # so TensorFlow is not loaded just to view the tab).
            st.markdown(tr("#### 🔥 Preview: two real burns, measured vs predicted", "#### 🔥 预览：两次真实燃烧，实测与预测对比"))
            examples = preview_burns()
            if examples:
                meta = load_json_file(str(ROOT / "models" / "spacecraft_fire_peak_hrr_v1.metadata.json")) or {}
                render_flame_gallery(examples, ref_peak=float(meta.get("training_data", {}).get("peak_hrr_max_kw_m2") or 3000.0),
                                     threshold=350.0)
                st.caption(tr(
                    f"{status_badge('REAL_MEASUREMENT')}: the burn's inputs and peak come from the verified MaCFP lab file. "
                    f"{status_badge('MODEL_PREDICTION')}: the deployed network's recipe, trained without that laboratory "
                    "(out-of-fold, mean of 5 repeats). Burns chosen by a fixed rule: the median-peak run at the lowest and "
                    "highest heat flux. Upload a CSV (or the sample above) to see your own burns.",
                    f"{status_badge('REAL_MEASUREMENT')}：输入与峰值来自已核实的 MaCFP 实验室文件。"
                    f"{status_badge('MODEL_PREDICTION')}：已部署网络的方案，在不含该实验室数据的情况下训练（折外预测，5 次重复取均值）。"
                    "试验按固定规则选取：最低与最高热通量下峰值居中的一次。上传 CSV（或上方的示例文件）即可查看你自己的燃烧数据。",
                ))
            else:
                st.info(tr("Preview unavailable: reports/cv_predictions.csv or the MaCFP dataset is missing. "
                           "Run `python -m src.run_cv_evaluation`.",
                           "预览不可用：缺少 reports/cv_predictions.csv 或 MaCFP 数据集。请运行 `python -m src.run_cv_evaluation`。"))
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
                    st.caption(status_badge("MODEL_PREDICTION") + tr(
                        " · Animated schematic of the cone-calorimeter test. Flame height scales with predicted peak HRR "
                        "(∝ HRR^0.4), the sustained flame with mean/peak HRR, and the orange arrows with heat flux. "
                        "Illustrative, not to scale.",
                        " · 锥形量热仪试验的动画示意图。火焰高度随预测峰值 HRR 变化（∝ HRR^0.4），持续火焰大小对应平均/峰值 HRR，"
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
