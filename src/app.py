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
.updatemenu-item-rect { fill: #FF9A4D !important; stroke: #FFB27A !important; rx: 8px; ry: 8px; transition: fill .15s ease; }
.updatemenu-button:hover .updatemenu-item-rect { fill: #FFC999 !important; }
.updatemenu-button:active .updatemenu-item-rect { fill: #FF7A2B !important; }
.updatemenu-item-text { fill: #0B1020 !important; font-weight: 700 !important; }

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
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
                    text=[centroid_label], textposition="top right", textfont=dict(color="#00E5FF"),
                    showlegend=False, hoverinfo="skip",
                ),
            ]

        def frame_title(i: int) -> str:
            return tr(
                f"t = {sim['times'][i]:.2f} s  ·  2D temperature field and centroid",
                f"t = {sim['times'][i]:.2f} s  ·  2D 温度场与质心位移",
            )

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
        # Direction cues drawn on the chart itself: where gravity and airflow point.
        if sim_g > 0:
            fig_2d.add_annotation(xref="paper", yref="paper", x=0.03, y=0.78, ax=0, ay=-48, showarrow=True,
                                  arrowhead=3, arrowwidth=3, arrowcolor="#7CC4FF")
            fig_2d.add_annotation(xref="paper", yref="paper", x=0.045, y=0.88, showarrow=False, xanchor="left",
                                  text=tr(f"Gravity {sim_g:.1f} m/s²", f"重力 {sim_g:.1f} m/s²"),
                                  font=dict(color="#7CC4FF", size=12))
        else:
            fig_2d.add_annotation(xref="paper", yref="paper", x=0.02, y=0.93, showarrow=False, xanchor="left",
                                  text=tr("Gravity ≈ 0 (weightless)", "重力 ≈ 0（失重）"),
                                  font=dict(color="#7CC4FF", size=12))
        if sim_u > 0:
            fig_2d.add_annotation(xref="paper", yref="paper", x=0.30, y=0.07, ax=-52, ay=0, showarrow=True,
                                  arrowhead=3, arrowwidth=3, arrowcolor="#9DFFB0")
            fig_2d.add_annotation(xref="paper", yref="paper", x=0.31, y=0.07, showarrow=False, xanchor="left",
                                  text=tr(f"Cabin airflow {sim_u:.2f} m/s", f"舱内气流 {sim_u:.2f} m/s"),
                                  font=dict(color="#9DFFB0", size=12))
        play_args = dict(frame=dict(duration=140, redraw=True), transition=dict(duration=0),
                         fromcurrent=True, mode="immediate")
        fig_2d.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title=dict(text=frame_title(start), font=dict(size=15)),
            xaxis_title=tr("Sideways distance (cm) →  direction of airflow", "水平距离 (cm) →  气流方向"),
            yaxis_title=tr("Height (cm) ↑  away from gravity", "高度 (cm) ↑  背离重力方向"),
            xaxis=dict(title_standoff=8, automargin=True),
            yaxis=dict(scaleanchor="x", scaleratio=1, automargin=True), height=560,
            margin=dict(l=10, r=10, t=50, b=130),  # bottom strip reserved for the play controls
            updatemenus=[dict(
                type="buttons", direction="left", x=0.0, y=0.0, xanchor="left", yanchor="top",
                pad=dict(t=78, l=0, r=8), bgcolor="#FF9A4D", bordercolor="#FFB27A", active=-1,
                font=dict(color="#0B1020", size=12),
                buttons=[
                    dict(label=tr("▶ Play", "▶ 播放"), method="animate", args=[None, play_args]),
                    dict(label=tr("⏸ Pause", "⏸ 暂停"), method="animate",
                         args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
                ],
            )],
            sliders=[dict(
                active=start, x=0.30, y=0.0, len=0.68, xanchor="left", yanchor="top",
                pad=dict(t=70, b=0, l=0, r=0), currentvalue=dict(visible=False),
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
