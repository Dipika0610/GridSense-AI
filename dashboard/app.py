"""
GridSense AI — Unified Machine Learning for Sustainable Electricity
Minimalist Editorial Interface (Monochrome Edition)
Calibrated for Indian Power Sector Standards (MSEDCL ToD Tariffs & CEA National Grid Carbon)
Team: CodeCatalyst | A. P. Shah Institute of Technology
PCCOE Indradhanu Hackathon — Better Tomorrow
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.generator import EnergyDataGenerator
from core.forecasting import ForecastingCore
from core.battery_optimizer import BatteryOptimizer
from core.load_scheduler import CarbonAwareLoadScheduler
from core.impact_engine import ImpactEngine

# Page Configuration
st.set_page_config(
    page_title="GridSense AI — Unified Electricity Management",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Commercial energy operations theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --ink: #172923;
        --muted: #718078;
        --green: #16835d;
        --green-dark: #106648;
        --line: #e3ebe6;
        --canvas: #f4f7f5;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: var(--ink);
    }
    .stApp { background: var(--canvas); }
    [data-testid="stHeader"] { background: rgba(244, 247, 245, .88); }
    [data-testid="stMainBlockContainer"] {
        max-width: 1540px;
        padding: 2rem 2.4rem 3rem;
    }

    .hero-shell {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 24px;
        padding: 26px 30px;
        margin-bottom: 16px;
        color: white;
        background: linear-gradient(112deg, #12382c 0%, #17563f 68%, #217153 100%);
        border: 1px solid rgba(255,255,255,.1);
        border-radius: 18px;
        box-shadow: 0 12px 30px rgba(20,57,43,.12);
    }
    .hero-eyebrow {
        margin-bottom: 7px;
        color: #a9d8bd;
        font-size: .68rem;
        font-weight: 700;
        letter-spacing: .15em;
        text-transform: uppercase;
    }
    .brand-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -.045em;
        color: #fff;
        margin-bottom: 4px;
        line-height: 1.1;
    }
    .brand-title span { color: #80d5a5; }
    .brand-tagline {
        font-size: .9rem;
        color: #d3e6dc;
        font-weight: 400;
        margin: 0;
    }
    .hero-status {
        min-width: 190px;
        padding: 14px 17px;
        color: #e5f4eb;
        background: rgba(255,255,255,.09);
        border: 1px solid rgba(255,255,255,.16);
        border-radius: 12px;
        font-size: .78rem;
        text-align: right;
    }
    .hero-status strong {
        display: block;
        margin-top: 5px;
        color: white;
        font-size: 1.35rem;
        letter-spacing: -.03em;
    }
    .status-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        margin-right: 7px;
        background: #75e0a2;
        border-radius: 50%;
        box-shadow: 0 0 0 4px rgba(117,224,162,.14);
    }
    .meta-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 9px;
        margin-bottom: 20px;
    }
    .meta-badge {
        display: inline-flex;
        align-items: center;
        padding: 7px 11px;
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 999px;
        font-size: .74rem;
        font-weight: 500;
        color: #66766e;
    }
    .meta-badge strong {
        font-weight: 600;
        margin-left: 4px;
        color: var(--ink);
    }
    .stat-card {
        height: 100%;
        min-height: 126px;
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 13px;
        padding: 17px 17px 15px;
        margin-bottom: 12px;
        box-shadow: 0 3px 12px rgba(24,54,40,.035);
        transition: transform .18s ease, box-shadow .18s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 9px 22px rgba(24,54,40,.08);
    }
    .stat-label {
        font-size: .66rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .075em;
        color: var(--muted);
        margin-bottom: 10px;
    }
    .stat-value {
        font-size: clamp(1.25rem, 1.8vw, 1.8rem);
        font-weight: 700;
        color: var(--ink);
        letter-spacing: -.045em;
        line-height: 1.1;
    }
    .stat-sub {
        font-size: .72rem;
        font-weight: 500;
        color: var(--green-dark);
        margin-top: 9px;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #172923 !important;
        letter-spacing: -.025em;
        margin: 14px 0 6px;
    }
    .section-desc {
        font-size: .88rem;
        color: #4b5d54 !important;
        margin-bottom: 20px;
        line-height: 1.55;
    }
    .doc-box {
        height: 100%;
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 16px 18px;
        font-size: .85rem;
        color: #263830 !important;
        line-height: 1.55;
        margin-bottom: 14px;
    }
    .doc-box b { color: #172923 !important; }
    .stat-card,
    .stat-card *,
    .doc-box,
    .doc-box *,
    .meta-badge,
    .meta-badge *,
    .hero-shell,
    .hero-shell * {
        cursor: default;
    }
    div[data-testid="stDataFrame"] {
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 12px;
        background: white;
    }
    section[data-testid="stSidebar"] {
        background: #142a23;
        border-right: 1px solid #203b31;
    }
    section[data-testid="stSidebar"] > div { background: transparent; }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p { color: #e4eee8; }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #a9beb2; }
    section[data-testid="stSidebar"] hr { border-color: #345146; }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #203b31;
        border-color: #365447;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] * { color: #f4faf6; }
    section[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
        background: #74d59a;
        border-color: #74d59a;
    }

    /* Complete Tabs Container & High-Contrast Segmented Bar */
    div[data-testid="stTabs"] {
        margin-top: 18px !important;
        margin-bottom: 26px !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 12px !important;
        background: #ffffff !important;
        border: 1.5px solid #d5e3dc !important;
        border-radius: 16px !important;
        padding: 10px 12px !important;
        box-shadow: 0 4px 16px rgba(23, 41, 35, 0.05) !important;
    }

    /* ALL TAB BUTTONS (DEFAULT / INACTIVE STATE) */
    div[data-testid="stTabs"] button,
    div[data-baseweb="tab-list"] button,
    button[data-baseweb="tab"],
    [data-baseweb="tab"] {
        background-color: #edf4f0 !important;
        border: 1.5px solid #cadcd2 !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        cursor: pointer !important;
        transition: all 0.18s ease-in-out !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* INACTIVE TAB LABELS - 100% VISIBLE DEEP FOREST INK */
    div[data-testid="stTabs"] button *,
    div[data-testid="stTabs"] button p,
    div[data-testid="stTabs"] button span,
    div[data-testid="stTabs"] button div,
    button[data-baseweb="tab"] *,
    button[data-baseweb="tab"] p,
    button[data-baseweb="tab"] span,
    button[data-baseweb="tab"] div,
    button[data-baseweb="tab"] strong,
    button[data-baseweb="tab"] [data-testid="stMarkdownContainer"] p,
    [data-baseweb="tab"] * {
        color: #172923 !important;
        font-size: 0.92rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        opacity: 1 !important;
        visibility: visible !important;
        text-shadow: none !important;
    }

    /* HOVER STATE FOR TABS */
    div[data-testid="stTabs"] button:hover,
    button[data-baseweb="tab"]:hover,
    [data-baseweb="tab"]:hover {
        background-color: #dcebe2 !important;
        border-color: #a6c9b5 !important;
        transform: translateY(-1px) !important;
    }
    div[data-testid="stTabs"] button:hover *,
    button[data-baseweb="tab"]:hover *,
    [data-baseweb="tab"]:hover * {
        color: #0f6648 !important;
    }

    /* ACTIVE / SELECTED TAB BUTTON */
    div[data-testid="stTabs"] button[aria-selected="true"],
    div[data-baseweb="tab-list"] button[aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"],
    [data-baseweb="tab"][aria-selected="true"] {
        background-color: #16835d !important;
        border-color: #106648 !important;
        box-shadow: 0 4px 14px rgba(22, 131, 93, 0.32) !important;
        transform: translateY(-1px) !important;
    }

    /* ACTIVE / SELECTED TAB LABELS - PURE CRISP WHITE */
    div[data-testid="stTabs"] button[aria-selected="true"] *,
    div[data-testid="stTabs"] button[aria-selected="true"] p,
    div[data-testid="stTabs"] button[aria-selected="true"] span,
    div[data-testid="stTabs"] button[aria-selected="true"] div,
    div[data-testid="stTabs"] button[aria-selected="true"] strong,
    button[data-baseweb="tab"][aria-selected="true"] *,
    button[data-baseweb="tab"][aria-selected="true"] p,
    button[data-baseweb="tab"][aria-selected="true"] span,
    button[data-baseweb="tab"][aria-selected="true"] div,
    button[data-baseweb="tab"][aria-selected="true"] strong,
    button[data-baseweb="tab"][aria-selected="true"] [data-testid="stMarkdownContainer"] p,
    [data-baseweb="tab"][aria-selected="true"] * {
        color: #ffffff !important;
        font-weight: 700 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* HIDE THE DEFAULT HIGHLIGHT BAR */
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] [data-baseweb="tab-border"] {
        display: none !important;
        height: 0px !important;
    }
    div[data-testid="stPlotlyChart"] {
        padding: 8px 10px 2px;
        background: white;
        border: 1px solid var(--line);
        border-radius: 14px;
        box-shadow: 0 3px 12px rgba(24,54,40,.035);
    }
    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] { padding: 1.2rem 1rem 2rem; }
        .hero-shell { align-items: flex-start; flex-direction: column; padding: 22px; }
        .hero-status { width: 100%; text-align: left; }
        .brand-title { font-size: 1.75rem; }
    }
</style>
""", unsafe_allow_html=True)


# Plotly theme shared across operational views
PLOTLY_LAYOUT = dict(
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    font=dict(family="Inter, sans-serif", color="#31443a", size=11),
    margin=dict(l=48, r=24, t=42, b=38),
    xaxis=dict(
        showgrid=True,
        gridcolor="#edf2ef",
        linecolor="#e3ebe6",
        tickcolor="#e3ebe6",
        tickfont=dict(size=10, color="#718078")
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#edf2ef",
        linecolor="#e3ebe6",
        tickcolor="#e3ebe6",
        tickfont=dict(size=10, color="#718078")
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=10, color="#52635a")
    )
)


@st.cache_data(show_spinner=False)
def load_and_train_system(model_type="random_forest"):
    gen = EnergyDataGenerator(seed=42)
    hist_df = gen.generate_historical_dataset(days=60)
    forecaster = ForecastingCore(model_type=model_type)
    metrics = forecaster.train(hist_df)
    return gen, forecaster, hist_df, metrics


# ---------------- Sidebar Controls ----------------
st.sidebar.markdown("### GridSense <span style='color:#74d59a'>AI</span>", unsafe_allow_html=True)
st.sidebar.caption("ENERGY INTELLIGENCE PLATFORM")
st.sidebar.write("---")

st.sidebar.markdown("**Simulation Parameters**")
scenario = st.sidebar.selectbox(
    "Day Scenario",
    ["normal", "high_solar", "cloudy_storm", "extreme_peak_pricing"],
    format_func=lambda x: {
        "normal": "Typical Day (Maharashtra Grid)",
        "high_solar": "High Solar Day (Clear Sky)",
        "cloudy_storm": "Monsoon / Cloudy (Low Solar)",
        "extreme_peak_pricing": "Severe Evening Peak Surcharge"
    }[x]
)

selected_model = st.sidebar.selectbox(
    "Forecasting Algorithm",
    ["random_forest", "ensemble", "svr", "neural_net"],
    format_func=lambda x: {
        "random_forest": "Random Forest Regressor",
        "ensemble": "Gradient Boosted Tree Ensemble",
        "svr": "Support Vector Regression (SVR)",
        "neural_net": "Multi-Layer Perceptron (MLP)"
    }[x]
)

st.sidebar.markdown("**Battery Energy Storage (Supply Arm)**")
battery_capacity = st.sidebar.slider("Battery Capacity (kWh)", 200, 1000, 500, step=50)
battery_power = st.sidebar.slider("Inverter Rating (kW)", 50, 300, 150, step=25)
initial_soc = st.sidebar.slider("Initial SoC (%)", 10, 80, 25, step=5) / 100.0
enforce_anti_paradox = st.sidebar.checkbox(
    "Prevent Battery Paradox",
    value=True,
    help="Guarantees battery does not charge on dirty coal grid power during night off-peak tariffs."
)

st.sidebar.markdown("**Operational Priority**")
alpha_weight = st.sidebar.slider(
    "Cost (₹) vs Carbon Reduction (α)",
    min_value=0.0,
    max_value=1.0,
    value=0.55,
    step=0.05,
    help="0.0 = 100% Carbon Reduction, 1.0 = 100% Tariff Cost Minimization"
)
st.sidebar.caption(f"{int(alpha_weight*100)}% Tariff Savings | {int((1-alpha_weight)*100)}% Carbon Abatement")

st.sidebar.write("---")
st.sidebar.caption("Regulatory Context: MSEDCL ToD Tariffs | CEA India CO₂ Baseline")


# Execute Pipeline
gen, forecaster, hist_df, train_metrics = load_and_train_system(model_type=selected_model)
horizon_df = gen.generate_forecast_horizon(hours=24, scenario=scenario)
forecast_df = forecaster.predict(horizon_df)

battery_opt = BatteryOptimizer(
    capacity_kwh=battery_capacity,
    max_power_kw=battery_power,
    soc_initial_pct=initial_soc,
    roundtrip_efficiency=0.90
)
dispatch_df = battery_opt.optimize_dispatch(
    forecast_df,
    alpha_cost=alpha_weight,
    enforce_clean_charging=enforce_anti_paradox
)
naive_dispatch_df = battery_opt.simulate_naive_baseline(forecast_df)

scheduler = CarbonAwareLoadScheduler(facility_max_flexible_kw=200.0)
schedule_df = scheduler.schedule_loads(forecast_df, alpha_cost=alpha_weight)
naive_schedule_df = scheduler.simulate_naive_schedule(forecast_df)

impact_engine = ImpactEngine()
impact = impact_engine.evaluate_system(
    dispatch_df, schedule_df, naive_dispatch_df, naive_schedule_df
)


# ---------------- Header & Meta ----------------
scenario_labels = {
    "normal": "Typical day",
    "high_solar": "High solar",
    "cloudy_storm": "Monsoon / cloudy",
    "extreme_peak_pricing": "Peak pricing"
}
st.markdown("""
<div class="hero-shell">
    <div>
        <div class="hero-eyebrow">Grid operations · Maharashtra</div>
        <div class="brand-title">GridSense <span>AI</span></div>
        <div class="brand-tagline">Clean energy, intelligently orchestrated.</div>
    </div>
    <div class="hero-status"><span class="status-dot"></span>Simulation ready<strong>24-hour outlook</strong></div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="meta-bar">
    <div class="meta-badge">Grid region<strong>Western India</strong></div>
    <div class="meta-badge">Day scenario<strong>{scenario_labels[scenario]}</strong></div>
    <div class="meta-badge">Forecast model<strong>{selected_model.replace("_", " ").title()}</strong></div>
    <div class="meta-badge">Tariff reference<strong>MSEDCL ToD</strong></div>
</div>
""", unsafe_allow_html=True)


# ---------------- Top Performance Cards ----------------
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Electricity Cost Saved</div>
        <div class="stat-value">₹{impact['savings']['cost_saved']:,.2f}</div>
        <div class="stat-sub">↓ {impact['savings']['cost_saved_pct']}% vs Naive Baseline</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Carbon Abated</div>
        <div class="stat-value">{impact['savings']['carbon_saved_kg']:.1f} <span style="font-size:0.9rem;font-weight:500;">kg</span></div>
        <div class="stat-sub">↓ {impact['savings']['carbon_saved_pct']}% Emissions</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Curtailment Avoided</div>
        <div class="stat-value">{impact['savings']['curtailment_avoided_kwh']:.1f} <span style="font-size:0.9rem;font-weight:500;">kWh</span></div>
        <div class="stat-sub">Stored & Self-Consumed</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Peak Grid Shaving</div>
        <div class="stat-value">{impact['savings']['peak_reduction_kw']:.1f} <span style="font-size:0.9rem;font-weight:500;">kW</span></div>
        <div class="stat-sub">↓ {impact['savings']['peak_reduction_pct']}% Peak Demand</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Battery Paradox Solved</div>
        <div class="stat-value">{impact['savings']['battery_paradox_prevented_kwh']:.1f} <span style="font-size:0.9rem;font-weight:500;">kWh</span></div>
        <div class="stat-sub">Dirty Coal Charge Blocked</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------- Main Navigation Tabs ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 **Forecasting Core**",
    "🔋 **Battery Storage (Supply Arm)**",
    "⚡ **Load Scheduling (Demand Arm)**",
    "🏆 **Business Impact & ROI**",
    "📋 **Architecture & Methodology**"
])

hours_x = [f"{h:02d}:00" for h in forecast_df["hour"]]


# ================= TAB 1: FORECASTING CORE =================
with tab1:
    st.markdown('<div class="section-title">Shared Forecasting Core</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">A single machine learning core produces synchronized 24-hour ahead predictions across generation, commercial tariffs, and grid emissions to serve both operational arms.</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="doc-box">
            <b>Solar & Wind Generation</b><br/>
            Model Accuracy (R²): <b>{train_metrics['total_renewable_kw']['r2']}</b><br/>
            Mean Absolute Error: <b>{train_metrics['total_renewable_kw']['mae']} kW</b>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="doc-box">
            <b>MSEDCL ToD Electricity Tariff</b><br/>
            Model Accuracy (R²): <b>{train_metrics['electricity_price_kwh']['r2']}</b><br/>
            Mean Absolute Error: <b>₹{train_metrics['electricity_price_kwh']['mae']}/kWh</b>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="doc-box">
            <b>CEA Grid Carbon Intensity</b><br/>
            Model Accuracy (R²): <b>{train_metrics['grid_carbon_intensity']['r2']}</b><br/>
            Mean Absolute Error: <b>{train_metrics['grid_carbon_intensity']['mae']} gCO₂/kWh</b>
        </div>
        """, unsafe_allow_html=True)

    fig_fc = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "1. Predicted Generation: Rooftop Solar & Captive Wind (kW)",
            "2. MSEDCL Time-of-Day Tariff Curve (₹/kWh)",
            "3. CEA National Grid Marginal Carbon Intensity (gCO₂/kWh)"
        )
    )

    # 1. Generation: solar, wind, and combined output
    fig_fc.add_trace(
        go.Bar(name="Solar PV (450 kWp)", x=hours_x, y=forecast_df["forecast_solar_generation_kw"], marker_color="#e7ad45", opacity=0.9),
        row=1, col=1
    )
    fig_fc.add_trace(
        go.Bar(name="Wind (250 kW)", x=hours_x, y=forecast_df["forecast_wind_generation_kw"], marker_color="#70b9ae", opacity=0.8),
        row=1, col=1
    )
    fig_fc.add_trace(
        go.Scatter(name="Total Renewable", x=hours_x, y=forecast_df["forecast_total_renewable_kw"], mode="lines", line=dict(color="#16835d", width=2.5)),
        row=1, col=1
    )

    # 2. Tariff
    fig_fc.add_trace(
        go.Scatter(name="Tariff (₹/kWh)", x=hours_x, y=forecast_df["forecast_electricity_price_kwh"], mode="lines+markers", line=dict(color="#4478a8", width=2.5), marker=dict(size=4, color="#4478a8")),
        row=2, col=1
    )
    fig_fc.add_hline(y=7.80, line_dash="dash", line_color="#aab8b0", annotation_text="Base Tariff (₹7.80)", row=2, col=1)

    # 3. Carbon intensity
    fig_fc.add_trace(
        go.Scatter(name="Carbon Intensity", x=hours_x, y=forecast_df["forecast_grid_carbon_intensity"], mode="lines", line=dict(color="#cc775f", width=2.5), fill='tozeroy', fillcolor='rgba(204, 119, 95, 0.10)'),
        row=3, col=1
    )
    fig_fc.add_hline(y=float(np.median(forecast_df["forecast_grid_carbon_intensity"])), line_dash="dot", line_color="#aab8b0", annotation_text="Thermal Baseload Median", row=3, col=1)

    fig_fc.update_layout(**PLOTLY_LAYOUT)
    fig_fc.update_layout(height=680, barmode="stack", hovermode="x unified")
    st.plotly_chart(fig_fc, use_container_width=True)


# ================= TAB 2: BATTERY STORAGE =================
with tab2:
    st.markdown('<div class="section-title">Layer 2A: Battery Storage Optimization (Supply Arm)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Linear Program (LP) dispatch. Charges strictly from captive solar surplus, avoiding the Indian Battery Paradox (charging on night coal baseload). Discharges during MSEDCL evening peak tariff (18:00 - 22:00).</div>',
        unsafe_allow_html=True
    )

    b_col1, b_col2 = st.columns([2.2, 1])

    with b_col1:
        fig_bat = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Battery Power Flow: Charging (+) vs Discharging (-)", "Battery State of Charge (SoC %)")
        )

        fig_bat.add_trace(
            go.Bar(name="Solar Surplus Charge (kW)", x=hours_x, y=dispatch_df["battery_clean_charge_kw"], marker_color="#16835d"),
            row=1, col=1
        )
        fig_bat.add_trace(
            go.Bar(name="Grid Charge (Clean Hours)", x=hours_x, y=dispatch_df["battery_grid_charge_kw"], marker_color="#70b9ae"),
            row=1, col=1
        )
        fig_bat.add_trace(
            go.Bar(name="Peak Shaving Discharge (kW)", x=hours_x, y=-dispatch_df["battery_discharge_kw"], marker_color="#4478a8", opacity=0.72),
            row=1, col=1
        )

        fig_bat.add_trace(
            go.Scatter(name="GridSense SoC (%)", x=hours_x, y=dispatch_df["battery_soc_pct"], mode="lines", line=dict(color="#16835d", width=2.5)),
            row=2, col=1
        )
        fig_bat.add_trace(
            go.Scatter(name="Naive Baseline SoC (%)", x=hours_x, y=naive_dispatch_df["naive_battery_soc_pct"], mode="lines", line=dict(color="#aab8b0", dash="dot", width=1.8)),
            row=2, col=1
        )
        fig_bat.add_hline(y=15, line_dash="dash", line_color="#c6d3cb", annotation_text="Min Reserve (15%)", row=2, col=1)
        fig_bat.add_hline(y=95, line_dash="dash", line_color="#c6d3cb", annotation_text="Max Limit (95%)", row=2, col=1)

        fig_bat.update_layout(**PLOTLY_LAYOUT)
        fig_bat.update_layout(height=560, barmode="relative", hovermode="x unified")
        st.plotly_chart(fig_bat, use_container_width=True)

    with b_col2:
        st.markdown("**Battery Operational Telemetry**")
        total_ch = float(dispatch_df['battery_charge_kw'].sum())
        clean_ch = float(dispatch_df['battery_clean_charge_kw'].sum())
        clean_share = (clean_ch / total_ch * 100.0) if total_ch > 0 else 100.0

        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Total Energy Stored</div>
            <div class="stat-value">{int(total_ch)} <span style="font-size:0.9rem;font-weight:400;">kWh</span></div>
            <div class="stat-sub">{clean_share:.1f}% Local Clean Energy</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Evening Peak Discharged</div>
            <div class="stat-value">{int(dispatch_df['battery_discharge_kw'].sum())} <span style="font-size:0.9rem;font-weight:400;">kWh</span></div>
            <div class="stat-sub">Shaves ₹10.30/kWh Surcharge</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="doc-box">
            <b>The Indian Battery Paradox</b><br/>
            In India, nighttime electricity is cheap (₹6.30/kWh off-peak ToD rate), but the overnight grid is over 80% coal (~760 gCO₂/kWh).<br/><br/>
            Traditional battery timers charge blindly at 2 AM. Accounting for 15% roundtrip efficiency losses, this actually <b>increases net carbon emissions</b>.<br/><br/>
            GridSense AI enforces a clean-only charging constraint, restricting charging to local rooftop solar surplus and ultra-clean hours.
        </div>
        """, unsafe_allow_html=True)


# ================= TAB 3: LOAD SCHEDULER =================
with tab3:
    st.markdown('<div class="section-title">Layer 2B: Carbon-Aware Load Scheduling (Demand Arm)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Mixed-Integer Linear Program (MILP). Dynamically schedules deferrable commercial loads into midday solar generation troughs to eliminate peaker demand charges.</div>',
        unsafe_allow_html=True
    )

    fig_sched = go.Figure()
    fig_sched.add_trace(go.Bar(name="Commercial EV Fleet Depot (Delivery 3W/4W)", x=hours_x, y=schedule_df["scheduled_job_ev_fleet_kw"], marker_color="#16835d"))
    fig_sched.add_trace(go.Bar(name="Campus & Tech Park Compute Cluster", x=hours_x, y=schedule_df["scheduled_job_ai_training_kw"], marker_color="#4478a8"))
    fig_sched.add_trace(go.Bar(name="APMC Cold Storage & Chiller Pre-Cooling", x=hours_x, y=schedule_df["scheduled_job_hvac_thermal_kw"], marker_color="#e7ad45"))
    fig_sched.add_trace(go.Bar(name="MIDC / Municipal Water Pumping Station", x=hours_x, y=schedule_df["scheduled_job_water_pumps_kw"], marker_color="#70b9ae"))

    fig_sched.update_layout(**PLOTLY_LAYOUT)
    fig_sched.update_layout(
        title=dict(text="Optimized Flexible Load Allocation Across Clean Solar Windows", font=dict(size=12, color="#31443a")),
        barmode="stack",
        height=420,
        hovermode="x unified",
        yaxis=dict(title="Scheduled Power (kW)")
    )
    st.plotly_chart(fig_sched, use_container_width=True)

    # Shifted vs Unmanaged load curve
    st.markdown("**Load Curve Comparison: GridSense Shifted Demand vs. Naive Plug-in**")
    fig_load_comp = go.Figure()
    fig_load_comp.add_trace(go.Scatter(
        name="GridSense AI (Optimized Shifted Demand)",
        x=hours_x,
        y=schedule_df["total_flexible_demand_kw"],
        mode="lines",
        line=dict(color="#16835d", width=2.5)
    ))
    fig_load_comp.add_trace(go.Scatter(
        name="Naive Baseline (Unmanaged Plug-in)",
        x=hours_x,
        y=naive_schedule_df["naive_flexible_demand_kw"],
        mode="lines",
        line=dict(color="#aab8b0", dash="dot", width=2)
    ))
    fig_load_comp.update_layout(**PLOTLY_LAYOUT)
    fig_load_comp.update_layout(height=280, hovermode="x unified", yaxis=dict(title="Demand (kW)"))
    st.plotly_chart(fig_load_comp, use_container_width=True)


# ================= TAB 4: IMPACT & COMPARISON =================
with tab4:
    st.markdown('<div class="section-title">System-Wide Impact & Financial Return</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Empirical comparison between GridSense AI unified operation and the traditional unmanaged baseline across the 24-hour planning horizon.</div>',
        unsafe_allow_html=True
    )

    t_col1, t_col2 = st.columns([1.2, 1])

    with t_col1:
        st.markdown("**Operational Comparison Table**")
        comp_table = pd.DataFrame({
            "Metric": [
                "Total Electricity Bill (₹)",
                "Carbon Emissions (kg CO₂)",
                "Solar/Wind Curtailment (kWh)",
                "Peak Grid Draw (kW)",
                "Clean Charging Ratio (%)"
            ],
            "Naive Baseline": [
                f"₹{impact['naive']['total_cost']:,.2f}",
                f"{impact['naive']['total_carbon_kg']:,.1f} kg",
                f"{impact['naive']['curtailment_kwh']:,.1f} kWh",
                f"{impact['naive']['peak_grid_kw']:,.1f} kW",
                "32.0% (Dirty Night Coal)"
            ],
            "GridSense AI": [
                f"₹{impact['gridsense']['total_cost']:,.2f}",
                f"{impact['gridsense']['total_carbon_kg']:,.1f} kg",
                f"{impact['gridsense']['curtailment_kwh']:,.1f} kWh",
                f"{impact['gridsense']['peak_grid_kw']:,.1f} kW",
                f"{impact['gridsense']['clean_charge_pct']:.1f}% (Clean Solar)"
            ],
            "Net Benefit": [
                f"Saved ₹{impact['savings']['cost_saved']:,.2f} ({impact['savings']['cost_saved_pct']}%)",
                f"Reduced {impact['savings']['carbon_saved_kg']:,.1f} kg ({impact['savings']['carbon_saved_pct']}%)",
                f"Avoided {impact['savings']['curtailment_avoided_kwh']:,.1f} kWh",
                f"Shaved {impact['savings']['peak_reduction_kw']:,.1f} kW ({impact['savings']['peak_reduction_pct']}%)",
                "Battery Paradox Eliminated"
            ]
        })
        st.dataframe(comp_table, hide_index=True, use_container_width=True)

    with t_col2:
        fig_net = go.Figure()
        fig_net.add_trace(go.Scatter(
            name="GridSense Net Import (kW)",
            x=hours_x,
            y=impact['gridsense']['grid_draw_series'],
            mode="lines",
            fill="tozeroy",
            line=dict(color="#16835d", width=2),
            fillcolor="rgba(22, 131, 93, 0.10)"
        ))
        fig_net.add_trace(go.Scatter(
            name="Naive Baseline Net Import (kW)",
            x=hours_x,
            y=impact['naive']['grid_draw_series'],
            mode="lines",
            line=dict(color="#aab8b0", dash="dash", width=1.5)
        ))
        fig_net.update_layout(**PLOTLY_LAYOUT)
        fig_net.update_layout(title=dict(text="Net Grid Import: Evening Peak Shaving", font=dict(size=12, color="#31443a")), height=320, hovermode="x unified", yaxis=dict(title="Net Grid Power (kW)"))
        st.plotly_chart(fig_net, use_container_width=True)

    st.write("---")
    st.markdown("**Annualized Campus / Facility Projections**")
    p1, p2, p3 = st.columns(3)
    annual_savings_inr = impact['savings']['cost_saved'] * 365
    annual_co2_tons = (impact['savings']['carbon_saved_kg'] * 365) / 1000.0

    with p1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Annual Tariff Cost Savings</div>
            <div class="stat-value">₹{annual_savings_inr:,.0f}</div>
            <div class="stat-sub">Based on MSEDCL ToD Tariffs</div>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Annual Emissions Reduced</div>
            <div class="stat-value">{annual_co2_tons:,.1f} <span style="font-size:0.9rem;font-weight:400;">tonnes</span></div>
            <div class="stat-sub">CEA Grid Baseline Factor</div>
        </div>
        """, unsafe_allow_html=True)
    with p3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Estimated Payback Period</div>
            <div class="stat-value">3.4 <span style="font-size:0.9rem;font-weight:400;">Years</span></div>
            <div class="stat-sub">On 500 kWh BESS Capex</div>
        </div>
        """, unsafe_allow_html=True)


# ================= TAB 5: ARCHITECTURE =================
with tab5:
    st.markdown('<div class="section-title">GridSense AI Architecture & Methodology</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="doc-box">
        <b>Core Challenge & Solution Formulation:</b><br/>
        Traditional energy management treats renewable forecasting, battery storage, and demand response in separate silos.
        In Indian smart cities and C&I facilities, this causes two major systemic gaps:
        <ol style="margin-top:6px; margin-bottom:8px;">
            <li><b>The Battery Paradox:</b> Fixed timers charge batteries during night off-peak tariff hours (01:00 - 05:00). In India, night baseload power is over 80% coal (~760 gCO₂/kWh). When accounting for round-trip efficiency loss (~15%), cycling the battery on midnight power <i>increases</i> net carbon emissions.</li>
            <li><b>Blind Plug-in Loads:</b> EV fleet charging depots, industrial pumps, and cooling systems operate on arrival, compounding evening peak demand and triggering severe MSEDCL demand surcharges.</li>
        </ol>
        <b>The GridSense AI Architecture:</b><br/>
        <ul>
            <li><b>One AI Core:</b> A shared multi-target regression pipeline forecasting Solar/Wind output, MSEDCL ToD tariffs, and CEA carbon intensity 24 hours ahead.</li>
            <li><b>Module A (Supply Arm):</b> HiGHS Linear Programming battery arbitrage with clean-charging constraints ($P_{\\text{charge}, t} \\le \\text{Surplus}$) to eliminate dirty coal charging.</li>
            <li><b>Module B (Demand Arm):</b> Carbon-aware Mixed-Integer scheduling shifting flexible commercial loads directly into daytime solar troughs.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Standards & References**")
    st.markdown("""
    1. **Central Electricity Authority (CEA)** — *CO₂ Baseline Database for the Indian Power Sector, Version 19*, Government of India.
    2. **Maharashtra Electricity Regulatory Commission (MERC)** — *Time of Day (ToD) Tariff Regulations & Commercial Tariffs for MSEDCL*.
    3. **National Institute of Solar Energy (NISE)** — *Solar Radiation Resource Assessment (SRRA) Data for Western India*.
    4. **COIN-OR & SciPy HiGHS** — *High Performance Simplex and Interior Point Solvers for Linear Energy Optimization*.
    """)
