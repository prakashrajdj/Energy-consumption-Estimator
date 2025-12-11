# app.py
# POLISHED — Fluid UI with pill-style theme picker, global Banschrift font, white text
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import time
import os
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Exclusive ML APP", layout="wide", initial_sidebar_state="expanded")

# --------------------------
# Theme palette definitions
# --------------------------
THEMES = {
    "Ocean": {
        "bg1": "#021627", "bg2": "#053642",
        "accent1": "#1fe4ff", "accent2": "#3b82f6",
        "muted": "#cfeefb", "glass": "rgba(255,255,255,0.06)"
    },
    "Sunset": {
        "bg1": "#2b0210", "bg2": "#3b0e0f",
        "accent1": "#ff7e5f", "accent2": "#feb47b",
        "muted": "#ffd6c2", "glass": "rgba(255,255,255,0.04)"
    },
    "Midnight": {
        "bg1": "#020617", "bg2": "#0f1724",
        "accent1": "#7c3aed", "accent2": "#06b6d4",
        "muted": "#9ca3af", "glass": "rgba(255,255,255,0.02)"
    },
    "Forest": {
        "bg1": "#021e1a", "bg2": "#0b4d3a",
        "accent1": "#9be7c4", "accent2": "#3ddc84",
        "muted": "#cfeede", "glass": "rgba(255,255,255,0.04)"
    },
    "Neon": {
        "bg1": "#050006", "bg2": "#0f0426",
        "accent1": "#ff3cac", "accent2": "#00ffd5",
        "muted": "#f3e8ff", "glass": "rgba(255,255,255,0.02)"
    }
}

# --------------------------
# Sidebar: Polished pill-style theme picker
# --------------------------
st.sidebar.markdown("## Theme")
theme_selected = st.sidebar.radio("", list(THEMES.keys()), index=0, label_visibility="collapsed")

# Decorative pill row (polished, with icons)
def render_pill_themes(selected):
    html = '<div style="display:flex; gap:10px; flex-wrap:wrap; padding-top:6px;">'
    for name, cfg in THEMES.items():
        is_active = (name == selected)
        transform = "translateY(-3px)" if is_active else "none"
        box_shadow = "0 10px 30px rgba(0,0,0,0.35)" if is_active else "0 6px 18px rgba(0,0,0,0.12)"
        opacity = "1" if is_active else "0.85"
    st.sidebar.markdown(html, unsafe_allow_html=True)

render_pill_themes(theme_selected)

# --------------------------
# SVG background based on theme
# --------------------------
cfg = THEMES[theme_selected]
svg = f'''
<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" preserveAspectRatio="none">
  <defs>
    <linearGradient id="g" x1="0" x2="1">
      <stop offset="0%" stop-color="{cfg['bg1']}" stop-opacity="1"/>
      <stop offset="100%" stop-color="{cfg['bg2']}" stop-opacity="1"/>
    </linearGradient>
    <linearGradient id="acc" x1="0" x2="1">
      <stop offset="0%" stop-color="{cfg['accent1']}" stop-opacity="0.95"/>
      <stop offset="100%" stop-color="{cfg['accent2']}" stop-opacity="0.95"/>
    </linearGradient>
    <filter id="blurf" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="60" result="b"/>
      <feBlend in="SourceGraphic" in2="b"/>
    </filter>
  </defs>
  <rect width="100%" height="100%" fill="url(#g)"/>
  <g opacity="0.42" filter="url(#blurf)">
    <circle cx="18%" cy="18%" r="260" fill="url(#acc)"/>
    <circle cx="86%" cy="78%" r="320" fill="url(#acc)"/>
  </g>
</svg>
'''
svg_b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
svg_uri = f"data:image/svg+xml;base64,{svg_b64}"

# --------------------------
# Global CSS — Banschrift font & white text
# --------------------------
st.markdown(f"""
<style>
@font-face {{
  font-family: 'Banschrift';
  src: local('Banschrift'), local('Banscrit'), local('Banscriht');
  font-weight: 700;
  font-style: normal;
  color: #0b1220;
}}
:root {{
  --accent1: {cfg['accent1']};
  --accent2: {cfg['accent2']};
  --muted: {cfg['muted']};
  --glass: {cfg['glass']};
  --text: #ffffff;
}}
.stApp {{
  background-image: url("{svg_uri}");
  background-size: cover;
  background-attachment: fixed;
  color: var(--text) !important;
  font-family: 'Banschrift', 'Segoe UI', Roboto, Arial, sans-serif;
}}
/* glass panels */
.glass {{
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border-radius: 14px;
  padding: 16px;
  color: var(--text) !important;
  box-shadow: 0 10px 30px rgba(2,6,23,0.25);
}}
.header-title {{
  font-weight:900;
  font-size:28px;
  color:var(--text) !important;
}}
.sub-text {{
  color: var(--muted) !important;
}}
.pill {{
  display:inline-flex; align-items:center; gap:8px;
  padding:8px 14px; border-radius:999px; font-weight:800;
  color:var(--text) !important;
}}
.kpi {{
  font-size:34px; font-weight:900; color:var(--accent2);
}}
.small-muted {{ color:var(--muted) !important; font-size:13px; }}
div.stButton > button {{
  background: linear-gradient(90deg, var(--accent1), var(--accent2)) !important;
  color: white !important;
  border-radius:10px;
  height:44px;
  font-weight:800;
}}
.stDataFrame table, .stTable table {{
  color: #0b1220;
}}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Header
# --------------------------
st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center;">
  <div>
    <div class="header-title">🏡 Smart Home Energy Consumption Estimator </div>
    <div class="sub-text">Fluid UI</div>
  </div>
  <div>
    <div style="display:flex; gap:10px; align-items:center;">
      <div class="pill" style="background:linear-gradient(90deg,{cfg['accent1']},{cfg['accent2']}); box-shadow:0 8px 30px rgba(0,0,0,0.25);">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" style="margin-right:8px;">
          <circle cx="12" cy="12" r="10" fill="white" opacity="0.12"/>
        </svg>
        <div style="font-size:13px;">Theme: {theme_selected}</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# --------------------------
# Model loader (robust)
# --------------------------
@st.cache_resource
def load_model():
    candidates = [Path("model.pkl"), Path("ecommerce_model.pkl"), Path("energy_model.pkl"),
                  Path("model.pickle"), Path("/mnt/data/model.pkl")]
    for p in candidates:
        if p.exists():
            try:
                with open(p, "rb") as f:
                    return pickle.load(f), p
            except Exception:
                continue
    for p in Path(".").glob("*.pkl"):
        try:
            with open(p, "rb") as f:
                return pickle.load(f), p
        except Exception:
            continue
    return None, None

obj, model_path = load_model()
if obj is None:
    st.error("No model pickle found in this folder. Place your trained pipeline pickle here (model.pkl).")
    st.stop()

st.markdown(f"**Loaded:** `{model_path.name}`")

def extract_pipeline(obj):
    pipeline = None; raw_columns = []; target = None
    if isinstance(obj, dict):
        if "model" in obj and hasattr(obj["model"], "predict"):
            pipeline = obj["model"]
        elif "pipeline" in obj and hasattr(obj["pipeline"], "predict"):
            pipeline = obj["pipeline"]
        else:
            for v in obj.values():
                if hasattr(v, "predict"):
                    pipeline = v; break
        for k in ("raw_columns", "columns", "feature_names"):
            if k in obj:
                raw_columns = obj[k]; break
        target = obj.get("target", None)
    else:
        if hasattr(obj, "predict"):
            pipeline = obj
    raw_columns = list(raw_columns) if raw_columns else []
    return pipeline, raw_columns, target

pipeline, raw_columns, target = extract_pipeline(obj)
if pipeline is None:
    st.error("No pipeline/estimator found inside pickle.")
    st.stop()

# --------------------------
# KPI row & gauges
# --------------------------
c1, c2, c3 = st.columns([1.2, 1, 1])

baseline = None
if raw_columns:
    try:
        sample = pd.DataFrame([{c: 0 for c in raw_columns}])
        baseline = float(pipeline.predict(sample)[0])
    except Exception:
        baseline = None

with c1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.write("Baseline prediction")
    st.markdown(f"<div class='kpi'>{baseline:.1f} kWh</div>" if baseline is not None else "N/A", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.write("Model")
    if hasattr(pipeline, "steps"):
        st.write("Pipeline: " + " → ".join([n for n,_ in pipeline.steps]))
    else:
        st.write(type(pipeline).__name__)
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.write("Target")
    st.write(target if target else "Not embedded")
    st.markdown("</div>", unsafe_allow_html=True)

# Gauges
g1, g2 = st.columns(2)
with g1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.write("Consumption Gauge")
    val = baseline if baseline is not None else 50
    fig = go.Figure(go.Indicator(mode="gauge+number", value=val,
                                 gauge={'axis':{'range':[0, max(val*2,200)]}, 'bar':{'color':cfg['accent2']}}))
    fig.update_layout(height=240, margin=dict(l=10,r=10,t=20,b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with g2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.write("Efficiency Gauge")
    eff = 72
    fig2 = go.Figure(go.Indicator(mode="gauge+number", value=eff,
                                  gauge={'axis':{'range':[0,100]}, 'bar':{'color':cfg['accent1']}}))
    fig2.update_layout(height=240, margin=dict(l=10,r=10,t=20,b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# Main tabs: Single and Batch
# --------------------------
tab1, tab2 = st.tabs(["Single Prediction", "Batch & Visuals"])

with tab1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.header("Single prediction — enter raw features")
    if raw_columns:
        left, right = st.columns(2)
        inputs = {}
        for i, c in enumerate(raw_columns):
            if i % 2 == 0:
                inputs[c] = left.text_input(c, value="")
            else:
                inputs[c] = right.text_input(c, value="")
    else:
        st.info("No feature names embedded. Paste comma-separated names or upload CSV in Batch tab.")
        names = st.text_area("Paste feature names (comma-separated)", height=80)
        inputs = {}
        if names.strip():
            raw_columns = [s.strip() for s in names.split(",") if s.strip()]
            left, right = st.columns(2)
            for i, c in enumerate(raw_columns):
                if i % 2 == 0:
                    inputs[c] = left.text_input(c, value="")
                else:
                    inputs[c] = right.text_input(c, value="")

    if st.button("Predict now"):
        if not raw_columns:
            st.error("No feature names available.")
        else:
            row = {}
            for c in raw_columns:
                v = inputs.get(c, "")
                if v is None or str(v).strip()=="":
                    row[c] = np.nan
                else:
                    try:
                        row[c] = float(v)
                    except:
                        row[c] = str(v)
            df_row = pd.DataFrame([row], columns=raw_columns)
            try:
                pred = pipeline.predict(df_row)
                val = float(pred[0])
                placeholder = st.empty()
                steps = 22
                start_v = baseline*0.3 if baseline else 0
                for i in range(steps+1):
                    cur = start_v + (val-start_v)*(i/steps)
                    placeholder.markdown(f"<div style='font-size:44px; font-weight:900; color:{cfg['accent2']}'>{cur:,.2f} kWh</div>", unsafe_allow_html=True)
                    time.sleep(0.02)
                if "history" not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append({"time": datetime.now().isoformat(), "pred": val})
                st.success(f"Prediction: {val:.2f} kWh")
            except Exception as e:
                st.error("Prediction failed: " + str(e))
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.header("Batch upload & visuals")
    uploaded = st.file_uploader("Upload CSV with raw features", type=["csv"])
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            st.write("Preview:")
            st.dataframe(df_up.head(5))
            if not raw_columns:
                raw_columns = list(df_up.columns)
                st.success("Detected columns from CSV.")
            missing = [c for c in raw_columns if c not in df_up.columns]
            if missing:
                st.error("Missing columns: " + ", ".join(missing))
            else:
                X = df_up[raw_columns].copy()
                for c in X.columns:
                    X[c] = pd.to_numeric(X[c], errors="ignore")
                preds = pipeline.predict(X)
                df_up["prediction"] = preds
                st.success("Predictions computed")
                st.dataframe(df_up.head(10))
                try:
                    import plotly.express as px
                    fig_hist = px.histogram(df_up, x="prediction", nbins=40, title="Prediction distribution")
                    st.plotly_chart(fig_hist, use_container_width=True)
                except Exception:
                    pass
                csv = df_up.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", csv, "predictions_with_preds.csv", "text/csv")
        except Exception as e:
            st.error("Failed to process file: " + str(e))
    else:
        st.info("Upload a CSV to batch predict.")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(f'<div style="text-align:center; color:{cfg["muted"]}">Made with Streamlit By Prakashraj </div>', unsafe_allow_html=True)
