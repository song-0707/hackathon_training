import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import joblib

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="TerraTwin: UTM KL Mega-Model", page_icon="🎓", layout="wide")

# --- 2. THEME & STYLING (Dark Mode Friendly) ---
st.markdown("""
    <style>
    /* UTM Maroon Header */
    .main-header { color: #800000; font-size: 2.5rem; font-weight: bold; margin-bottom: 0px; }
    .sub-text { color: #888; margin-bottom: 20px; }

    /* Styled Button */
    div.stButton > button {
        background-color: #800000;
        color: white;
        border-radius: 8px;
        width: 100%;
        height: 3.5em;
        font-weight: bold;
        border: none;
        margin-top: 20px;
    }
    div.stButton > button:hover { background-color: #a00000; color: white; border: 1px solid white; }

    /* Metrics Fix for visibility */
    [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)


# --- 3. CORE ENGINES ---
@st.cache_resource
def load_data():
	try:
		return joblib.load('climate_model.pkl')
	except:
		return None


model = load_data()


def get_prediction(inputs):
	if model:
		input_df = pd.DataFrame([inputs])
		return max(0, round(model.predict(input_df)[0], 2))
	return 0


def calculate_kpis(depth, industrial_pct, mangroves):
	# Financial Loss in Millions
	loss = round((depth * 50000 * (1 + (industrial_pct / 100) * 5)) / 1000000, 2)
	# Logistics Logic
	status = "⛔ SHUTDOWN" if depth > 50 else ("⚠️ DELAYS" if depth > 30 else "✅ ACTIVE")
	# Carbon Finance
	co2 = int(mangroves * 500)
	revenue = int(co2 * 30)
	return loss, status, revenue


# --- 4. SIDEBAR (ALL FEATURES RESTORED) ---
with st.sidebar:
	st.image("https://utm.my/wp-content/uploads/2021/11/UTM-LOGO-FULL.png", width=120)
	st.header("🎛️ Global Parameters")

	# FORM WRAPPER (Prevents Flickering)
	with st.form("mega_form"):
		with st.expander("🌧️ Hydrology (Water)", expanded=True):
			rain = st.slider("Rainfall (mm/hr)", 0, 300, 150)
			tide = st.slider("River/Tide Level (m)", 0.0, 5.0, 1.5)
			river = st.slider("Upstream Flow (m3/s)", 10, 500, 100)
			soil = st.slider("Soil Saturation (%)", 0, 100, 80)

		with st.expander("🏗️ Urban Infrastructure"):
			concrete = st.slider("Impervious Surface (%)", 0, 100, 85)
			ind_mix = st.slider("Industrial/Asset Mix (%)", 0, 100, 60)
			road_dens = st.slider("Road Density (km/km2)", 0, 50, 30)

		with st.expander("🌿 Mitigation Solutions"):
			mangrove = st.slider("Urban Forest (Ha)", 0, 100, 0)
			ponds = st.slider("Retention Ponds", 0, 50, 0)
			pavement = st.slider("Permeable Pavement (%)", 0, 100, 0)

		submitted = st.form_submit_button("🚀 RUN GLOBAL SIMULATION")

# --- 5. LOGIC CALCULATION ---
inputs = {
	'rainfall': rain, 'tide': tide, 'river_flow': river, 'soil_saturation': soil,
	'elevation': 5, 'impervious_surface': concrete, 'distance_river': 200,
	'drainage_efficiency': 60, 'road_density': road_dens,
	'mangroves': mangrove, 'retention_ponds': ponds, 'permeable_pavement': pavement
}

depth = get_prediction(inputs)
loss, logistics, carbon_rev = calculate_kpis(depth, ind_mix, mangrove)

# --- 6. MAIN DASHBOARD ---
st.markdown('<div class="main-header">🎓 TerraTwin: UTM KL Digital Twin</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Gombak River Basin Analysis & Carbon Finance Dashboard</div>',
			unsafe_allow_html=True)

# Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("🌊 Flood Depth", f"{depth} cm", delta="CRITICAL" if depth > 50 else "STABLE", delta_color="inverse")
m2.metric("💸 Economic Loss", f"${loss}M")
m3.metric("🚚 Logistics", logistics)
m4.metric("🌍 Carbon Credit", f"${carbon_rev:,}")

st.divider()

# Maps and Charts
tab1, tab2 = st.tabs(["🗺️ Geospatial Twin", "📊 Analytics Report"])

with tab1:
	utm_lat, utm_lon = 3.1729, 101.7209
	m = folium.Map(location=[utm_lat, utm_lon], zoom_start=15, tiles="CartoDB dark_matter")

	# Flood Zone
	folium.Circle(
		[utm_lat, utm_lon], radius=400 + (depth * 5), color="red" if depth > 40 else "blue",
		fill=True, fill_opacity=0.3, tooltip=f"Risk Area: {depth}cm"
	).add_to(m)

	# Supply Route
	folium.PolyLine(
		locations=[[3.180, 101.715], [3.1729, 101.7209], [3.160, 101.725]],
		color="green" if logistics == "✅ ACTIVE" else "red", weight=5, opacity=0.8
	).add_to(m)

	st_folium(m, width="100%", height=500)

with tab2:
	c1, c2 = st.columns(2)
	with c1:
		st.subheader("💰 Damage vs Solution")
		# Estimate solution cost: $200k per Ha forest, $500k per pond
		cost = (mangrove * 0.2) + (ponds * 0.5)
		st.bar_chart(
			pd.DataFrame({"Category": ["Flood Loss", "Inv. Cost"], "USD (M)": [loss, cost]}).set_index("Category"))

	with c2:
		st.subheader("📈 10-Year Resilience Trend")
		years = list(range(2024, 2035))
		# Logic: If solutions are low, risk increases 5% yearly. If solutions high, risk drops.
		trend = [depth * (1.05 ** i) if mangrove < 10 else depth * (0.9 ** i) for i in range(len(years))]
		st.line_chart(pd.DataFrame({"Year": years, "Projected Depth": trend}).set_index("Year"))

# Final AI Report
if depth > 50:
	st.error(
		f"### 🤖 AI Insight: Critical Breach\nAt {depth}cm, the UTM KL Data Center and lower labs are at risk. The Gombak river flow is exceeding drainage capacity by {int(river * 0.2)}m3/s.")
else:
	st.success(
		f"### 🤖 AI Insight: Resilient\nUrban interventions (Mangrove/Ponds) are providing an absorption buffer of {int(mangrove * 15)}% relative to rainfall volume.")