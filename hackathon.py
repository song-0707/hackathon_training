import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import joblib

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="TerraTwin: UTM KL Resilience Center", page_icon="🎓", layout="wide")

st.markdown("""
    <style>
    /* 1. Force Black Text on Metrics */
    div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {
        color: #000000 !important;
    }
    .stMetric {
        background-color: #f0f2f6 !important;
        border-left: 5px solid #800000; /* UTM Maroon Color */
        padding: 10px;
        border-radius: 10px;
    }
    /* 2. Make Sidebar readable in Dark Mode */
    .streamlit-expanderHeader {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    .streamlit-expanderContent {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)


# --- 2. LOAD BRAIN ---
@st.cache_resource
def load_data():
	try:
		return joblib.load('climate_model.pkl')
	except:
		return None


model = load_data()


# --- 3. CALCULATOR ENGINES ---
def get_prediction(inputs):
	if model:
		# Match the 12 features from training
		input_df = pd.DataFrame([inputs])
		depth = model.predict(input_df)[0]
		return max(0, round(depth, 2))
	return 0


def calculate_financials(depth, industrial_pct):
	base_cost_per_cm = 50000
	industrial_multiplier = 1 + (industrial_pct / 100) * 5
	total_loss = depth * base_cost_per_cm * industrial_multiplier
	return round(total_loss / 1000000, 2)


def calculate_logistics(depth):
	if depth > 50:
		return "⛔ TOTAL SHUTDOWN"
	elif depth > 30:
		return "⚠️ HEAVY DELAYS"
	else:
		return "✅ OPERATIONAL"


def calculate_carbon(mangroves):
	carbon_tons = mangroves * 500
	credit_value = carbon_tons * 30
	return int(carbon_tons), int(credit_value)


# --- 4. MAP ENGINE (UTM KL COORDINATES) ---
def render_map(depth, mangroves, logistics_status):
	# CENTER: UTM KL Campus (Jalan Sultan Yahya Petra)
	utm_lat = 3.1729
	utm_lon = 101.7209

	m = folium.Map(location=[utm_lat, utm_lon], zoom_start=14, tiles="CartoDB positron")

	# 1. Flood Layer (Centered on Campus)
	color = "red" if depth > 50 else "blue"
	radius = 400 + (depth * 10)
	folium.Circle(
		[utm_lat, utm_lon], radius=radius, color=color, fill=True, fill_opacity=0.3,
		popup=f"Flood Level: {depth}cm"
	).add_to(m)

	# 2. Simulated 'Assets' Box (e.g., Nearby Commercial Area/KLCC view)
	# Placing it slightly Southwest of campus
	folium.Rectangle(
		bounds=[[3.165, 101.710], [3.170, 101.718]],
		color="orange", fill=False, weight=3,
		tooltip="Critical Infrastructure (Labs/Server Rooms)"
	).add_to(m)

	# 3. Supply Route (Jalan Tun Razak / AKLEH Simulation)
	road_color = "green" if logistics_status == "✅ OPERATIONAL" else "black"
	folium.PolyLine(
		locations=[[3.180, 101.715], [3.1729, 101.7209], [3.160, 101.725]],
		color=road_color, weight=5, tooltip=f"Main Access Road: {logistics_status}"
	).add_to(m)

	# 4. Green Interventions (Trees on Campus)
	if mangroves > 0:
		np.random.seed(42)
		for _ in range(int(mangroves / 2)):
			# Scatter trees around the campus coordinates
			folium.Marker(
				[utm_lat + np.random.uniform(-0.005, 0.005), utm_lon + np.random.uniform(-0.005, 0.005)],
				icon=folium.Icon(color="green", icon="leaf")
			).add_to(m)

	return m


# --- 5. UI LAYOUT ---
st.title("🎓 TerraTwin: UTM KL Resilience Center")

# --- SIDEBAR (INPUTS WITH FORM) ---
with st.sidebar:
	st.header("🎛️ Simulation Controls")

	with st.form("main_form"):
		with st.expander("1. Hydrology (Water)", expanded=True):
			rain = st.slider("Rainfall (mm/hr)", 0, 300, 150)
			tide = st.slider("River Level (m)", 0.0, 5.0, 1.5, help="Gombak/Klang River Level")
			river = st.slider("Upstream Flow (m3/s)", 10, 500, 100)
			soil = st.slider("Soil Saturation (%)", 0, 100, 80)

		with st.expander("2. Urban Features"):
			concrete = st.slider("Impervious Surface (%)", 0, 100, 85, help="KL is very concrete-heavy")
			ind_mix = st.slider("Asset Density (%)", 0, 100, 70)

		with st.expander("3. Mitigation (Solutions)"):
			mangrove = st.slider("🌳 Urban Forest (Ha)", 0, 50, 0)
			ponds = st.slider("💧 Retention Ponds", 0, 50, 0)
			pavement = st.slider("🧱 Permeable Pavement (%)", 0, 100, 0)

		submitted = st.form_submit_button("🚀 SIMULATE SCENARIO")

# --- LOGIC EXECUTION ---
inputs = {
	'rainfall': rain, 'tide': tide, 'river_flow': river, 'soil_saturation': soil,
	'elevation': 5, 'impervious_surface': concrete, 'distance_river': 200,  # Closer to river in KL
	'drainage_efficiency': 50, 'road_density': 80,  # High density in KL
	'mangroves': mangrove, 'retention_ponds': ponds, 'permeable_pavement': pavement
}

depth = get_prediction(inputs)
loss_millions = calculate_financials(depth, ind_mix)
logistics = calculate_logistics(depth)
co2, credit_revenue = calculate_carbon(mangrove)

# --- DASHBOARD DISPLAY ---

k1, k2, k3, k4 = st.columns(4)
k1.metric("🌊 Flood Depth", f"{depth} cm", delta="Critical" if depth > 50 else "Safe", delta_color="inverse")
k2.metric("💸 Campus Damage", f"${loss_millions}M")
k3.metric("🚗 Access Roads", logistics, delta="Blocked" if logistics != "✅ OPERATIONAL" else "OK", delta_color="inverse")
k4.metric("🌍 Green Offset", f"+${credit_revenue:,}")

st.divider()

# Tabs
tab_map, tab_charts, tab_ai = st.tabs(["🗺️ UTM KL Digital Twin", "📊 Cost Analysis", "🤖 AI Recommendations"])

with tab_map:
	st.subheader("📍 Live Campus Situation Map (Jalan Semarak)")
	map_obj = render_map(depth, mangrove, logistics)
	st_folium(map_obj, width=1200, height=500)

	if logistics == "⛔ TOTAL SHUTDOWN":
		st.error("🚨 **ALERT:** Jalan Tun Razak is flooded. Campus is inaccessible.")

with tab_charts:
	c1, c2 = st.columns(2)
	with c1:
		st.subheader("💰 Damage vs. Mitigation Cost")
		chart_data = pd.DataFrame({
			"Category": ["Projected Damage", "Solution Cost"],
			"Amount ($M)": [loss_millions, (mangrove * 0.2 + ponds * 0.5)]
		})
		st.bar_chart(chart_data.set_index("Category"))

	with c2:
		st.subheader("📉 Risk Reduction Projection")
		years = list(range(2024, 2035))
		base_risk = [depth * (1 + i * 0.06) for i in range(len(years))]
		mitigated_risk = [depth * (0.88 ** i) for i in range(len(years))] if mangrove > 0 else base_risk
		line_data = pd.DataFrame({"Year": years, "No Action": base_risk, "With Solution": mitigated_risk})
		st.line_chart(line_data.set_index("Year"))

with tab_ai:
	st.subheader("🤖 AI Strategy Report")
	if depth > 80:
		st.error("Scenario: HIGH RISK. The Gombak River overflow is threatening campus basements.")
	else:
		st.success(f"Scenario: STABLE. Urban forestry is effectively absorbing surface runoff.")