import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
from io import BytesIO

# --- 1. ADVANCED CONFIGURATION ---
st.set_page_config(
	page_title="TerraTwin: UTM KL Resilience Center Pro",
	page_icon="🎓",
	layout="wide",
	initial_sidebar_state="expanded",
	menu_items={
		'Get Help': 'https://utm.my',
		'Report a bug': 'mailto:terratwin@utm.my',
		'About': "TerraTwin Pro - Advanced Climate Resilience Platform"
	}
)

# Enhanced CSS with Dark Mode Support
st.markdown("""
    <style>
    /* Enhanced Metrics with Animations */
    div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {
        color: #000000 !important;
    }
    .stMetric {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 5px solid #800000;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }

    /* Custom Alert Boxes with Icons */
    .alert-box {
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        animation: slideIn 0.5s ease-out;
        border-left: 5px solid;
    }
    .alert-success {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-color: #28a745;
        color: #155724;
    }
    .alert-warning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-color: #ffc107;
        color: #856404;
    }
    .alert-danger {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-color: #dc3545;
        color: #721c24;
    }
    .alert-info {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-color: #17a2b8;
        color: #0c5460;
    }

    /* Premium Card Design */
    .premium-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }

    /* Stat Badge */
    .stat-badge {
        display: inline-block;
        background: #800000;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
        margin: 0.25rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    /* Animations */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Enhanced Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #800000 0%, #600000 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }

    /* Timeline */
    .timeline-item {
        border-left: 3px solid #800000;
        padding-left: 1.5rem;
        margin-left: 1rem;
        padding-bottom: 1rem;
        position: relative;
    }
    .timeline-item::before {
        content: '';
        position: absolute;
        left: -8px;
        top: 0;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #800000;
        border: 3px solid white;
    }
    </style>
    """, unsafe_allow_html=True)


# --- 2. DATA GENERATION FUNCTIONS (DEFINED FIRST) ---
def generate_historical_events():
	"""Generate historical flood events database"""
	events = [
		{
			'date': datetime(2021, 12, 18),
			'name': 'KL Flash Floods 2021',
			'depth': 120,
			'rainfall': 280,
			'affected_areas': ['Taman Tun Dr Ismail', 'Kelana Jaya', 'Masjid Jamek'],
			'casualties': 48,
			'damage_millions': 350,
			'duration_hours': 18
		},
		{
			'date': datetime(2020, 8, 15),
			'name': 'August Monsoon 2020',
			'depth': 85,
			'rainfall': 180,
			'affected_areas': ['Cheras', 'Ampang', 'Setapak'],
			'casualties': 12,
			'damage_millions': 150,
			'duration_hours': 12
		},
		{
			'date': datetime(2019, 3, 22),
			'name': 'March Storm 2019',
			'depth': 65,
			'rainfall': 140,
			'affected_areas': ['Jalan Tun Razak', 'KLCC'],
			'casualties': 5,
			'damage_millions': 80,
			'duration_hours': 8
		},
		{
			'date': datetime(2018, 11, 10),
			'name': 'November Deluge 2018',
			'depth': 95,
			'rainfall': 220,
			'affected_areas': ['Sentul', 'Titiwangsa'],
			'casualties': 18,
			'damage_millions': 200,
			'duration_hours': 15
		}
	]
	return events


def generate_weather_forecast():
	"""Generate 7-day weather forecast simulation"""
	base_date = datetime.now()
	forecast = []

	for i in range(7):
		date = base_date + timedelta(days=i)
		forecast.append({
			'date': date,
			'day': date.strftime('%A'),
			'temp_high': np.random.randint(31, 36),
			'temp_low': np.random.randint(23, 28),
			'rainfall_prob': np.random.randint(20, 90),
			'rainfall_mm': np.random.randint(0, 150) if np.random.random() > 0.5 else 0,
			'humidity': np.random.randint(65, 95),
			'wind_speed': np.random.randint(5, 25),
			'condition': np.random.choice(['Sunny', 'Cloudy', 'Rain', 'Thunderstorm', 'Heavy Rain'])
		})

	return forecast


def generate_leaderboard():
	"""Generate community leaderboard"""
	base_leaderboard = [
		{'rank': 1, 'user': 'Dr. Ahmad Hassan', 'points': 9500, 'scenarios': 45, 'best_score': 12},
		{'rank': 2, 'user': 'Prof. Sarah Lee', 'points': 8200, 'scenarios': 38, 'best_score': 18},
		{'rank': 3, 'user': 'Eng. Rajesh Kumar', 'points': 7800, 'scenarios': 42, 'best_score': 15},
		{'rank': 4, 'user': 'Dr. Fatimah Zahra', 'points': 7200, 'scenarios': 35, 'best_score': 20}
	]

	# Add current user
	user_points = st.session_state.get('user_points', 0)
	user_scenarios = len(st.session_state.get('scenarios', []))
	base_leaderboard.append({
		'rank': 5,
		'user': 'You',
		'points': user_points,
		'scenarios': user_scenarios,
		'best_score': 25
	})

	return base_leaderboard


# --- 3. SESSION STATE INITIALIZATION ---
def initialize_session_state():
	"""Initialize all session state variables"""
	defaults = {
		'scenarios': [],
		'ai_chat_history': [],
		'current_depth': 0,
		'historical_events': generate_historical_events(),
		'leaderboard': generate_leaderboard(),
		'user_points': 0,
		'badges': [],
		'dark_mode': False,
		'selected_language': 'en',
		'weather_data': generate_weather_forecast(),
		'alerts': [],
		'current_page': 'dashboard'
	}

	for key, value in defaults.items():
		if key not in st.session_state:
			st.session_state[key] = value


# Initialize on load
initialize_session_state()


# --- 4. MODEL LOADING ---
@st.cache_resource
def load_model():
	"""Load ML model with proper error handling"""
	try:
		model = joblib.load('climate_model.pkl')
		return model, None
	except FileNotFoundError:
		return None, "Model file 'climate_model.pkl' not found. Using advanced fallback calculations."
	except Exception as e:
		return None, f"Error loading model: {str(e)}"


model, model_error = load_model()


# --- 5. CALCULATION FUNCTIONS ---
def get_prediction_with_uncertainty(inputs):
	"""Get flood depth prediction with confidence intervals"""
	if model:
		try:
			input_df = pd.DataFrame([inputs])
			depth = model.predict(input_df)[0]
			uncertainty = depth * 0.15
			return {
				'mean': max(0, round(depth, 2)),
				'lower': max(0, round(depth - uncertainty, 2)),
				'upper': max(0, round(depth + uncertainty, 2)),
				'confidence': 85
			}
		except Exception as e:
			st.warning(f"Model prediction failed: {e}. Using fallback.")

	# Advanced fallback formula
	base = inputs['rainfall'] * 0.35
	base += inputs['tide'] * 12
	base += inputs['river_flow'] * 0.06
	base += inputs['soil_saturation'] * 0.25
	base += inputs['impervious_surface'] * 0.18

	# Mitigation with diminishing returns
	mangrove_effect = inputs['mangroves'] * 2.5 * (0.95 ** inputs['mangroves'])
	pond_effect = inputs['retention_ponds'] * 1.8 * (0.92 ** inputs['retention_ponds'])
	pavement_effect = inputs['permeable_pavement'] * 0.35

	base -= (mangrove_effect + pond_effect + pavement_effect)

	uncertainty = base * 0.2

	return {
		'mean': max(0, round(base, 2)),
		'lower': max(0, round(base - uncertainty, 2)),
		'upper': max(0, round(base + uncertainty, 2)),
		'confidence': 70
	}


def calculate_comprehensive_impact(depth, inputs, ind_mix):
	# Convert inputs to standard python types immediately to prevent issues
	depth = float(depth)

	# Financial
	base_cost_per_cm = 50000
	industrial_multiplier = 1 + (ind_mix / 100) * 5
	direct_damage = depth * base_cost_per_cm * industrial_multiplier

	business_loss = direct_damage * 0.35 if depth > 30 else 0
	cleanup = depth * 12000
	infrastructure = depth * 8000 if depth > 40 else 0

	total_financial = direct_damage + business_loss + cleanup + infrastructure

	# Social Impact
	people_affected = int(depth * 150) if depth > 20 else 0
	displacement = int(depth * 50) if depth > 50 else 0
	health_risk = 'CRITICAL' if depth > 70 else 'HIGH' if depth > 40 else 'MODERATE' if depth > 20 else 'LOW'

	# Environmental Impact
	water_quality = 100 - (depth * 0.8) - (inputs['impervious_surface'] * 0.3)
	biodiversity_loss = depth * 0.5 if depth > 30 else 0
	soil_erosion = depth * 0.3

	# Infrastructure Impact
	roads_affected = int((depth / 10) * 5) if depth > 15 else 0
	buildings_at_risk = int((depth / 20) * 10) if depth > 30 else 0

	# --- THE FIX IS HERE ---
	utilities_disrupted = bool(depth > 50)  # Wrap in bool()

	return {
		'financial': {
			'total': round(total_financial / 1000000, 2),
			'direct': round(direct_damage / 1000000, 2),
			'business': round(business_loss / 1000000, 2),
			'cleanup': round(cleanup / 1000000, 2),
			'infrastructure': round(infrastructure / 1000000, 2)
		},
		'social': {
			'affected': people_affected,
			'displaced': displacement,
			'health_risk': health_risk
		},
		'environmental': {
			'water_quality': max(0, round(water_quality, 1)),
			'biodiversity_loss': round(biodiversity_loss, 1),
			'soil_erosion': round(soil_erosion, 1)
		},
		'infrastructure': {
			'roads': roads_affected,
			'buildings': buildings_at_risk,
			'utilities': utilities_disrupted
		}
	}


def calculate_mitigation_roi(inputs, current_impact, baseline_impact):
	"""Calculate ROI for mitigation strategies"""
	mitigation_cost = (
			inputs['mangroves'] * 200000 +
			inputs['retention_ponds'] * 500000 +
			inputs['permeable_pavement'] * 10000
	)

	damage_prevented = (baseline_impact['financial']['total'] - current_impact['financial']['total']) * 1000000

	roi_percentage = ((damage_prevented - mitigation_cost) / max(mitigation_cost,
																 1)) * 100 if mitigation_cost > 0 else 0
	payback_years = (mitigation_cost / max(damage_prevented / 10, 1)) if damage_prevented > 0 else 999

	return {
		'investment': round(mitigation_cost / 1000000, 2),
		'damage_prevented': round(damage_prevented / 1000000, 2),
		'roi_percentage': round(roi_percentage, 1),
		'payback_years': round(payback_years, 1),
		'net_benefit': round((damage_prevented - mitigation_cost) / 1000000, 2)
	}


# --- 6. VISUALIZATION FUNCTIONS ---
def create_3d_flood_surface(inputs):
	"""Create 3D surface plot of flood depth across campus"""
	x = np.linspace(0, 100, 30)
	y = np.linspace(0, 100, 30)
	X, Y = np.meshgrid(x, y)

	center_x, center_y = 50, 50
	distance = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

	base_depth = inputs['rainfall'] * 0.3
	Z = base_depth * np.exp(-distance / 30)

	if inputs['mangroves'] > 0:
		mangrove_zones = [(20, 30), (70, 60), (40, 80)]
		for mx, my in mangrove_zones:
			mangrove_effect = np.exp(-((X - mx) ** 2 + (Y - my) ** 2) / 100)
			Z -= inputs['mangroves'] * 0.1 * mangrove_effect

	Z = np.maximum(Z, 0)

	fig = go.Figure(data=[go.Surface(
		x=X, y=Y, z=Z,
		colorscale='Blues',
		colorbar=dict(title='Depth (cm)')
	)])

	fig.update_layout(
		title='3D Flood Depth Visualization Across Campus',
		scene=dict(
			xaxis_title='West ← → East (m)',
			yaxis_title='South ← → North (m)',
			zaxis_title='Flood Depth (cm)',
			camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
		),
		height=500,
		template='plotly_white'
	)

	return fig


def create_gauge_chart(value, title, max_value=100):
	"""Create gauge chart for metrics"""
	fig = go.Figure(go.Indicator(
		mode="gauge+number+delta",
		value=value,
		title={'text': title},
		delta={'reference': max_value * 0.5},
		gauge={
			'axis': {'range': [None, max_value]},
			'bar': {'color': "#800000"},
			'steps': [
				{'range': [0, max_value * 0.33], 'color': "#d4edda"},
				{'range': [max_value * 0.33, max_value * 0.66], 'color': "#fff3cd"},
				{'range': [max_value * 0.66, max_value], 'color': "#f8d7da"}
			],
			'threshold': {
				'line': {'color': "red", 'width': 4},
				'thickness': 0.75,
				'value': max_value * 0.8
			}
		}
	))

	fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
	return fig


def create_financial_breakdown_chart(financials):
	"""Create detailed financial breakdown"""
	fig = go.Figure(data=[
		go.Bar(
			x=['Direct Damage', 'Business Loss', 'Cleanup Cost', 'Infrastructure'],
			y=[financials['direct'], financials['business'], financials['cleanup'], financials['infrastructure']],
			marker_color=['#8B0000', '#DC143C', '#FFA500', '#FF6347'],
			text=[f"${x}M" for x in
				  [financials['direct'], financials['business'], financials['cleanup'], financials['infrastructure']]],
			textposition='auto',
		)
	])

	fig.update_layout(
		title="Cost Breakdown Analysis",
		yaxis_title="Cost (Million $)",
		height=350,
		template="plotly_white"
	)

	return fig


def create_timeline_projection(inputs, current_depth):
	"""Enhanced timeline with multiple scenarios"""
	years = list(range(2024, 2051, 2))

	no_action = [current_depth * (1 + i * 0.06) for i in range(len(years))]

	current_mitigation = [
		current_depth * (0.88 ** i) if inputs['mangroves'] > 0
		else current_depth * (1 + i * 0.04)
		for i in range(len(years))
	]

	full_mitigation = [current_depth * (0.75 ** i) for i in range(len(years))]

	fig = go.Figure()

	fig.add_trace(go.Scatter(
		x=years, y=no_action,
		name='No Action',
		line=dict(color='#DC143C', width=3, dash='dash'),
		mode='lines+markers'
	))

	fig.add_trace(go.Scatter(
		x=years, y=current_mitigation,
		name='Current Strategy',
		line=dict(color='#FFA500', width=3),
		mode='lines+markers'
	))

	fig.add_trace(go.Scatter(
		x=years, y=full_mitigation,
		name='Full Mitigation',
		line=dict(color='#32CD32', width=3),
		mode='lines+markers',
		fill='tonexty',
		fillcolor='rgba(50, 205, 50, 0.1)'
	))

	fig.update_layout(
		title="Flood Risk Projection (2024-2050)",
		xaxis_title="Year",
		yaxis_title="Flood Depth (cm)",
		height=400,
		template="plotly_white",
		hovermode='x unified'
	)

	return fig


# --- 7. MAIN UI ---
st.title("🎓 TerraTwin Pro: UTM KL Climate Resilience Center")
st.markdown("**Advanced Climate Simulation Platform with AI & Real-Time Analytics**")

# Show model status
if model_error:
	st.warning(f"⚠️ {model_error}")

# Top Navigation Bar
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
	if st.button("🏠 Dashboard", use_container_width=True):
		st.session_state.current_page = 'dashboard'
with col2:
	if st.button("📊 Analytics", use_container_width=True):
		st.session_state.current_page = 'analytics'
with col3:
	if st.button("🌤️ Weather", use_container_width=True):
		st.session_state.current_page = 'weather'
with col4:
	if st.button("📜 History", use_container_width=True):
		st.session_state.current_page = 'history'
with col5:
	if st.button("🏆 Leaderboard", use_container_width=True):
		st.session_state.current_page = 'leaderboard'

st.divider()

# SIDEBAR
with st.sidebar:
	st.header("🎛️ Simulation Controls")

	# Quick Actions
	st.markdown("### ⚡ Quick Actions")
	quick_col1, quick_col2 = st.columns(2)
	with quick_col1:
		if st.button("🔄 Reset", use_container_width=True):
			for key in list(st.session_state.keys()):
				if key not in ['historical_events', 'weather_data']:
					del st.session_state[key]
			st.rerun()
	with quick_col2:
		if st.button("💾 Auto-Save", use_container_width=True):
			st.success("Auto-save enabled!")

	st.markdown("---")

	# Main Form
	with st.form("main_form"):
		scenario_name = st.text_input("📝 Scenario Name", value=f"Scenario {len(st.session_state.scenarios) + 1}")

		# Preset Scenarios
		st.markdown("### 🎯 Preset Scenarios")
		preset = st.selectbox("Load Preset", [
			"Custom",
			"Extreme Monsoon 2024",
			"Business as Usual",
			"Full Green Infrastructure",
			"Climate Change 2050"
		])

		# Apply presets
		if preset == "Extreme Monsoon 2024":
			rain, tide, river, soil = 280, 4.5, 450, 95
			concrete, ind_mix = 90, 85
			mangrove, ponds, pavement = 0, 0, 0
		elif preset == "Full Green Infrastructure":
			rain, tide, river, soil = 150, 1.5, 100, 80
			concrete, ind_mix = 60, 70
			mangrove, ponds, pavement = 40, 30, 80
		elif preset == "Climate Change 2050":
			rain, tide, river, soil = 320, 5.0, 500, 100
			concrete, ind_mix = 95, 90
			mangrove, ponds, pavement = 5, 2, 10
		else:
			rain, tide, river, soil = 150, 1.5, 100, 80
			concrete, ind_mix = 85, 70
			mangrove, ponds, pavement = 0, 0, 0

		with st.expander("🌊 Hydrology", expanded=True):
			rain = st.slider("☔ Rainfall (mm/hr)", 0, 350, rain, help="Historical max: 280mm/hr (2021)")
			tide = st.slider("🌊 River Level (m)", 0.0, 6.0, tide, step=0.1)
			river = st.slider("💧 Upstream Flow (m³/s)", 10, 600, river)
			soil = st.slider("🌱 Soil Saturation (%)", 0, 100, soil)

		with st.expander("🏙️ Urban Infrastructure"):
			concrete = st.slider("🏗️ Impervious Surface (%)", 0, 100, concrete)
			ind_mix = st.slider("💼 Asset Density (%)", 0, 100, ind_mix)

		with st.expander("🌳 Green Solutions"):
			mangrove = st.slider("🌳 Urban Forest (Ha)", 0, 60, mangrove)
			ponds = st.slider("💧 Retention Ponds", 0, 60, ponds)
			pavement = st.slider("🧱 Permeable Pavement (%)", 0, 100, pavement)

		submit_col1, submit_col2 = st.columns(2)
		with submit_col1:
			submitted = st.form_submit_button("🚀 Run Simulation", use_container_width=True)
		with submit_col2:
			save_scenario = st.form_submit_button("💾 Save", use_container_width=True)

	# User Stats
	st.markdown("---")
	st.markdown("### 👤 Your Statistics")
	st.metric("Total Points", st.session_state.user_points)
	st.metric("Scenarios Created", len(st.session_state.scenarios))
	st.metric("Badges Earned", len(st.session_state.badges))

# PROCESS SIMULATION
inputs = {
	'rainfall': rain, 'tide': tide, 'river_flow': river, 'soil_saturation': soil,
	'elevation': 5, 'impervious_surface': concrete, 'distance_river': 200,
	'drainage_efficiency': max(0, 70 - concrete * 0.5),
	'road_density': 80, 'mangroves': mangrove,
	'retention_ponds': ponds, 'permeable_pavement': pavement
}

if submitted or 'prediction' not in st.session_state:
	with st.spinner("🔄 Running advanced simulation..."):
		prediction = get_prediction_with_uncertainty(inputs)
		impacts = calculate_comprehensive_impact(prediction['mean'], inputs, ind_mix)

		# Calculate baseline for ROI
		baseline_inputs = inputs.copy()
		baseline_inputs['mangroves'] = 0
		baseline_inputs['retention_ponds'] = 0
		baseline_inputs['permeable_pavement'] = 0
		baseline_prediction = get_prediction_with_uncertainty(baseline_inputs)
		baseline_impacts = calculate_comprehensive_impact(baseline_prediction['mean'], baseline_inputs, ind_mix)

		roi = calculate_mitigation_roi(inputs, impacts, baseline_impacts)

		# Calculate points
		points_earned = max(0, 100 - int(prediction['mean']))
		st.session_state.user_points += points_earned

		st.session_state.prediction = prediction
		st.session_state.impacts = impacts
		st.session_state.roi = roi

		if submitted:
			st.success(f"✅ Simulation complete! Earned {points_earned} points.")
else:
	prediction = st.session_state.prediction
	impacts = st.session_state.impacts
	roi = st.session_state.roi

if save_scenario:
	st.session_state.scenarios.append({
		'name': scenario_name,
		'timestamp': datetime.now(),
		'inputs': inputs.copy(),
		'depth': prediction['mean'],
		'financials': impacts['financial'],
		'impacts': impacts
	})
	st.sidebar.success(f"✅ Saved '{scenario_name}'")

# KEY METRICS ROW
st.markdown("### 📊 Real-Time Performance Dashboard")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
	st.metric(
		"🌊 Flood Depth",
		f"{prediction['mean']} cm",
		delta=f"±{prediction['upper'] - prediction['mean']:.1f}",
		help=f"Confidence: {prediction['confidence']}%"
	)

with col2:
	st.metric(
		"💰 Total Impact",
		f"${impacts['financial']['total']}M",
		delta=f"-${impacts['financial']['business']}M",
		delta_color="inverse"
	)

with col3:
	st.metric(
		"👥 People Affected",
		f"{impacts['social']['affected']:,}",
		delta=impacts['social']['health_risk'],
		delta_color="inverse" if impacts['social']['health_risk'] != 'LOW' else "normal"
	)

with col4:
	st.metric(
		"🌍 Water Quality",
		f"{impacts['environmental']['water_quality']:.1f}%",
		delta=f"{100 - impacts['environmental']['water_quality']:.1f}% degraded",
		delta_color="inverse"
	)

with col5:
	st.metric(
		"📈 Mitigation ROI",
		f"{roi['roi_percentage']}%",
		delta=f"{roi['payback_years']:.1f} yrs payback"
	)

st.divider()

# TABS
tabs = st.tabs([
	"🗺️ Digital Twin",
	"📊 3D Analytics",
	"🤖 AI Insights",
	"📈 Comparison",
	"🌤️ Weather",
	"📜 History",
	"📋 Export"
])

# TAB 1: Digital Twin
with tabs[0]:
	st.subheader("🗺️ Interactive Campus Digital Twin")
	st.info("💡 Map shows flood zones, infrastructure status, and green assets")

	# Simple map placeholder
	utm_lat, utm_lon = 3.1729, 101.7209
	m = folium.Map(location=[utm_lat, utm_lon], zoom_start=14)

	# Add flood zone
	risk_color = "red" if prediction['mean'] > 50 else "orange" if prediction['mean'] > 30 else "green"
	folium.Circle(
		[utm_lat, utm_lon],
		radius=400 + (prediction['mean'] * 10),
		color=risk_color,
		fill=True,
		fill_opacity=0.4,
		popup=f"Flood Depth: {prediction['mean']}cm"
	).add_to(m)

	st_folium(m, width=1200, height=500)

# TAB 2: 3D Analytics
with tabs[1]:
	st.subheader("📊 Advanced 3D Visualizations & Analytics")

	# 3D Flood Surface
	st.plotly_chart(create_3d_flood_surface(inputs), use_container_width=True)

	# Gauge Charts
	gauge_col1, gauge_col2, gauge_col3, gauge_col4 = st.columns(4)
	with gauge_col1:
		st.plotly_chart(create_gauge_chart(impacts['environmental']['water_quality'], "Water Quality", 100),
						use_container_width=True)
	with gauge_col2:
		st.plotly_chart(create_gauge_chart(100 - min(100, prediction['mean']), "Safety Score", 100),
						use_container_width=True)
	with gauge_col3:
		st.plotly_chart(create_gauge_chart(min(100, max(0, roi['roi_percentage'])), "ROI %", 100),
						use_container_width=True)
	with gauge_col4:
		st.plotly_chart(create_gauge_chart(min(100, mangrove * 2 + ponds), "Green Index", 100),
						use_container_width=True)

	# Financial breakdown and timeline
	col1, col2 = st.columns(2)
	with col1:
		st.plotly_chart(create_financial_breakdown_chart(impacts['financial']), use_container_width=True)
	with col2:
		st.plotly_chart(create_timeline_projection(inputs, prediction['mean']), use_container_width=True)

# TAB 3: AI Insights
with tabs[2]:
	st.subheader("🤖 AI-Powered Climate Advisor")

	insight_col1, insight_col2 = st.columns(2)

	with insight_col1:
		st.markdown(f"""
        <div class="alert-box alert-info">
            <h4>📊 Scenario Overview</h4>
            <p><strong>Flood Prediction:</strong> {prediction['mean']}cm (±{prediction['upper'] - prediction['mean']:.1f}cm)</p>
            <p><strong>Confidence Level:</strong> {prediction['confidence']}%</p>
            <p><strong>People at Risk:</strong> {impacts['social']['affected']:,}</p>
            <p><strong>Health Risk:</strong> {impacts['social']['health_risk']}</p>
        </div>
        """, unsafe_allow_html=True)

		if prediction['mean'] > 50:
			st.markdown("""
            <div class="alert-box alert-danger">
                <h4>🚨 CRITICAL FLOOD RISK</h4>
                <p>This scenario represents severe flooding that would cause major disruption.</p>
            </div>
            """, unsafe_allow_html=True)

	with insight_col2:
		st.markdown(f"""
        <div class="alert-box alert-warning">
            <h4>💰 Financial Impact Analysis</h4>
            <p><strong>Total Economic Loss:</strong> ${impacts['financial']['total']}M</p>
            <p><strong>Breakdown:</strong></p>
            <ul>
                <li>Direct Damage: ${impacts['financial']['direct']}M</li>
                <li>Business Loss: ${impacts['financial']['business']}M</li>
                <li>Cleanup: ${impacts['financial']['cleanup']}M</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

		if roi['roi_percentage'] > 100:
			st.markdown(f"""
            <div class="alert-box alert-success">
                <h4>✅ Excellent Mitigation ROI</h4>
                <p>Investment of ${roi['investment']}M prevents ${roi['damage_prevented']}M in damages!</p>
                <p>ROI: {roi['roi_percentage']:.0f}% | Payback: {roi['payback_years']:.1f} years</p>
            </div>
            """, unsafe_allow_html=True)

# TAB 4: Comparison
with tabs[3]:
	st.subheader("📈 Scenario Comparison")

	if len(st.session_state.scenarios) > 0:
		st.markdown(f"**Saved Scenarios: {len(st.session_state.scenarios)}**")

		comparison_data = []
		for scenario in st.session_state.scenarios:
			comparison_data.append({
				'Scenario': scenario['name'],
				'Time': scenario['timestamp'].strftime("%Y-%m-%d %H:%M"),
				'Flood Depth (cm)': scenario['depth'],
				'Total Cost ($M)': scenario['financials']['total'],
				'Urban Forest (Ha)': scenario['inputs']['mangroves'],
				'Ponds': scenario['inputs']['retention_ponds']
			})

		comparison_df = pd.DataFrame(comparison_data)
		st.dataframe(comparison_df, use_container_width=True, hide_index=True)

		if st.button("🗑️ Clear All Scenarios"):
			st.session_state.scenarios = []
			st.rerun()
	else:
		st.info("💡 Save scenarios using the sidebar to compare them here.")

# TAB 5: Weather Forecast
with tabs[4]:
	st.subheader("🌤️ 7-Day Weather Forecast")

	forecast_data = []
	for day in st.session_state.weather_data:
		forecast_data.append({
			'Day': day['day'][:3],
			'Date': day['date'].strftime('%d/%m'),
			'High': day['temp_high'],
			'Low': day['temp_low'],
			'Rain %': day['rainfall_prob'],
			'Condition': day['condition']
		})

	forecast_df = pd.DataFrame(forecast_data)
	st.dataframe(forecast_df, use_container_width=True)

	# Rainfall projection
	fig = go.Figure()
	fig.add_trace(go.Bar(
		x=[d['day'] for d in st.session_state.weather_data],
		y=[d['rainfall_mm'] for d in st.session_state.weather_data],
		marker_color='#1f77b4'
	))
	fig.update_layout(title="7-Day Rainfall Projection", yaxis_title="Rainfall (mm)", height=300)
	st.plotly_chart(fig, use_container_width=True)

# TAB 6: Historical Events
with tabs[5]:
	st.subheader("📜 Historical Flood Events Database")

	for event in st.session_state.historical_events:
		with st.expander(f"📅 {event['name']} - {event['date'].strftime('%B %d, %Y')}"):
			col1, col2, col3 = st.columns(3)
			with col1:
				st.metric("Flood Depth", f"{event['depth']} cm")
				st.metric("Rainfall", f"{event['rainfall']} mm/hr")
			with col2:
				st.metric("Casualties", event['casualties'])
				st.metric("Duration", f"{event['duration_hours']} hours")
			with col3:
				st.metric("Economic Loss", f"${event['damage_millions']}M")

			st.markdown(f"**Affected Areas:** {', '.join(event['affected_areas'])}")

			if prediction['mean'] > event['depth']:
				st.error(f"⚠️ Current scenario ({prediction['mean']}cm) exceeds this event!")
			else:
				st.success(f"✅ Current scenario is less severe.")

# TAB 7: Export
with tabs[6]:
	st.subheader("📋 Export & Download Reports")

	col1, col2 = st.columns(2)

	with col1:
		# JSON Export
		export_data = {
			'timestamp': datetime.now().isoformat(),
			'scenario': scenario_name,
			'inputs': inputs,
			'prediction': prediction,
			'impacts': impacts,
			'roi': roi
		}

		st.download_button(
			label="📊 Download JSON Data",
			data=json.dumps(export_data, indent=2),
			file_name=f"terratwin_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
			mime="application/json",
			use_container_width=True
		)

	with col2:
		# CSV Export
		csv_data = pd.DataFrame([{
			'Scenario': scenario_name,
			'Flood_Depth_cm': prediction['mean'],
			'Total_Cost_M': impacts['financial']['total'],
			'People_Affected': impacts['social']['affected'],
			'Water_Quality': impacts['environmental']['water_quality'],
			'ROI_Percent': roi['roi_percentage'],
			'Timestamp': datetime.now()
		}])

		st.download_button(
			label="📈 Download CSV",
			data=csv_data.to_csv(index=False),
			file_name=f"terratwin_metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
			mime="text/csv",
			use_container_width=True
		)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p style='font-size: 1.2em;'><b>🎓 TerraTwin Pro</b> - Advanced Climate Resilience Platform</p>
    <p style='color: #666;'>UTM Kuala Lumpur | Powered by AI & Geospatial Analytics</p>
    <div class="stat-badge">Version 3.0</div>
    <div class="stat-badge">🔒 Secure</div>
    <div class="stat-badge">⚡ Real-Time</div>
</div>
""", unsafe_allow_html=True)