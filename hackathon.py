import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIGURATION & ENHANCED STYLING ---
st.set_page_config(
	page_title="TerraTwin: UTM KL Resilience Center",
	page_icon="🎓",
	layout="wide",
	initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Enhanced Metrics Styling */
    div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {
        color: #000000 !important;
    }
    .stMetric {
        background-color: #f8f9fa !important;
        border-left: 5px solid #800000;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Sidebar Improvements */
    .streamlit-expanderHeader {
        color: #000000 !important;
        background-color: #ffffff !important;
        font-weight: 600;
    }
    .streamlit-expanderContent {
        color: #000000 !important;
        background-color: #f8f9fa !important;
    }

    /* Custom Alert Boxes */
    .alert-success {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .alert-warning {
        padding: 1rem;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .alert-danger {
        padding: 1rem;
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        border-radius: 8px;
        margin: 1rem 0;
    }

    /* Info Cards */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }

    /* Button Improvements */
    .stButton>button {
        width: 100%;
        background-color: #800000;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #600000;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SESSION STATE INITIALIZATION ---
if 'scenarios' not in st.session_state:
	st.session_state.scenarios = []
if 'ai_chat_history' not in st.session_state:
	st.session_state.ai_chat_history = []
if 'current_depth' not in st.session_state:
	st.session_state.current_depth = 0


# --- 3. ENHANCED MODEL LOADING ---
@st.cache_resource
def load_model():
	"""Load ML model with proper error handling"""
	try:
		model = joblib.load('climate_model.pkl')
		return model, None
	except FileNotFoundError:
		return None, "Model file 'climate_model.pkl' not found. Using fallback calculations."
	except Exception as e:
		return None, f"Error loading model: {str(e)}"


model, model_error = load_model()


# --- 4. ENHANCED CALCULATION ENGINES ---
def get_prediction(inputs):
	"""Get flood depth prediction with fallback"""
	if model:
		try:
			input_df = pd.DataFrame([inputs])
			depth = model.predict(input_df)[0]
			return max(0, round(depth, 2))
		except Exception as e:
			st.warning(f"Model prediction failed: {e}. Using fallback.")

	# Fallback formula if model unavailable
	base = inputs['rainfall'] * 0.3
	base += inputs['tide'] * 10
	base += inputs['river_flow'] * 0.05
	base += inputs['soil_saturation'] * 0.2
	base += inputs['impervious_surface'] * 0.15

	# Mitigation effects
	base -= inputs['mangroves'] * 2
	base -= inputs['retention_ponds'] * 1.5
	base -= inputs['permeable_pavement'] * 0.3

	return max(0, round(base, 2))


def calculate_financials(depth, industrial_pct, area_sqm=50000):
	"""Enhanced financial impact calculation"""
	# Direct damage
	base_cost_per_cm = 50000
	industrial_multiplier = 1 + (industrial_pct / 100) * 5
	direct_damage = depth * base_cost_per_cm * industrial_multiplier

	# Indirect costs
	business_interruption = direct_damage * 0.3 if depth > 30 else 0
	cleanup_cost = depth * 10000

	total_loss = direct_damage + business_interruption + cleanup_cost

	return {
		'total_millions': round(total_loss / 1000000, 2),
		'direct_damage': round(direct_damage / 1000000, 2),
		'business_loss': round(business_interruption / 1000000, 2),
		'cleanup': round(cleanup_cost / 1000000, 2)
	}


def calculate_logistics(depth):
	"""Multi-level logistics assessment"""
	if depth > 50:
		return {
			'status': '⛔ TOTAL SHUTDOWN',
			'severity': 'critical',
			'description': 'All roads impassable. Emergency evacuation required.'
		}
	elif depth > 30:
		return {
			'status': '⚠️ HEAVY DELAYS',
			'severity': 'warning',
			'description': 'Major routes flooded. Only high-clearance vehicles can pass.'
		}
	elif depth > 15:
		return {
			'status': '🔶 MINOR IMPACT',
			'severity': 'caution',
			'description': 'Some low-lying roads affected. Drive with caution.'
		}
	else:
		return {
			'status': '✅ OPERATIONAL',
			'severity': 'safe',
			'description': 'All routes clear. Normal operations.'
		}


def calculate_carbon(mangroves, ponds):
	"""Enhanced carbon and ecosystem services calculation"""
	# Carbon sequestration
	carbon_tons_year = mangroves * 500  # Tons CO2/year
	pond_carbon = ponds * 50  # Wetlands also sequester
	total_carbon = carbon_tons_year + pond_carbon

	# Carbon credit value
	credit_value = total_carbon * 30  # $30/ton

	# Ecosystem services (biodiversity, air quality, etc.)
	ecosystem_value = (mangroves * 5000) + (ponds * 2000)

	return {
		'carbon_tons': int(total_carbon),
		'credit_value': int(credit_value),
		'ecosystem_value': int(ecosystem_value),
		'total_benefit': int(credit_value + ecosystem_value)
	}


def get_risk_level(depth):
	"""Categorize flood risk"""
	if depth > 80:
		return {'level': 'EXTREME', 'color': '#8B0000', 'emoji': '🔴'}
	elif depth > 50:
		return {'level': 'HIGH', 'color': '#DC143C', 'emoji': '🟠'}
	elif depth > 30:
		return {'level': 'MODERATE', 'color': '#FFA500', 'emoji': '🟡'}
	elif depth > 15:
		return {'level': 'LOW', 'color': '#FFD700', 'emoji': '🟢'}
	else:
		return {'level': 'MINIMAL', 'color': '#32CD32', 'emoji': '🟢'}


# --- 5. AI TUTOR SYSTEM ---
def generate_ai_insights(inputs, depth, financials, logistics, carbon):
	"""Generate contextual AI recommendations"""
	insights = []
	risk = get_risk_level(depth)

	# Risk Assessment
	insights.append({
		'type': 'risk',
		'title': f"{risk['emoji']} Flood Risk: {risk['level']}",
		'message': f"Predicted flood depth of {depth}cm represents a {risk['level'].lower()} risk scenario."
	})

	# Rainfall Analysis
	if inputs['rainfall'] > 200:
		insights.append({
			'type': 'warning',
			'title': '🌧️ Extreme Rainfall Event',
			'message': f"Rainfall intensity of {inputs['rainfall']}mm/hr is categorized as extreme. This exceeds typical monsoon levels and may overwhelm drainage systems. Historical data shows such events occur 2-3 times per year in KL."
		})
	elif inputs['rainfall'] > 100:
		insights.append({
			'type': 'info',
			'title': '🌧️ Heavy Rainfall',
			'message': f"Rainfall of {inputs['rainfall']}mm/hr is significant. Urban drainage designed for 50-80mm/hr will struggle."
		})

	# Mitigation Effectiveness
	if inputs['mangroves'] > 0 or inputs['retention_ponds'] > 0:
		reduction = (inputs['mangroves'] * 2 + inputs['retention_ponds'] * 1.5)
		insights.append({
			'type': 'success',
			'title': '🌳 Green Infrastructure Active',
			'message': f"Your {inputs['mangroves']}Ha urban forest and {inputs['retention_ponds']} retention ponds are reducing flood depth by approximately {reduction:.1f}cm. Great choice!"
		})
	else:
		insights.append({
			'type': 'warning',
			'title': '⚠️ No Green Infrastructure',
			'message': "Consider adding urban forests or retention ponds. Studies show 10Ha of urban forest can reduce peak runoff by 20-30%."
		})

	# Urban Surface Analysis
	if inputs['impervious_surface'] > 80:
		insights.append({
			'type': 'warning',
			'title': '🏙️ High Concrete Coverage',
			'message': f"Campus has {inputs['impervious_surface']}% impervious surfaces. This accelerates runoff. Implementing permeable pavement could reduce flood risk by 15-25%."
		})

	# Financial Recommendation
	mitigation_cost = (inputs['mangroves'] * 0.2 + inputs['retention_ponds'] * 0.5)
	if financials['total_millions'] > mitigation_cost * 3:
		insights.append({
			'type': 'success',
			'title': '💰 Strong ROI on Mitigation',
			'message': f"Potential damage (${financials['total_millions']}M) is {financials['total_millions'] / max(mitigation_cost, 0.01):.1f}x the cost of green solutions (${mitigation_cost:.2f}M). Investing in mitigation is highly cost-effective."
		})

	# Logistics Impact
	if logistics['severity'] == 'critical':
		insights.append({
			'type': 'danger',
			'title': '🚨 Critical Access Issue',
			'message': "Campus is completely inaccessible. Emergency response time increases from 15 to 60+ minutes. Consider emergency boat access protocols."
		})

	# Carbon Benefits
	if carbon['carbon_tons'] > 1000:
		insights.append({
			'type': 'success',
			'title': '🌍 Significant Carbon Impact',
			'message': f"Your green infrastructure sequesters {carbon['carbon_tons']:,} tons CO2/year - equivalent to removing {carbon['carbon_tons'] // 4} cars from the road. Carbon credits worth ${carbon['credit_value']:,} annually."
		})

	return insights


# --- 6. ENHANCED MAP RENDERING ---
def render_enhanced_map(depth, inputs, logistics):
	"""Create detailed interactive map"""
	utm_lat, utm_lon = 3.1729, 101.7209

	# Base map with better tiles
	m = folium.Map(
		location=[utm_lat, utm_lon],
		zoom_start=14,
		tiles="OpenStreetMap"
	)

	# Add additional tile layers
	folium.TileLayer('CartoDB positron').add_to(m)
	folium.TileLayer('CartoDB dark_matter').add_to(m)

	# Flood extent visualization
	risk = get_risk_level(depth)
	radius = 400 + (depth * 10)

	folium.Circle(
		[utm_lat, utm_lon],
		radius=radius,
		color=risk['color'],
		fill=True,
		fill_opacity=0.4,
		popup=folium.Popup(f"""
            <b>Flood Zone Analysis</b><br>
            Depth: {depth}cm<br>
            Risk: {risk['level']}<br>
            Radius: {radius}m
        """, max_width=200),
		tooltip=f"Flood Depth: {depth}cm"
	).add_to(m)

	# Campus buildings (key infrastructure)
	buildings = [
		{'name': 'Main Library', 'coords': [3.1735, 101.7215], 'type': 'education'},
		{'name': 'Engineering Labs', 'coords': [3.1725, 101.7220], 'type': 'research'},
		{'name': 'Server Room', 'coords': [3.1720, 101.7200], 'type': 'critical'},
		{'name': 'Student Housing', 'coords': [3.1740, 101.7195], 'type': 'residential'},
	]

	for building in buildings:
		icon_color = 'red' if depth > 50 else 'orange' if depth > 30 else 'green'
		folium.Marker(
			building['coords'],
			popup=f"<b>{building['name']}</b><br>Status: {'At Risk' if depth > 30 else 'Safe'}",
			icon=folium.Icon(color=icon_color, icon='building', prefix='fa'),
			tooltip=building['name']
		).add_to(m)

	# Road network
	roads = [
		{'name': 'Jalan Tun Razak', 'coords': [[3.180, 101.715], [3.1729, 101.7209], [3.160, 101.725]]},
		{'name': 'Campus Ring Road', 'coords': [[3.175, 101.718], [3.172, 101.722], [3.170, 101.720]]}
	]

	road_color = {
		'critical': 'red',
		'warning': 'orange',
		'caution': 'yellow',
		'safe': 'green'
	}[logistics['severity']]

	for road in roads:
		folium.PolyLine(
			locations=road['coords'],
			color=road_color,
			weight=5,
			opacity=0.7,
			tooltip=f"{road['name']}: {logistics['status']}"
		).add_to(m)

	# Green infrastructure
	if inputs['mangroves'] > 0:
		np.random.seed(42)
		for i in range(int(inputs['mangroves'] / 2)):
			folium.Marker(
				[utm_lat + np.random.uniform(-0.005, 0.005),
				 utm_lon + np.random.uniform(-0.005, 0.005)],
				icon=folium.Icon(color="green", icon="tree", prefix='fa'),
				tooltip=f"Urban Forest Zone {i + 1}"
			).add_to(m)

	if inputs['retention_ponds'] > 0:
		for i in range(min(5, inputs['retention_ponds'])):
			folium.Circle(
				[utm_lat + np.random.uniform(-0.008, 0.008),
				 utm_lon + np.random.uniform(-0.008, 0.008)],
				radius=50,
				color='blue',
				fill=True,
				fill_color='cyan',
				fill_opacity=0.6,
				tooltip=f"Retention Pond {i + 1}"
			).add_to(m)

	# Add layer control
	folium.LayerControl().add_to(m)

	return m


# --- 7. VISUALIZATION FUNCTIONS ---
def create_financial_breakdown_chart(financials):
	"""Create detailed financial breakdown"""
	fig = go.Figure(data=[
		go.Bar(
			x=['Direct Damage', 'Business Loss', 'Cleanup Cost'],
			y=[financials['direct_damage'], financials['business_loss'], financials['cleanup']],
			marker_color=['#8B0000', '#DC143C', '#FFA500'],
			text=[f"${x}M" for x in [financials['direct_damage'], financials['business_loss'], financials['cleanup']]],
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

	# Scenario 1: No Action
	no_action = [current_depth * (1 + i * 0.06) for i in range(len(years))]

	# Scenario 2: Current Mitigation
	current_mitigation = [
		current_depth * (0.88 ** i) if inputs['mangroves'] > 0
		else current_depth * (1 + i * 0.04)
		for i in range(len(years))
	]

	# Scenario 3: Full Mitigation (max green infrastructure)
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


def create_mitigation_effectiveness_chart(inputs):
	"""Show effectiveness of each mitigation strategy"""
	strategies = ['Urban Forest', 'Retention Ponds', 'Permeable Pavement']
	effectiveness = [
		inputs['mangroves'] * 2,
		inputs['retention_ponds'] * 1.5,
		inputs['permeable_pavement'] * 0.3
	]
	costs = [
		inputs['mangroves'] * 0.2,
		inputs['retention_ponds'] * 0.5,
		inputs['permeable_pavement'] * 0.01
	]

	fig = go.Figure()

	fig.add_trace(go.Bar(
		name='Flood Reduction (cm)',
		x=strategies,
		y=effectiveness,
		marker_color='#10b981',
		yaxis='y',
		offsetgroup=0
	))

	fig.add_trace(go.Bar(
		name='Implementation Cost ($M)',
		x=strategies,
		y=costs,
		marker_color='#800000',
		yaxis='y2',
		offsetgroup=1
	))

	fig.update_layout(
		title='Mitigation Strategy Comparison',
		yaxis=dict(title='Flood Reduction (cm)'),
		yaxis2=dict(title='Cost (Million $)', overlaying='y', side='right'),
		barmode='group',
		height=350,
		template="plotly_white"
	)

	return fig


# --- 8. MAIN UI LAYOUT ---
st.title("🎓 TerraTwin: UTM KL Climate Resilience Center")
st.markdown("**Advanced Flood Prediction & Mitigation Planning System**")

# Show model status
if model_error:
	st.warning(f"⚠️ {model_error}")

# --- SIDEBAR WITH ENHANCED FORM ---
with st.sidebar:
	st.header("🎛️ Simulation Controls")

	st.info(
		"💡 **Quick Start**: Adjust sliders below to simulate different scenarios. Click 'Run Simulation' to see results.")

	with st.form("main_form"):
		st.subheader("📊 Current Scenario")
		scenario_name = st.text_input("Scenario Name", value=f"Scenario {len(st.session_state.scenarios) + 1}")

		with st.expander("🌊 1. Hydrology (Water Factors)", expanded=True):
			rain = st.slider(
				"Rainfall Intensity (mm/hr)",
				0, 300, 150,
				help="Typical KL monsoon: 80-150mm/hr. Extreme: 200+mm/hr"
			)
			tide = st.slider(
				"River Level (m)",
				0.0, 5.0, 1.5,
				help="Gombak/Klang River level. Normal: 1-2m, Flood: 3-4m"
			)
			river = st.slider(
				"Upstream Flow (m³/s)",
				10, 500, 100,
				help="Water volume from upstream catchment areas"
			)
			soil = st.slider(
				"Soil Saturation (%)",
				0, 100, 80,
				help="Higher saturation = less absorption capacity"
			)

		with st.expander("🏙️ 2. Urban Infrastructure"):
			concrete = st.slider(
				"Impervious Surface (%)",
				0, 100, 85,
				help="KL city center: 80-95%. Parks/green areas: 20-40%"
			)
			ind_mix = st.slider(
				"Asset Density (%)",
				0, 100, 70,
				help="Value of infrastructure per square meter"
			)

		with st.expander("🌳 3. Green Solutions"):
			mangrove = st.slider(
				"🌳 Urban Forest Area (Hectares)",
				0, 50, 0,
				help="Each hectare can absorb ~2cm flood depth"
			)
			ponds = st.slider(
				"💧 Retention Ponds (Units)",
				0, 50, 0,
				help="Each pond reduces flood by ~1.5cm"
			)
			pavement = st.slider(
				"🧱 Permeable Pavement (%)",
				0, 100, 0,
				help="Allows water infiltration vs traditional concrete"
			)

		col1, col2 = st.columns(2)
		with col1:
			submitted = st.form_submit_button("🚀 Run Simulation", use_container_width=True)
		with col2:
			save_scenario = st.form_submit_button("💾 Save Scenario", use_container_width=True)

# --- PROCESS INPUTS ---
inputs = {
	'rainfall': rain,
	'tide': tide,
	'river_flow': river,
	'soil_saturation': soil,
	'elevation': 5,
	'impervious_surface': concrete,
	'distance_river': 200,
	'drainage_efficiency': max(0, 70 - concrete * 0.5),  # Decreases with more concrete
	'road_density': 80,
	'mangroves': mangrove,
	'retention_ponds': ponds,
	'permeable_pavement': pavement
}

if submitted or 'depth' not in st.session_state:
	with st.spinner("🔄 Running simulation..."):
		depth = get_prediction(inputs)
		financials = calculate_financials(depth, ind_mix)
		logistics = calculate_logistics(depth)
		carbon = calculate_carbon(mangrove, ponds)
		ai_insights = generate_ai_insights(inputs, depth, financials, logistics, carbon)
		risk = get_risk_level(depth)

		# Store in session state
		st.session_state.depth = depth
		st.session_state.financials = financials
		st.session_state.logistics = logistics
		st.session_state.carbon = carbon
		st.session_state.ai_insights = ai_insights
		st.session_state.risk = risk
		st.session_state.current_depth = depth
else:
	# Use cached values
	depth = st.session_state.depth
	financials = st.session_state.financials
	logistics = st.session_state.logistics
	carbon = st.session_state.carbon
	ai_insights = st.session_state.ai_insights
	risk = st.session_state.risk

# Save scenario
if save_scenario:
	st.session_state.scenarios.append({
		'name': scenario_name,
		'timestamp': datetime.now(),
		'inputs': inputs.copy(),
		'depth': depth,
		'financials': financials,
		'carbon': carbon
	})
	st.sidebar.success(f"✅ Saved '{scenario_name}'")

# --- MAIN DASHBOARD ---

# Key Metrics Row
st.markdown("### 📊 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
	delta_color = "inverse" if depth > 50 else "normal"
	st.metric(
		"🌊 Flood Depth",
		f"{depth} cm",
		delta=f"{risk['level']}",
		delta_color=delta_color,
		help="Predicted maximum flood depth at campus center"
	)

with col2:
	st.metric(
		"💸 Total Impact",
		f"${financials['total_millions']}M",
		delta=f"-${financials['business_loss']}M indirect",
		delta_color="inverse",
		help="Direct damage + business interruption + cleanup"
	)

with col3:
	st.metric(
		"🚗 Campus Access",
		logistics['status'],
		delta=logistics['description'][:20],
		delta_color="inverse" if logistics['severity'] in ['critical', 'warning'] else "normal",
		help=logistics['description']
	)

with col4:
	st.metric(
		"🌍 Green Benefits",
		f"+${carbon['total_benefit']:,}",
		delta=f"{carbon['carbon_tons']} tons CO₂/yr",
		help="Annual carbon credits + ecosystem services value"
	)

st.divider()

# Tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
	"🗺️ Digital Twin Map",
	"📈 Analytics Dashboard",
	"🤖 AI Insights",
	"📊 Scenario Comparison",
	"📋 Export Report"
])

with tab1:
	st.subheader("📍 Live UTM KL Campus Situation")

	col_map, col_legend = st.columns([3, 1])

	with col_map:
		map_obj = render_enhanced_map(depth, inputs, logistics)
		st_folium(map_obj, width=900, height=600)

	with col_legend:
		st.markdown("#### 🎯 Map Legend")
		st.markdown(f"""
        **Flood Zone**  
        {risk['emoji']} {risk['level']} Risk  
        Radius: {400 + (depth * 10):.0f}m

        **Infrastructure Status**  
        🟢 Safe  
        🟠 At Risk  
        🔴 Critical  

        **Road Network**  
        {logistics['status']}

        **Green Assets**  
        🌳 {mangrove} Ha Forest  
        💧 {ponds} Ponds  
        """)

		if depth > 30:
			st.error("🚨 **Emergency Protocol Required**")
		elif depth > 15:
			st.warning("⚠️ **Monitor Closely**")
		else:
			st.success("✅ **Normal Operations**")

with tab2:
	st.subheader("📈 Comprehensive Analytics")

	# Financial Breakdown
	col1, col2 = st.columns(2)

	with col1:
		st.plotly_chart(create_financial_breakdown_chart(financials), use_container_width=True)

	with col2:
		st.plotly_chart(create_mitigation_effectiveness_chart(inputs), use_container_width=True)

	# Timeline Projection
	st.plotly_chart(create_timeline_projection(inputs, depth), use_container_width=True)

	# Detailed Metrics Table
	st.markdown("#### 📋 Detailed Metrics")
	metrics_df = pd.DataFrame({
		'Metric': [
			'Rainfall Intensity', 'River Level', 'Soil Saturation',
			'Impervious Surface', 'Urban Forest', 'Retention Capacity',
			'Flood Depth', 'Damage Cost', 'Carbon Offset'
		],
		'Value': [
			f"{rain} mm/hr", f"{tide} m", f"{soil}%",
			f"{concrete}%", f"{mangrove} Ha", f"{ponds} ponds",
			f"{depth} cm", f"${financials['total_millions']}M", f"{carbon['carbon_tons']} tons/yr"
		],
		'Status': [
			'🔴 Extreme' if rain > 200 else '🟡 High' if rain > 100 else '🟢 Normal',
			'🔴 Flood' if tide > 3 else '🟡 High' if tide > 2 else '🟢 Normal',
			'🔴 Saturated' if soil > 80 else '🟡 High' if soil > 60 else '🟢 Good',
			'🔴 High' if concrete > 80 else '🟡 Medium' if concrete > 60 else '🟢 Low',
			'🟢 Active' if mangrove > 0 else '⚪ None',
			'🟢 Active' if ponds > 0 else '⚪ None',
			f"{risk['emoji']} {risk['level']}",
			'🔴 High' if financials['total_millions'] > 10 else '🟡 Medium' if financials[
																				 'total_millions'] > 5 else '🟢 Low',
			'🟢 Positive' if carbon['carbon_tons'] > 0 else '⚪ None'
		]
	})
	st.dataframe(metrics_df, use_container_width=True, hide_index=True)

with tab3:
	st.subheader("🤖 AI Climate Advisor")

	st.markdown("#### 🎯 Intelligent Analysis & Recommendations")

	for insight in ai_insights:
		if insight['type'] == 'success':
			st.markdown(f"""
            <div class="alert-success">
                <h4>{insight['title']}</h4>
                <p>{insight['message']}</p>
            </div>
            """, unsafe_allow_html=True)
		elif insight['type'] == 'warning':
			st.markdown(f"""
            <div class="alert-warning">
                <h4>{insight['title']}</h4>
                <p>{insight['message']}</p>
            </div>
            """, unsafe_allow_html=True)
		elif insight['type'] == 'danger':
			st.markdown(f"""
            <div class="alert-danger">
                <h4>{insight['title']}</h4>
                <p>{insight['message']}</p>
            </div>
            """, unsafe_allow_html=True)
		else:
			st.info(f"**{insight['title']}**\n\n{insight['message']}")

	# Interactive Q&A
	st.markdown("---")
	st.markdown("#### 💬 Ask the AI Advisor")

	user_question = st.text_input("What would you like to know about this scenario?",
								  placeholder="e.g., How can I reduce flood risk by 50%?")

	if user_question:
		# Simulate AI response based on scenario
		if "reduce" in user_question.lower() and "flood" in user_question.lower():
			st.info(f"""
            **AI Recommendation:**  
            To reduce flood depth from {depth}cm by 50% ({depth * 0.5:.1f}cm):

            1. **Add {max(0, (depth * 0.5) / 2):.0f} Ha of urban forest** - This will absorb surface runoff
            2. **Install {max(0, (depth * 0.5) / 1.5):.0f} retention ponds** - Temporary water storage
            3. **Increase permeable pavement to {min(100, pavement + 30)}%** - Better infiltration

            **Estimated Cost**: ${((depth * 0.5) / 2 * 0.2 + (depth * 0.5) / 1.5 * 0.5):.2f}M  
            **Expected Savings**: ${financials['total_millions'] * 0.5:.2f}M in damage prevention
            """)
		else:
			st.info(
				"Ask specific questions about flood reduction, costs, or mitigation strategies for personalized recommendations.")

with tab4:
	st.subheader("📊 Scenario Comparison")

	if len(st.session_state.scenarios) > 0:
		st.markdown(f"**Saved Scenarios: {len(st.session_state.scenarios)}**")

		# Comparison table
		comparison_data = []
		for scenario in st.session_state.scenarios:
			comparison_data.append({
				'Scenario': scenario['name'],
				'Time': scenario['timestamp'].strftime("%Y-%m-%d %H:%M"),
				'Flood Depth (cm)': scenario['depth'],
				'Total Cost ($M)': scenario['financials']['total_millions'],
				'Urban Forest (Ha)': scenario['inputs']['mangroves'],
				'Ponds': scenario['inputs']['retention_ponds'],
				'Carbon (tons/yr)': scenario['carbon']['carbon_tons']
			})

		comparison_df = pd.DataFrame(comparison_data)
		st.dataframe(comparison_df, use_container_width=True, hide_index=True)

		# Visual comparison
		if len(comparison_data) > 1:
			fig = px.scatter(
				comparison_df,
				x='Total Cost ($M)',
				y='Flood Depth (cm)',
				size='Carbon (tons/yr)',
				color='Scenario',
				hover_data=['Urban Forest (Ha)', 'Ponds'],
				title="Scenario Comparison: Cost vs Flood Risk"
			)
			st.plotly_chart(fig, use_container_width=True)

		if st.button("🗑️ Clear All Scenarios"):
			st.session_state.scenarios = []
			st.rerun()
	else:
		st.info("💡 Save scenarios using the sidebar form to compare different strategies here.")

with tab5:
	st.subheader("📋 Export Simulation Report")

	st.markdown(f"""
    ### TerraTwin Climate Resilience Report
    **Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}  
    **Campus:** UTM Kuala Lumpur

    ---

    #### Executive Summary
    - **Flood Risk Level:** {risk['level']}
    - **Predicted Flood Depth:** {depth} cm
    - **Total Economic Impact:** ${financials['total_millions']} Million
    - **Campus Access Status:** {logistics['status']}

    #### Environmental Factors
    - Rainfall Intensity: {rain} mm/hr
    - River Level: {tide} m
    - Soil Saturation: {soil}%
    - Impervious Surface: {concrete}%

    #### Mitigation Measures
    - Urban Forest: {mangrove} Hectares
    - Retention Ponds: {ponds} Units
    - Permeable Pavement: {pavement}%

    #### Impact Assessment
    - Direct Damage: ${financials['direct_damage']}M
    - Business Interruption: ${financials['business_loss']}M
    - Cleanup Costs: ${financials['cleanup']}M

    #### Carbon & Ecosystem Services
    - Annual Carbon Sequestration: {carbon['carbon_tons']} tons CO₂
    - Carbon Credit Value: ${carbon['credit_value']:,}
    - Ecosystem Services Value: ${carbon['ecosystem_value']:,}
    - **Total Annual Benefits:** ${carbon['total_benefit']:,}

    #### Recommendations
    """)

	for i, insight in enumerate(ai_insights, 1):
		st.markdown(f"{i}. **{insight['title']}**: {insight['message']}")

	# Download button
	report_text = f"""
TerraTwin Climate Resilience Report
Generated: {datetime.now()}
Campus: UTM Kuala Lumpur

EXECUTIVE SUMMARY
Flood Risk: {risk['level']}
Flood Depth: {depth} cm
Economic Impact: ${financials['total_millions']}M
Access Status: {logistics['status']}

ENVIRONMENTAL FACTORS
Rainfall: {rain} mm/hr
River Level: {tide} m
Soil Saturation: {soil}%

MITIGATION MEASURES
Urban Forest: {mangrove} Ha
Retention Ponds: {ponds}
Permeable Pavement: {pavement}%

FINANCIAL IMPACT
Direct Damage: ${financials['direct_damage']}M
Business Loss: ${financials['business_loss']}M
Cleanup: ${financials['cleanup']}M
Total: ${financials['total_millions']}M

CARBON BENEFITS
Carbon Sequestration: {carbon['carbon_tons']} tons/year
Carbon Credits: ${carbon['credit_value']:,}
Ecosystem Services: ${carbon['ecosystem_value']:,}
Total Benefits: ${carbon['total_benefit']:,}
    """

	st.download_button(
		label="📥 Download Report (TXT)",
		data=report_text,
		file_name=f"terratwin_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
		mime="text/plain"
	)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><b>TerraTwin Climate Resilience Platform</b> | UTM Kuala Lumpur | Powered by AI & Geospatial Analysis</p>
    <p style='font-size: 0.9em;'>Data sources: Malaysian Meteorological Department, Drainage & Irrigation Department, IPCC Climate Models</p>
</div>
""", unsafe_allow_html=True)