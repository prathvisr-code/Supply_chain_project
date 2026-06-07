import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="Proactive FMCG Smart Inventory System",
    page_icon="📦",
    layout="wide"
)

# 2. Load Assets (Cached to ensure lightning-fast dashboard navigation)
@st.cache_data
def load_data():
    df = pd.read_csv('inventory_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_resource
def load_model():
    # Loading our single optimized 7-day lookahead brain
    model_7d = joblib.load('model_7d.pkl')
    return model_7d

try:
    df = load_data()
    model_7d = load_model()
except FileNotFoundError:
    st.error("⚠️ Missing project files! Please make sure 'inventory_data.csv' and 'model_7d.pkl' are in this exact folder.")
    st.stop()

# 3. Sidebar Setup (Product Selector & Timeline Simulation)
st.sidebar.header("⚙️ Supply Chain Controls")
st.sidebar.markdown("---")

sku_options = df['SKU_ID'].unique()
selected_sku = st.sidebar.selectbox("Select Product SKU:", sku_options)

sku_data = df[df['SKU_ID'] == selected_sku].sort_values(by='Date')
product_name = sku_data['Product_Name'].iloc[0]

# Slider simulating an inventory manager scrolling through active days
available_dates = sku_data['Date'].dt.date.unique()
selected_date = st.sidebar.select_slider(
    "Operational Timeline View:", 
    options=available_dates, 
    value=available_dates[-15] # Anchors view near the end of the timeline
)

# Filter data to that single simulated operational day
current_day_row = sku_data[sku_data['Date'].dt.date == selected_date]

if current_day_row.empty:
    st.warning("No data found for this simulated date boundary.")
    st.stop()
else:
    current_day_row = current_day_row.iloc[0]

# Extract daily supply chain metrics
current_stock = int(current_day_row['Current_Stock'])
eoq = int(current_day_row['EOQ'])
rop = int(current_day_row['Reorder_Point'])
safety_stock = int(current_day_row['Safety_Stock'])

# --- MAIN DASHBOARD INTERFACE ---
st.title("📦 Smart Inventory Reorder & Stockout Prediction System")
st.markdown(f"### Current SKU Focus: **{selected_sku} — {product_name}**")
st.markdown(f"**Operational Reporting Date:** `{selected_date.strftime('%B %d, %Y')}`")
st.markdown("---")

# 4. ROW 1: Operations Research Metric Cards
st.subheader("📊 Fundamental Supply Chain Parameters")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="On-Hand Stock Level", value=f"{current_stock} units")
with col2:
    st.metric(label="Economic Order Quantity (EOQ)", value=f"{eoq} units", help="The mathematically optimal order volume calculated via Wilson's formula.")
with col3:
    st.metric(label="Reorder Point (ROP)", value=f"{rop} units", help="The threshold boundary where replenishment is traditionally triggered.")
with col4:
    st.metric(label="Safety Stock Buffer", value=f"{safety_stock} units", help="Emergency backstock held to absorb consumer demand variance.")

st.markdown("---")

# 5. ROW 2: Balanced Operational Alert Engine
st.subheader("🤖 Dual-Horizon Operational Risk Panel")
col_deterministic, col_predictive = st.columns(2)

# Preparing model variables precisely matching our 7-Day features matrix
# ['Current_Stock', 'Demand_Lag_1', 'Rolling_Avg_Demand_7D', 'Rolling_Std_Demand_7D', 'Inventory_Runway', 'Is_Weekend']
is_weekend = 1 if selected_date.weekday() in [4, 5, 6] else 0
historical_slice = sku_data[sku_data['Date'].dt.date <= selected_date].tail(7)
rolling_avg_demand = historical_slice['Demand'].mean()
rolling_std_demand = historical_slice['Demand'].std() if len(historical_slice) > 1 else 1
demand_lag_1 = historical_slice['Demand'].iloc[-1] if not historical_slice.empty else current_day_row['Demand']
inventory_runway = current_stock / (rolling_avg_demand + 1)

features_input = np.array([[
    current_stock, demand_lag_1, rolling_avg_demand, 
    rolling_std_demand, inventory_runway, is_weekend
]])

# Execute prediction using our custom calibrated 20% risk threshold
prob_7d = model_7d.predict_proba(features_input)[0][1]
pred_7d = 1 if prob_7d > 0.20 else 0

with col_deterministic:
    st.markdown("### 📋 Current Stock Status (Deterministic Rule)")
    if current_stock == 0:
        st.error("🚨 **OUT OF STOCK:** Current stock is empty. Execute immediate emergency store transfer.")
    elif current_stock <= rop:
        st.warning(f"⚠️ **REORDER BOUNDARY REACHED:** Stock level has dipped below ROP ({rop} units). Standard procurement advised.")
    else:
        st.success("🟢 **HEALTHY RUNWAY:** Existing shelf inventory is safely above standard reorder thresholds.")

with col_predictive:  # <-- Fixed column variable name here
    st.markdown("### 📅 7-Day Proactive Risk Forecasting (ML Prediction)")
    if pred_7d == 1:
        st.markdown(
            f"""
            <div style="background-color:#ffe6e6; padding:15px; border-radius:10px; border-left: 5px solid #ff4d4d;">
                <h4 style="color:#cc0000; margin:0;">⚠️ AI STOCKOUT ALERT</h4>
                <p style="color:#330000; margin-top:10px;">
                    Model forecasts a <b>{prob_7d:.1%} probability</b> of stock exhaustion within the upcoming 7 days.<br>
                    <b>Recommended Action:</b> Dispatch an advance order of <b>{eoq} units (EOQ)</b> to outrun supplier lead time.
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="background-color:#e6f9ff; padding:15px; border-radius:10px; border-left: 5px solid #33ccff;">
                <h4 style="color:#006699; margin:0;">✅ RUNWAY SECURE</h4>
                <p style="color:#002233; margin-top:10px;">
                    Model predicts a negligible <b>{prob_7d:.1%} risk</b> of stocking out this week. Logistics runway is stable.
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )

st.markdown("---")

# 6. ROW 3: Time-Series Line Chart Visualizations
st.subheader("📈 Inventory Run-Rate & Sawtooth Analytics")
st.markdown("Analyzing inventory depletion run-rates across the last 30 operational days against math floors.")

chart_data = sku_data[sku_data['Date'].dt.date <= selected_date].tail(30)

if not chart_data.empty:
    chart_df = pd.DataFrame({
        'Date': chart_data['Date'],
        'On-Hand Stock': chart_data['Current_Stock'],
        'Reorder Threshold (ROP)': chart_data['Reorder_Point'],
        'Safety Stock Floor': chart_data['Safety_Stock']
    }).set_index('Date')
    
    # Render Streamlit's built-in chart engine using clear categorical colors
    st.line_chart(chart_df, color=["#29b5e8", "#ff9922", "#ff4444"])
else:
    st.text("Insufficient historical boundaries to display runtime trends.")