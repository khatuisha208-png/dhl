import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="DHL: The Efficiency Gap", layout="wide")

# --- 1. CORE OPERATIONAL VARIABLES (2026 REFINED) ---
# Wasted costs when stuck in peak traffic
IDLING_FUEL_COST_HR = 4.80  # Fuel burned doing 0 miles
PEAK_LABOR_WASTE = 1.30     # 30% "inefficiency tax" on labor time
MAINTENANCE_FACTOR = 1.60   # 60% higher wear-and-tear in stop-and-go

# Efficiency gains in Non-Peak windows
OFF_PEAK_SPEED_BOOST = 1.25 # 25% faster travel/delivery speed

# --- 2. THE BALANCE SIMULATOR ---
st.title("⚖️ DHL Strategic Balance: Peak vs. Non-Peak")
st.markdown("Finding the 'Sweet Spot' between **Service Responsiveness** and **Cost Leadership**.")

st.sidebar.header("Global Logistics Settings")
daily_vol = st.sidebar.number_input("Total Daily Shipments", value=12000)
shift_target = st.sidebar.slider("Strategy: Volume Shift to Non-Peak (%)", 0, 100, 35)

# --- 3. THE CALCULATION ENGINE ---
days = 30 # Simulate one month
actual_costs, proposed_costs = [], []

for _ in range(days):
    # ACTUAL MODEL (100% Peak-Heavy)
    # High labor hours + Idling waste + Maintenance penalty
    a_labor = (daily_vol / 12) * 45 * PEAK_LABOR_WASTE 
    a_fuel = (daily_vol * 0.15) * 1.2 * 1.4 # High burn rate in traffic
    a_maint = (daily_vol * 0.05) * MAINTENANCE_FACTOR
    actual_costs.append(a_labor + a_fuel + a_maint)
    
    # PROPOSED MODEL (The Hybrid Balance)
    v_peak = daily_vol * (1 - shift_target/100)
    v_off = daily_vol * (shift_target/100)
    
    # Peak portion remains expensive (maintains responsiveness)
    p_cost_peak = ((v_peak/12)*45*PEAK_LABOR_WASTE) + ((v_peak*0.15)*1.2*1.4) + (v_peak*0.05*MAINTENANCE_FACTOR)
    
    # Non-Peak portion is optimized (achieves cost leadership)
    p_cost_off = ((v_off/(12*OFF_PEAK_SPEED_BOOST))*45) + ((v_off*0.15)*1.2) + (v_off*0.05)
    
    proposed_costs.append(p_cost_peak + p_cost_off)

# --- 4. VISUALIZING THE GAP ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Monthly Operational Spending")
    plot_df = pd.DataFrame({
        "Day": list(range(1, 31)),
        "Actual (Legacy)": actual_costs,
        "Proposed (Balanced)": proposed_costs
    })
    fig = px.line(plot_df, x="Day", y=["Actual (Legacy)", "Proposed (Balanced)"],
                  color_discrete_sequence=['#D40511', '#FFCC00'])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    savings = sum(actual_costs) - sum(proposed_costs)
    st.metric("Total Monthly Savings", f"${savings:,.2f}")
    st.write("### Why this works:")
    st.write(f"- **Asset Utilization:** Shifting **{shift_target}%** to non-peak frees up vehicles for express peak-hour demand.")
    st.write(f"- **Maintenance:** Reducing stop-and-go idling saves approx. **${(sum(actual_costs)*0.08):,.0f}** in engine wear-and-tear.")
    st.write(f"- **Fuel Efficiency:** You avoid the **45% Congestion Penalty** on {shift_target}% of your total volume.")



# --- 5. THE STRATEGY MATRIX ---
st.divider()
st.subheader("Strategic Pillar Alignment")
matrix_data = {
    "Pillar": ["Cost Leadership", "Responsiveness", "Sustainability"],
    "Action": ["Off-Peak Resource Shift", "Reserved Peak-Hour Capacity", "Reduced Idling Emissions"],
    "Impact": ["Lower Fuel/Labor/Maint", "Guaranteed Express SLA", "Lower CO2 per Parcel"]
}
st.table(pd.DataFrame(matrix_data))
