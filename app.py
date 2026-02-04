import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="DHL Advanced Strategy Sim", layout="wide")

# --- 1. STRATEGIC CONSTANTS (2026 DATA) ---
PEAK_SURCHARGE = 0.67       # Per package cost during peak
SAF_PREMIUM = 0.22          # Extra cost per kg for GoGreen Plus
MAINTENANCE_WASTE = 1.60    # 60% increase in wear/tear during peak

STRATEGY_MAP = {0: "Actual (Peak-Heavy)", 1: "Proposed (Hybrid/Time-Bound)"}

# --- 2. ENHANCED SIMULATION ENGINE ---
def run_advanced_sim(vol, off_peak_pct, fuel_price, saf_adoption):
    days = 30 # Simulation for one operational month
    actual_total, proposed_total = 0, 0
    actual_co2, proposed_co2 = 0, 0
    
    for _ in range(days):
        # BASE COSTS (Standard)
        base_labor = 45 * 8 # 8 hour shift
        base_fuel = (vol * 0.1) * fuel_price # 0.1 liters per pkg avg
        
        # --- ACTUAL (100% Peak Operations) ---
        # Penalties: Peak Surcharges + Maintenance + Traffic Idling
        a_cost = (base_labor * 1.3) + (base_fuel * 1.5) + (vol * PEAK_SURCHARGE)
        a_cost *= MAINTENANCE_WASTE # Apply wear-and-tear
        actual_total += a_cost
        actual_co2 += vol * 0.25 * 1.2 # Peak traffic burns 20% more fuel
        
        # --- PROPOSED (The Balanced Shift) ---
        # Volume split: some in Peak, some in Off-Peak
        v_peak = vol * (1 - off_peak_pct/100)
        v_off = vol * (off_peak_pct/100)
        
        # Off-Peak saves on surcharges and fuel waste
        p_cost_peak = (v_peak * PEAK_SURCHARGE) + ((base_labor/2) * 1.3) + ((base_fuel/2) * 1.5)
        p_cost_off = (v_off * 0) + ((base_labor/2) * 1.0) + ((base_fuel/2) * 1.0)
        
        # Sustainability Adjustment (SAF)
        saf_cost = (vol * saf_adoption/100) * SAF_PREMIUM
        
        proposed_total += (p_cost_peak + p_cost_off + saf_cost)
        
        # CO2: SAF reduces emissions by 80% for the adopted volume
        p_co2_clean = (vol * saf_adoption/100) * 0.05
        p_co2_dirty = (vol * (1 - saf_adoption/100)) * 0.25 * (1 - (off_peak_pct/100)*0.15)
        proposed_co2 += (p_co2_clean + p_co2_dirty)

    return actual_total, proposed_total, actual_co2, proposed_co2

# --- 3. UI LAYOUT ---
st.title("📊 DHL 2026 Strategic Value Simulator")
st.sidebar.header("Global Variables")
vol = st.sidebar.number_input("Daily Volume", value=10000)
fuel = st.sidebar.slider("Fuel Market Price ($/L)", 0.8, 2.5, 1.2)

st.sidebar.header("Proposed Strategy Levers")
off_peak = st.sidebar.slider("Off-Peak Volume Shift (%)", 0, 100, 40)
saf = st.sidebar.slider("SAF (GoGreen Plus) Adoption (%)", 0, 100, 30)

a_val, p_val, a_co2, p_co2 = run_advanced_sim(vol, off_peak, fuel, saf)

# --- 4. DISPLAY RESULTS ---
m1, m2, m3 = st.columns(3)
m1.metric("Financial Delta (Monthly)", f"${(a_val - p_val):,.2f}", delta="Cost Saved")
m2.metric("CO2 Mitigation", f"{((a_co2 - p_co2)/a_co2)*100:.1f}%", delta="Green Impact")
m3.metric("Response Score", f"{100 - (off_peak*0.5):.0f}/100", help="Score drops if too much is moved off-peak (Responsiveness risk)")

st.divider()

# Comparison Graph
fin_df = pd.DataFrame({
    "Category": ["Actual (Legacy)", "Proposed (Refined)"],
    "Monthly OpEx ($)": [a_val, p_val],
    "Carbon Footprint (kg)": [a_co2, p_co2]
})

fig = px.bar(fin_df, x="Category", y="Monthly OpEx ($)", color="Category", 
             color_discrete_map={"Actual (Legacy)": "#D40511", "Proposed (Refined)": "#FFCC00"})
st.plotly_chart(fig, use_container_width=True)

st.subheader("Strategy Pillar Breakdown")
tabs = st.tabs(["Cost Leadership", "Responsiveness", "Differentiation"])
with tabs[0]:
    st.write("**Variable:** Maintenance & Fuel Idling.")
    st.info(f"By shifting {off_peak}% to off-peak, you reduced wear-and-tear costs by approx. ${ (a_val * 0.15 * off_peak/100):,.0f} per month.")
with tabs[1]:
    st.write("**Variable:** Peak Surcharges & SLAs.")
    st.success(f"Reserved {100-off_peak}% capacity for 'Express' ensures DHL retains its premium speed service.")
with tabs[2]:
    st.write("**Variable:** GoGreen Plus (SAF).")
    st.warning(f"Insetting {saf}% SAF differentiates DHL from competitors who only use carbon offsets.")
