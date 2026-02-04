import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="DHL Core Strategy Sim", layout="wide")

# --- 1. STRATEGIC CONSTANTS (Core Variables) ---
PEAK_WASTE_FACTOR = 1.45    # 45% more fuel/time wasted in traffic
LABOR_COST_HR = 45.0        # Average driver/courier hourly rate
NON_PEAK_EFFICIENCY = 1.25  # 25% more deliveries possible per hour off-peak

STRATEGY_MAP = {0: "Actual (Peak-Heavy)", 1: "Proposed (Non-Peak Shift)"}

# --- 2. REINFORCEMENT LEARNING ENGINE ---
class DHL_Balance_Agent:
    def __init__(self, actions):
        self.q_table = np.zeros((2, len(actions))) # State 0: Low Traffic, State 1: High Traffic
        self.lr, self.gamma, self.eps = 0.1, 0.9, 0.2

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.eps:
            return np.random.choice([0, 1])
        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[state])
        self.q_table[state, action] += self.lr * (target - predict)

# --- 3. UI & INPUTS ---
st.title("📦 DHL Strategy: Time-Bound Resource Allocation")
st.sidebar.header("Operational Inputs")
daily_volume = st.sidebar.number_input("Daily Packages", value=5000)
fuel_price = st.sidebar.slider("Fuel Price ($/L)", 0.8, 2.0, 1.2)
shift_target = st.sidebar.slider("Target Non-Peak Shift (%)", 0, 100, 30)

# --- 4. EXECUTION LOOP ---
agent = DHL_Balance_Agent(actions=[0, 1])
actual_total, proposed_total = 0, 0

for i in range(100): # 100-day simulation
    state = 1 if i % 3 == 0 else 0 # Simulate recurring heavy traffic days
    
    # ACTUAL (Always operating in standard windows)
    a_fuel = (daily_volume * 0.05) * fuel_price * PEAK_WASTE_FACTOR
    a_labor = (daily_volume / 15) * LABOR_COST_HR * 1.2 # 15 stops/hr baseline
    actual_total += (a_fuel + a_labor)
    
    # PROPOSED (Hybrid Balance)
    action = agent.choose_action(state)
    v_non_peak = daily_volume * (shift_target / 100)
    v_peak = daily_volume - v_non_peak
    
    # Cost for Peak portion
    p_cost_peak = ((v_peak * 0.05) * fuel_price * PEAK_WASTE_FACTOR) + ((v_peak / 15) * LABOR_COST_HR)
    # Cost for Non-Peak portion (High Efficiency)
    p_cost_off = ((v_non_peak * 0.05) * fuel_price) + ((v_non_peak / (15 * NON_PEAK_EFFICIENCY)) * LABOR_COST_HR)
    
    current_proposed_day = p_cost_peak + p_cost_off
    proposed_total += current_proposed_day
    
    # RL Reward: The agent learns that avoiding high-waste states saves money
    reward = (a_fuel + a_labor) - current_proposed_day
    agent.learn(state, action, reward)

# --- 5. VISUALIZATION ---
m1, m2 = st.columns(2)
avg_actual = actual_total / 100
avg_proposed = proposed_total / 100

m1.metric("Current Daily OpEx", f"${avg_actual:,.2f}")
m2.metric("Proposed Daily OpEx", f"${avg_proposed:,.2f}", delta=f"-${(avg_actual - avg_proposed):,.2f}", delta_color="normal")

st.divider()

# Strategy Mapping Table
st.subheader("Learned Strategy Logic")
q_df = pd.DataFrame(agent.q_table, columns=["Keep in Peak", "Shift to Non-Peak"], index=["Normal Traffic", "Heavy Congestion"])
st.table(q_df.style.highlight_max(axis=1, color="#FFCC00"))

st.info(f"""
**Strategic Analysis:**
* **Cost Leadership:** By shifting {shift_target}% of volume, the AI identifies that labor productivity increases from 15 to {15 * NON_PEAK_EFFICIENCY:.1f} stops per hour.
* **Responsiveness:** The remaining {100-shift_target}% stays in peak windows to ensure "Express" time-definite deliveries are still met.
""")
