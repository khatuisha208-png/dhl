import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="DHL Financial Strategy Sim", layout="wide")

# --- 1. DEFINE STRATEGIC VARIABLES ---
STRATEGY_MAP = {
    0: "Peak Hour Road (Traditional)",
    1: "Non-Peak Road (Proposed - Cost Leadership)",
    2: "Peak Air Traffic (Traditional)",
    3: "Non-Peak Air Traffic (Proposed - Responsiveness)"
}

# --- 2. REINFORCEMENT LEARNING ENGINE ---
class DHL_RL_Agent:
    def __init__(self, actions, lr=0.1, gamma=0.9, epsilon=0.2):
        self.actions = actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((2, len(actions)))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions)
        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.lr * (target - predict)

# --- 3. UI & FINANCIAL PARAMETERS ---
st.title("💰 DHL Strategy: Financial Impact Simulation")
st.markdown("Comparing **Actual Operations** vs. **Proposed Time-Bound Strategy** using RL.")

st.sidebar.header("Economic Inputs")
fuel_cost_per_liter = st.sidebar.number_input("Jet Fuel Cost ($/Liter)", value=1.10)
labor_cost_per_hour = st.sidebar.number_input("Driver/Pilot Labor ($/Hour)", value=45.0)
daily_volume = st.sidebar.slider("Daily Shipment Volume", 5000, 50000, 15000)

# Initialize Agent
agent = DHL_RL_Agent(actions=list(STRATEGY_MAP.keys()))
total_actual_cost = 0
total_proposed_cost = 0

# --- 4. SIMULATION LOOP (Financial Calculation) ---
for _ in range(500): # Training episodes
    state = np.random.choice([0, 1]) 
    action = agent.choose_action(state)
    
    # Financial Logic
    # ACTUAL (Peak) - High waste
    peak_waste = (fuel_cost_per_liter * 1.5) + (labor_cost_per_hour * 1.3)
    # PROPOSED (Non-Peak) - Optimized
    off_peak_save = (fuel_cost_per_liter * 1.0) + (labor_cost_per_hour * 1.0)
    
    if action in [1, 3]: # Proposed Actions
        reward = 20 
        total_proposed_cost += off_peak_save * (daily_volume / 1000)
    else: # Actual/Peak Actions
        reward = -10
        total_actual_cost += peak_waste * (daily_volume / 1000)
        
    agent.learn(state, action, reward, next_state=0)

# --- 5. RESULTS DISPLAY ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Financial Comparison (Daily OpEx)")
    comparison_df = pd.DataFrame({
        "Strategy": ["Actual (Peak-Heavy)", "Proposed (Time-Bound)"],
        "Daily Cost ($)": [total_actual_cost / 500, total_proposed_cost / 500]
    })
    fig_money = px.bar(comparison_df, x="Strategy", y="Daily Cost ($)", 
                       color="Strategy", color_discrete_map={"Actual (Peak-Heavy)": "#D40511", "Proposed (Time-Bound)": "#00A94F"})
    st.plotly_chart(fig_money, use_container_width=True)

with col2:
    savings = (total_actual_cost - total_proposed_cost) / 500
    st.metric("Potential Daily Savings", f"${savings:,.2f}", delta="Optimized")
    st.write("**Strategy Analysis:**")
    st.write("- **Road:** Non-peak avoids $20\%$ fuel waste from idling.")
    st.write("- **Air:** Non-peak air avoids runway 'holding patterns' which cost ~$1,000/hr in fuel.")

st.divider()
st.subheader("Learned Optimal Strategy (Q-Table)")
q_df = pd.DataFrame(agent.q_table, columns=STRATEGY_MAP.values(), index=["Normal Day", "Congested Day"])
st.dataframe(q_df.style.highlight_max(axis=1, color='#FFCC00'))
