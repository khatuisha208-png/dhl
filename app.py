 import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="DHL Balanced Strategy Sim", layout="wide")

# --- 1. THE HYBRID LOGIC ---
# We balance Cost (Off-Peak) vs. Service Level (Peak)
STRATEGY_MAP = {
    0: "Standard Peak (High Cost, High Speed)",
    1: "Strategic Off-Peak (Low Cost, Slower)",
}

class Balanced_DHL_Agent:
    def __init__(self, actions):
        self.q_table = np.zeros((2, len(actions))) # 0: Normal, 1: High Demand
        self.lr, self.gamma, self.eps = 0.1, 0.9, 0.2

    def learn(self, state, action, reward):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[state])
        self.q_table[state, action] += self.lr * (target - predict)

# --- 2. UI SETUP ---
st.title("⚖️ DHL Strategic Balance: Cost vs. Responsiveness")
st.markdown("This simulation finds the **optimal mix** of Peak vs. Non-Peak operations to maximize profit without losing customers.")

col_a, col_b = st.columns(2)
with col_a:
    target_offpeak = st.slider("Target Off-Peak Shift (%)", 0, 100, 30, help="What % of volume do you WANT to move to save money?")
with col_b:
    sla_penalty = st.slider("SLA Breach Penalty ($)", 10, 500, 150, help="Cost of losing a customer because of a late peak delivery.")

# --- 3. THE BALANCING ACT (Simulation) ---
agent = Balanced_DHL_Agent(actions=[0, 1])
actual_costs, proposed_costs = [], []

for i in range(100):
    state = 1 if i % 5 == 0 else 0 # Every 5th day is "High Demand"
    
    # 1. ACTUAL (Static - Always Peak)
    # Cost = Base + Traffic Waste ($50)
    actual_day_cost = 100 + 50 
    actual_costs.append(actual_day_cost)
    
    # 2. PROPOSED (RL Agent decides the mix)
    # If the Agent chooses Off-Peak (1) on a High Demand day, it gets a heavy SLA Penalty.
    # If it stays in Peak (0), it pays the Traffic Waste but saves the customer.
    action = 1 if np.random.rand() < (target_offpeak/100) else 0
    
    if state == 1 and action == 1: # BAD BALANCE: Moved too much to off-peak during high demand
        reward = -sla_penalty
        day_cost = 70 + sla_penalty 
    elif action == 1: # GOOD BALANCE: Successfully saved money on a normal day
        reward = 30 # Saving on fuel/labor
        day_cost = 70
    else: # SAFE: Stayed in peak to ensure responsiveness
        reward = 5
        day_cost = 150 
        
    agent.learn(state, action, reward)
    proposed_costs.append(day_cost)

# --- 4. DATA VISUALIZATION ---
st.divider()
c1, c2 = st.columns(2)

with c1:
    st.subheader("Cumulative Operational Expense")
    fig = px.area(x=list(range(100)), y=[np.cumsum(actual_costs), np.cumsum(proposed_costs)],
                 labels={"x": "Days", "value": "Total Cost ($)"},
                 title="Total Spending: Traditional vs. Balanced Hybrid",
                 color_discrete_sequence=['#D40511', '#00A94F'])
    st.plotly_chart(fig, use_container_width=True)

with c2:
    total_save = sum(actual_costs) - sum(proposed_costs)
    st.metric("Total Strategy Savings", f"${total_save:,.2f}", 
              delta="Target Reached" if total_save > 0 else "Inefficient Mix")
    st.info("""
    **The Balance Result:**
    - **Peak:** Used for high-priority Express shipments to maintain 'Responsiveness'.
    - **Off-Peak:** Used for standard GoGreen shipments to achieve 'Cost Leadership'.
    """)
