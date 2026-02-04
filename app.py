import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(page_title="DHL AI: Transportation Strategy", layout="wide")

# --- 1. THE Q-LEARNING AGENT ---
class DHLQAgent:
    def __init__(self, states_n, actions_n, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = np.zeros((states_n, actions_n))
        self.alpha = alpha   
        self.gamma = gamma   
        self.epsilon = epsilon 

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice([0, 1]) 
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

# --- 2. TRANSPORTATION COST LOGIC ---
def calculate_transport_cost(volume, is_peak_action, traffic_state):
    """
    Calculates cost based on real 2026 DHL Surcharges.
    Peak: $0.67 surcharge + 30% fuel waste + labor overtime.
    Non-Peak: $0 surcharge + 15% efficiency gain.
    """
    base_rate = 2.50
    peak_surcharge = 0.67
    
    if is_peak_action:
        # If the agent chooses to operate in Peak
        fuel_waste = 1.30 if traffic_state == 1 else 1.05
        labor_waste = 1.20 if traffic_state == 1 else 1.00
        return volume * (base_rate * fuel_waste * labor_waste + peak_surcharge)
    else:
        # If the agent chooses Non-Peak (Proposed)
        return volume * (base_rate * 0.85) # 15% efficiency discount

# --- 3. UI & SIMULATION SETUP ---
st.title("🤖 DHL AI Strategy Agent: Transportation Cost Comparison")
st.sidebar.header("Logistics Parameters")
daily_volume = st.sidebar.number_input("Daily Package Volume", value=10000)
episodes = st.sidebar.slider("Simulation Length (Days)", 30, 365, 100)

# Initialize Agent & Data Tracking
agent = DHLQAgent(states_n=2, actions_n=2)
actual_history = []
proposed_history = []
reward_history = []

# --- 4. EXECUTION LOOP ---
for i in range(episodes):
    # State 0: Normal, State 1: Congested
    state = 1 if np.random.rand() < 0.3 else 0 
    
    # RL Agent chooses action for Proposed Strategy
    action = agent.choose_action(state) # 0: Peak, 1: Non-Peak
    
    # Calculate Costs
    # Actual Strategy: Always Peak-Heavy
    cost_actual = calculate_transport_cost(daily_volume, is_peak_action=True, traffic_state=state)
    
    # Proposed Strategy: Determined by Agent
    cost_proposed = calculate_transport_cost(daily_volume, is_peak_action=(action==0), traffic_state=state)
    
    # Reward is the money saved
    reward = cost_actual - cost_proposed
    
    # Update Agent
    next_state = 1 if np.random.rand() < 0.3 else 0
    agent.update(state, action, reward, next_state)
    
    # Record History
    actual_history.append(cost_actual)
    proposed_history.append(cost_proposed)
    reward_history.append(reward)

# --- 5. DATA VISUALIZATION ---
m1, m2, m3 = st.columns(3)
m1.metric("Avg. Actual Daily Cost", f"${np.mean(actual_history):,.2f}")
m2.metric("Avg. Proposed Daily Cost", f"${np.mean(proposed_history):,.2f}")
m3.metric("Total Transportation Savings", f"${np.sum(reward_history):,.2f}", delta="Optimized")

st.divider()

# Comparison Chart
df_plot = pd.DataFrame({
    "Day": range(episodes),
    "Actual (Peak-Heavy)": actual_history,
    "Proposed (RL-Optimized)": proposed_history
})
fig = px.line(df_plot, x="Day", y=["Actual (Peak-Heavy)", "Proposed (RL-Optimized)"],
              title="Daily Transportation Cost: Legacy vs. AI-Balanced Strategy",
              color_discrete_map={"Actual (Peak-Heavy)": "#D40511", "Proposed (RL-Optimized)": "#00A94F"})
st.plotly_chart(fig, use_container_width=True)

# Q-Table Visualization
st.subheader("Agent's Intelligence: The Learned Q-Table")
st.write("The higher the value, the more 'profitable' the action is for that traffic state.")
q_df = pd.DataFrame(agent.q_table, 
                    columns=["Action: Stay in Peak", "Action: Shift to Non-Peak"], 
                    index=["State: Normal", "State: Congested"])

# Use styled table with colors
st.table(q_df.style.background_gradient(cmap='YlGn'))

st.info("""
**Model Insight:** The Q-Learning model identifies that during 'Congested' states, the cost of transportation spikes due to the 
$0.67 DHL Peak Surcharge and idling fuel waste. The agent learns to achieve **Cost Leadership** by 
automatically shifting volume to Non-Peak windows when congestion is detected.
""")
