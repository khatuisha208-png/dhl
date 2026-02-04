import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="DHL RL Strategy Sim", layout="wide")

# --- 1. Q-LEARNING ENGINE (DEBUGGED) ---
class DHL_RL_Agent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.actions = actions 
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        # Q-Table: 2 States (Normal, Congested) x 4 Actions
        self.q_table = np.zeros((2, len(actions)))

    def choose_action(self, state):
        # Epsilon-greedy: Ensures we don't get stuck in a loop
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions) 
        else:
            return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state):
        # The Bellman Equation: Update the 'memory' of the strategy
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.lr * (target - predict)

# --- 2. USER INTERFACE ---
st.title("🤖 DHL Strategy: Reinforcement Learning Model")
st.markdown("This agent learns the optimal **Time-Bound** strategy through trial and error.")

# Inputs
episodes = st.sidebar.slider("Training Episodes (Iterations)", 10, 1000, 250)
fuel_price = st.sidebar.slider("Fuel Price Multiplier", 0.5, 2.0, 1.2)

# --- 3. EXECUTION ---
try:
    agent = DHL_RL_Agent(actions=[0, 1, 2, 3])
    history = []

    # Training Loop
    for _ in range(episodes):
        state = np.random.choice([0, 1]) 
        action = agent.choose_action(state)
        
        # Reward Logic (Higher is better)
        if action == 1: # Non-Peak Road Delivery
            reward = 12 * fuel_price 
        elif action == 3: # Low-Traffic Air Express
            reward = 20 
        elif action == 0: # Peak Road (Traffic waste)
            reward = -10
        else: # High Air Traffic (Fuel waste)
            reward = -15
            
        agent.learn(state, action, reward, next_state=0)
        history.append(reward)

    # --- 4. DATA VISUALIZATION ---
    # Convert Q-Table to a clean DataFrame for Streamlit display
    q_df = pd.DataFrame(
        agent.q_table, 
        columns=["Peak Road", "Non-Peak Road", "High Air Traffic", "Low Air Traffic"],
        index=["Normal Day", "Congested Day"]
    )

    st.subheader("Learned Strategy Intelligence (Q-Table)")
    st.dataframe(q_df.style.highlight_max(axis=1, color='#00A94F'))

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Learning Progress")
        fig = px.line(y=np.cumsum(history), 
                      labels={'x': 'Episodes', 'y': 'Cumulative Reward'},
                      color_discrete_sequence=['#D40511'])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Strategy Alignment")
        # Showing the chosen optimal path
        opt_action = q_df.columns[np.argmax(agent.q_table[1])]
        st.success(f"**RL Agent Recommendation:** On congested days, prioritize **{opt_action}**.")
        st.info("This matches the 'Cost Leadership' pillar of the proposed strategy.")

except Exception as e:
    st.error(f"Error in Simulation Engine: {e}")
