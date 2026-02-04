import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# --- RL AGENT CLASS (Q-MODEL) ---
class DHLQAgent:
    def __init__(self, states_n, actions_n, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = np.zeros((states_n, actions_n))
        self.alpha = alpha   # Learning Rate
        self.gamma = gamma   # Discount Factor
        self.epsilon = epsilon # Exploration Rate

    def choose_action(self, state):
        # Epsilon-Greedy Strategy
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice([0, 1]) # Explore
        return np.argmax(self.q_table[state]) # Exploit

    def update(self, state, action, reward, next_state):
        # Bellman Equation update
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

# --- SIMULATION CONFIG ---
st.title("🤖 DHL AI Strategy Agent (Q-Learning)")

# User Inputs for Reward Logic
fuel_penalty = st.sidebar.slider("Traffic Fuel Penalty ($)", 10, 100, 45)
efficiency_reward = st.sidebar.slider("Non-Peak Productivity Reward ($)", 10, 100, 60)

agent = DHLQAgent(states_n=2, actions_n=2) # States: Normal/Congested | Actions: Peak/Off-Peak
history = []

# Training Loop (365 "Days")
for i in range(365):
    state = 1 if i % 4 == 0 else 0 # Every 4th day is Congested
    action = agent.choose_action(state)
    
    # Calculate Reward
    if action == 1: # Shifted to Non-Peak
        reward = efficiency_reward
    else: # Stayed in Peak
        reward = -fuel_penalty if state == 1 else 10 # Penalty if stuck, small reward if clear
        
    next_state = 1 if (i+1) % 4 == 0 else 0
    agent.update(state, action, reward, next_state)
    history.append(reward)

# --- VISUALIZING THE INTELLIGENCE ---
st.subheader("The Learned Q-Table (Strategy Knowledge)")
q_df = pd.DataFrame(agent.q_table, columns=["Action: Stay in Peak", "Action: Shift Off-Peak"], 
                    index=["State: Normal Traffic", "State: Congested"])
st.table(q_df.style.background_gradient(cmap='YlGn'))

# Learning Curve
st.subheader("Agent Learning Curve")
fig = px.line(y=np.cumsum(history), labels={'x': 'Days of Experience', 'y': 'Cumulative Reward'},
              title="Agent Strategy Optimization Over Time", color_discrete_sequence=['#D40511'])
st.plotly_chart(fig, use_container_width=True)
