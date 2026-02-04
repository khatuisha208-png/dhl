# --- REFINED TRANSPORTATION COST CALCULATION ---

def calculate_transport_cost(volume, is_peak, traffic_state):
    # Base transport cost per package
    base_rate = 2.50 
    
    if is_peak and traffic_state == 1: # High Congestion
        # Cost = Base + Idling Fuel + Overtime Premium + Peak Surcharge ($0.67)
        return volume * (base_rate + 1.15 + 0.67) 
    elif not is_peak:
        # Cost = Base - Fuel Savings (due to steady speed) - Consolidated Load Discount
        return volume * (base_rate - 0.35) 
    else:
        return volume * base_rate

# --- RL AGENT REWARD LOGIC ---
# The agent compares its chosen cost to the "Actual" baseline
actual_cost = calculate_transport_cost(daily_volume, is_peak=True, traffic_state=state)
proposed_cost = calculate_transport_cost(daily_volume, is_peak=(action==0), traffic_state=state)

# Reward is the money saved on transportation
reward = actual_cost - proposed_cost
