# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 02:10:30 2025

@author: ilker kayra orman
"""

import numpy as np
from sympy import symbols, cos, sin, solve, simplify, lambdify
import matplotlib.pyplot as plt

# Symbols for equations of motion
m1, m2, l, g = symbols('m1 m2 l g')
F = symbols('F')
x, dot_x, theta, dot_theta = symbols('x dot_x theta dot_theta')
ddot_x, ddot_theta = symbols('ddot_x ddot_theta')

# Equations of motion
e1 = (m1 + m2) * ddot_x - m2 * l * ddot_theta * cos(theta) + m2 * l * (dot_theta**2) * sin(theta) - F
e2 = l * ddot_theta - ddot_x * cos(theta) - g * sin(theta)

result = solve([e1, e2], (ddot_x, ddot_theta), dict=True)
sol_ddot_x = simplify(result[0][ddot_x]).subs({l: 1, g: 9.81, m1: 10, m2: 1})
sol_ddot_theta = simplify(result[0][ddot_theta]).subs({l: 1, g: 9.81, m1: 10, m2: 1})

func_ddot_x = lambdify([x, dot_x, theta, dot_theta, F], sol_ddot_x)
func_ddot_theta = lambdify([x, dot_x, theta, dot_theta, F], sol_ddot_theta)

# Q-learning parameters
action_space = [-20, -10, -5, 5, 10, 20]  # Moderated force values
alpha = 0.1  # Learning rate
gamma = 1  # Discount factor
epsilon = 0.8  #Exploration rate
episodes = 5000  # Number of episodes
time_step = 0.01
epsilon_decay = 0.999  # Decay 

# Discretization bins
state_bins = {
    "x": np.linspace(-5, 5, 100),  
    "dot_x": np.linspace(-5, 5, 120),
    "theta": np.linspace(-np.pi, np.pi, 200),
    "dot_theta": np.linspace(-10, 10, 120)
}

def discretize(value, bins):
    """Maps continuous value."""
    return np.digitize(value, bins) - 1

def normalize_angle(angle):
    """Normalizes an angle to the range [-pi, pi] using np.arctan2()."""
    return np.arctan2(np.sin(angle), np.cos(angle))

def get_state_index(state):
    """Combines state variables into a single tuple of indices."""
    normalized_theta = normalize_angle(state[2])
    indices = [
        discretize(state[0], state_bins["x"]),
        discretize(state[1], state_bins["dot_x"]),
        discretize(normalized_theta, state_bins["theta"]),
        discretize(state[3], state_bins["dot_theta"]),
    ]
    return tuple(indices)

def calculate_reward(state):
    x, dot_x, theta, dot_theta = state
    theta = normalize_angle(theta)
    theta_penalty = -500 * abs(theta)  
    x_penalty = -1 * abs(x) 
    dot_x_penalty = -10 * abs(dot_x)
    dot_theta_penalty = -0.5 * abs(dot_theta)
    time_reward = 20
    return theta_penalty + x_penalty + dot_x_penalty + dot_theta_penalty + time_reward

def rk4_step(state, force):
    x, dot_x, theta, dot_theta = state

    def derivatives(state, force):
        ddot_x = func_ddot_x(*state, force)
        ddot_theta = func_ddot_theta(*state, force)
        return np.array([state[1], ddot_x, state[3], ddot_theta])

    k1 = time_step * derivatives(state, force)
    k2 = time_step * derivatives(state + 0.5 * k1, force)
    k3 = time_step * derivatives(state + 0.5 * k2, force)
    k4 = time_step * derivatives(state + k3, force)
    new_state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    new_state[2] = normalize_angle(new_state[2])
    return new_state

def choose_action(state_index, theta):
    if np.random.uniform(0, 1) < epsilon:
        # Explore: Choose a random action, but bias it using the angle
        if theta > 0:
            action = np.random.choice([i for i, a in enumerate(action_space) if a < 0])
        else:
            action = np.random.choice([i for i, a in enumerate(action_space) if a > 0])
        
    else:
        # Exploit: Choose the best action according to the Q-table
        action_values = q_table[state_index]
        if theta > 0:
            action = np.argmax(action_values - np.array(action_space) * np.sign(theta))
        else:
            action = np.argmax(action_values + np.array(action_space) * np.sign(theta))
    return action

# Initialize Q-table
state_space_size = tuple(len(bins) + 1 for bins in state_bins.values())
q_table = np.zeros(state_space_size + (len(action_space),),dtype=np.float32)

# Additional variables to store the best episode
best_episode_states = None
best_episode_forces = None
best_episode_total_reward = float('-inf')

# Training loop
for episode in range(episodes):
    # Start from varying initial conditions for robustness
    state = [0, 0, np.random.uniform(-np.pi/60, np.pi/60), 0]
    state_index = get_state_index(state)
    total_reward = 0
    episode_states = []
    episode_forces = []

    terminated = False
    termination_reason = ""
    steps_in_episode = 0  #TODO not accurate
    while not terminated:
        episode_states.append(state)
        
        # Choose action using epsilon-greedy strategy
        action = choose_action(state_index, state[2])
        
        # Apply action and get next state
        force = action_space[action]
        episode_forces.append(force)
        next_state = rk4_step(state, force)
        reward = calculate_reward(next_state)
        next_state_index = get_state_index(next_state)
        
        # Q-learning update
        q_value_current = q_table[state_index + (action,)]
        q_value_max_next = np.max(q_table[next_state_index])

        td_target = reward + gamma * q_value_max_next
        td_error = td_target - q_value_current
        
        # Update Q-value
        q_table[state_index + (action,)] += alpha * td_error
        
        state = next_state
        state_index = next_state_index
        total_reward += reward
        # Increment the step count
        steps_in_episode += 1

        # Termination conditions
        if abs(state[0]) > 3:
            terminated = True
            termination_reason = "Cart out of bounds"
        elif abs(state[2]) > np.pi/8:
            terminated = True
            termination_reason = "Pole angle exceeded"

    # Update best episode if this one had a higher reward
    # TODO: when fix steps_in_episode, try with one who survived longest
    if total_reward > best_episode_total_reward:
        best_episode_total_reward = total_reward
        best_episode_states = episode_states
        best_episode_forces = episode_forces
    
    # Decay epsilon for less exploration over time
    epsilon = max(0.01, epsilon * epsilon_decay)
    # Calculate duration for the episode
    sim_duration = steps_in_episode * time_step
    if episode % 50 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Termination Reason: {termination_reason}, Duration: {sim_duration:.2f} seconds")

# Visualization of Dynamics (Best Performing Episode)
final_states = np.array(best_episode_states)
final_forces = np.array(best_episode_forces)
plt.figure(figsize=(10, 8))
time_values = np.arange(len(final_states)) * time_step

plt.plot(time_values, final_states[:, 0], 'g-', linewidth=2, label='x (m)')
plt.plot(time_values, final_states[:, 1], 'r--', alpha=0.7, linewidth=2, label='x_dot (m/s)')

plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Cart Dynamics', fontsize=14, labelpad=10)

y_min_1 = np.min(final_states[:, 1])
y_max_1 = np.max(final_states[:, 1])
# Set the y-axis limits with 1 unit added to the min and max values
plt.ylim(y_min_1 - 0.5, y_max_1 + 0.5)

# Set the x-axis limits to zoom in between 300 and 305 seconds
#plt.xlim(300, 305)

plt.legend(fontsize=10)
plt.grid()
plt.title('Cart Dynamics', fontsize=16)
plt.show()

# Pendulum Dynamics Plot
plt.figure(figsize=(10, 8))
plt.plot(time_values, final_states[:, 2], 'g-', linewidth=2, label='Theta (rad)')
plt.plot(time_values, final_states[:, 3], 'r--', alpha=0.7, linewidth=2, label='Theta_dot (rad/s)')

plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Pendulum Dynamics', fontsize=14, labelpad=10)

y_min_2 = np.min(final_states[:, 3])
y_max_2 = np.max(final_states[:, 3])
# Set the y-axis limits with 1 unit added to the min and max values
plt.ylim(y_min_2 - 0.5, y_max_2 + 0.5)

# Set the x-axis limits to zoom in between 300 and 305 seconds
#plt.xlim(300, 305)

plt.legend(fontsize=10)
plt.grid()
plt.title('Pendulum Dynamics', fontsize=16)
plt.show()

# Phase space plot (Theta vs Theta_dot)
plt.figure(figsize=(10, 8))
plt.plot(final_states[:, 3], final_states[:, 2], 'b', linewidth=2)
plt.xlabel('Theta_dot (rad/s)', fontsize=14)
plt.ylabel('Theta (rad)', fontsize=14, labelpad=10)
plt.grid()
plt.title('Phase Space Plot', fontsize=16)
plt.show()

# Force plot
plt.figure(figsize=(10, 8))
plt.plot(np.arange(len(final_forces)) * time_step, final_forces, 'k', linewidth=2, label='Force (N)')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Force (N)', fontsize=14)
plt.legend(fontsize=10)
plt.grid()
plt.title('Force Applied Over Time')
plt.show()


# Force plot (histogram)
plt.figure(figsize=(10, 8))
plt.hist(final_forces, bins=50, color='gray', edgecolor='black', alpha=0.7)
plt.xlabel('Force (N)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Force Distribution', fontsize=16)
plt.grid()
plt.show()
