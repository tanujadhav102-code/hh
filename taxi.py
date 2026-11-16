#python -m pip install numpy "gymnasium[toy_text]"


import gymnasium as gym
import numpy as np
import time

# Create Taxi environment for training
env = gym.make("Taxi-v4")

# Initialize Q-table
q_table = np.zeros(
    (env.observation_space.n, env.action_space.n)
)

alpha = 0.1
gamma = 0.9
epsilon = 0.1
episodes = 10000

print("Training the Taxi agent...")

# Training using Q-Learning
for episode in range(episodes):
    state, _ = env.reset()
    done = False

    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        q_table[state, action] += alpha * (
            reward + gamma * np.max(q_table[next_state]) * (not done)
            - q_table[state, action]
        )

        state = next_state

print("Training completed!")
env.close()

# Testing the trained agent with visual environment
print("Testing the trained agent...")

test_env = gym.make("Taxi-v4", render_mode="human")

state, _ = test_env.reset()
done = False
total_reward = 0
steps = 0

while not done and steps < 100:
    action = np.argmax(q_table[state])

    state, reward, terminated, truncated, _ = test_env.step(action)
    done = terminated or truncated

    total_reward += reward
    steps += 1

    print("Step:", steps, "Reward:", reward)
    time.sleep(0.3)

# Display results
print("\n--- Taxi Problem Results ---")
print("Total Steps:", steps)
print("Total Reward:", total_reward)

if terminated and reward == 20:
    print("Passenger successfully delivered!")
else:
    print("Episode ended without successful delivery.")

test_env.close()
