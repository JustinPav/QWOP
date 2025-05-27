from stable_baselines3 import PPO
import gymnasium as gym
import qwop_gym
import os

# Create vectorized environment
env = gym.make("QWOP-v1", browser="/usr/bin/google-chrome", driver="/usr/local/bin/chromedriver")

# Check if the model file exists
model_path = "ppo_qwop"
if not os.path.exists(model_path + ".zip"):
    raise FileNotFoundError(f"Model file '{model_path}.zip' not found. Train the model first.")

# Load the PPO model
model = PPO.load(model_path)

# Test the model
obs, _ = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
