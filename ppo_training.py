from stable_baselines3 import PPO
import gymnasium as gym
import qwop_gym

# Create vectorized environment
env = gym.make("QWOP-v1", browser="/usr/bin/google-chrome", driver="/usr/local/bin/chromedriver")

# Create PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save the model
model.save("ppo_qwop")

# Test the model
obs, _ = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
