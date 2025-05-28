import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import qwop_gym
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
GAMMA = 0.99
LR = 3e-4
EPS_CLIP = 0.2
K_EPOCHS = 4
BATCH_SIZE = 64
TIMESTEPS_PER_BATCH = 2048

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make QWOP environment
env = gym.make("QWOP-v1", browser="/usr/bin/google-chrome", driver="/usr/local/bin/chromedriver", failure_cost=30, success_reward=100)
obs_dim = env.observation_space.shape[0]
is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
act_dim = env.action_space.n if is_discrete else env.action_space.shape[0]


# Neural Network for Policy and Value Function
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = 256
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
        )

        # Policy head
        if is_discrete:
            self.pi = nn.Sequential(
                nn.Linear(hidden, act_dim),
                nn.Softmax(dim=-1)
            )
        else:
            self.mu = nn.Linear(hidden, act_dim)
            self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Value head
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.pi(x) if is_discrete else self.mu(x), self.v(x)

    def act(self, obs):
        x = torch.tensor(obs, dtype=torch.float32).to(device)
        if is_discrete:
            probs = self.pi(self.shared(x))
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.item()  # Ensure scalar (int) for discrete
        else:
            mu = self.mu(self.shared(x))
            std = self.log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            return action.detach().cpu().numpy(), log_prob.item()  # Ensure array for continuous


    def evaluate(self, obs, act):
        x = self.shared(obs)
        if is_discrete:
            probs = self.pi(x)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(act)
            entropy = dist.entropy()
        else:
            mu = self.mu(x)
            std = self.log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            log_probs = dist.log_prob(act).sum(-1)
            entropy = dist.entropy().sum(-1)

        values = self.v(x).squeeze()
        return log_probs, values, entropy


# PPO Training Loop
def compute_returns(rewards, masks, values, gamma):
    returns = []
    R = values[-1]
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return torch.tensor(returns).to(device)


def ppo_train(model, optimizer):
    obs_buffer, act_buffer, logprob_buffer, rew_buffer, val_buffer, done_buffer = [], [], [], [], [], []

    total_reward = 0.0
    obs, _ = env.reset()
    for _ in range(TIMESTEPS_PER_BATCH):
        action, logprob = model.act(obs)
        val = model(torch.tensor(obs, dtype=torch.float32).to(device))[1].item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        # Adjust reward from QWOP environment
        torso_n_velx = obs[3]
        torso_velx = env.vel_x.denormalize(torso_n_velx)

        head_n_y = obs[6]
        head_y = env.pos_y.denormalize(head_n_y)
        target_y = -3.5
        y_reward = -abs(head_y - target_y)

        reward += torso_velx * 0.005

        total_reward += reward

        done = terminated or truncated

        obs_buffer.append(obs)
        act_buffer.append(action)
        logprob_buffer.append(logprob)
        rew_buffer.append(reward)
        val_buffer.append(val)
        done_buffer.append(0.0 if done else 1.0)

        obs = next_obs
        if done:
            obs, _ = env.reset()

    val_buffer.append(model(torch.tensor(obs, dtype=torch.float32).to(device))[1].item())
    returns = compute_returns(rew_buffer, done_buffer, val_buffer, GAMMA)
    obs_tensor = torch.tensor(np.array(obs_buffer), dtype=torch.float32).to(device)
    act_array = np.array(act_buffer)

    if is_discrete:
        act_tensor = torch.tensor(act_array, dtype=torch.long).to(device)
    else:
        act_tensor = torch.tensor(act_array, dtype=torch.float32).to(device)

    logprob_old = torch.tensor(np.array(logprob_buffer), dtype=torch.float32).to(device)
    val_tensor = torch.tensor(val_buffer[:-1], dtype=torch.float32).to(device)
    val_tensor = torch.tensor(val_buffer[:-1], dtype=torch.float32).to(device)
    adv_tensor = returns - val_tensor
    adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

    for _ in range(K_EPOCHS):
        indices = np.arange(TIMESTEPS_PER_BATCH)
        np.random.shuffle(indices)
        for i in range(0, TIMESTEPS_PER_BATCH, BATCH_SIZE):
            idx = indices[i:i + BATCH_SIZE]
            sampled_obs = obs_tensor[idx]
            sampled_act = act_tensor[idx]
            sampled_adv = adv_tensor[idx]
            sampled_ret = returns[idx]
            sampled_oldlog = logprob_old[idx]

            logprobs, values, entropy = model.evaluate(sampled_obs, sampled_act)
            ratio = torch.exp(logprobs - sampled_oldlog)
            surr1 = ratio * sampled_adv
            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * sampled_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((sampled_ret - values) ** 2).mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return total_reward


# Train
model = ActorCritic().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
reward_history = []


# Uncomment to load from a checkpoint. Comment out to start fresh.
model.load_state_dict(torch.load("ppo_qwop_torch.pth"))
model.eval()
print("Loaded model weights from checkpoint.")



for i in range(2000):  # ~10000 updates
    episode_reward = ppo_train(model, optimizer)
    reward_history.append(episode_reward)
    print(f"Update {i+36001} done. Episode reward: {episode_reward:.2f}")

    # Plot every 10 updates
    if (i + 1) % 10 == 0:
        plt.figure(figsize=(10, 5))
        plt.plot(reward_history, label="Episode Reward")
        plt.xlabel("Update")
        plt.ylabel("Reward")
        plt.title("Reward Progress Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("reward_progress.png")
        plt.close()


# Save model
torch.save(model.state_dict(), "ppo_qwop_torch.pth")
print("Model saved. Press Ctrl+C to exit.")
