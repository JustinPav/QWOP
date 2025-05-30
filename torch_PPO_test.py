import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import qwop_gym
import numpy as np
import time

GAMMA = 0.99
LR = 1e-4
EPS_CLIP = 0.2
K_EPOCHS = 6
BATCH_SIZE = 128
TIMESTEPS_PER_BATCH = 16384

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make QWOP environment
env = gym.make("QWOP-v1", browser="/usr/bin/google-chrome", driver="/usr/local/bin/chromedriver", stat_in_browser=True)
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
        else:
            mu = self.mu(self.shared(x))
            std = self.log_std.exp()
            dist = torch.distributions.Normal(mu, std)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1) if not is_discrete else dist.log_prob(action)
        return action.cpu().numpy(), log_prob.detach().cpu().numpy()

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

model = ActorCritic().to(device)
model.load_state_dict(torch.load("ppo_qwop_torch.pth"))
model.eval()


success = False
while not success:
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()
        torso_n_x = obs[0]
        torso_x = env.pos_x.denormalize(torso_n_x)
        time.sleep(0.01)
    if torso_x > 990 and done:
        success = True
        print("Success! Torso X:", torso_x)
        break
