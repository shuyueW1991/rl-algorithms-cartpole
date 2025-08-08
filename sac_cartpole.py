import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment setup (fake continuous CartPole)
env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
act_dim = 1  # fake continuous action in [-1, 1]

# Hyperparameters
gamma = 0.99
tau = 0.005
lr = 3e-4

alpha = 0.2  # entropy regularization coefficient

batch_size = 64
buffer_size = 100000
warmup_steps = 1000
max_episodes = 500
max_steps_per_episode = 500



class PolicyNetwork(nn.Module):
    """Policy network outputting mean and log_std for continuous actions"""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU()
        )
        self.mean_head = nn.Linear(128, act_dim)
        self.log_std_head = nn.Linear(128, act_dim)
        
        # Constrain log_std to reasonable range
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, obs):
        features = self.network(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        action = torch.tanh(x_t)  # bound to [-1, 1]
        
        # Compute log probability with change of variables formula
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)  # tanh jacobian
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class QNetwork(nn.Module):
    """Q-network for continuous actions"""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, size=buffer_size):
        self.buffer = deque(maxlen=size)

    def push(self, obs, action, reward, next_obs, done):
        # Convert tensors to numpy for storage
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = map(np.stack, zip(*batch))
        return (
            torch.tensor(obs, dtype=torch.float32, device=device),
            torch.tensor(action, dtype=torch.float32, device=device),
            torch.tensor(reward, dtype=torch.float32, device=device),
            torch.tensor(next_obs, dtype=torch.float32, device=device),
            torch.tensor(done, dtype=torch.float32, device=device)
        )

    def __len__(self):
        return len(self.buffer)


def soft_update_target_networks(network, target_network, tau):
    """Soft update target network parameters"""
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def update_networks():
    """Update Q-networks and policy network"""
    obs, action, reward, next_obs, done = replay_buffer.sample(batch_size)

    # Compute target Q-values
    with torch.no_grad():
        next_action, next_log_prob = policy_network.sample(next_obs)
        q1_next = q1_target(next_obs, next_action)
        q2_next = q2_target(next_obs, next_action)
        q_next = torch.min(q1_next, q2_next)
        target_q = reward.unsqueeze(1) + (1 - done.unsqueeze(1)) * gamma * (q_next - alpha * next_log_prob)

    # Update Q-networks
    q1_current = q1_network(obs, action)
    q2_current = q2_network(obs, action)
    
    q1_loss = F.mse_loss(q1_current, target_q)
    q2_loss = F.mse_loss(q2_current, target_q)
    
    q1_optimizer.zero_grad()
    q1_loss.backward()
    q1_optimizer.step()
    
    q2_optimizer.zero_grad()
    q2_loss.backward()
    q2_optimizer.step()

    # Update policy network
    current_action, current_log_prob = policy_network.sample(obs)
    q1_policy = q1_network(obs, current_action)
    q2_policy = q2_network(obs, current_action)
    min_q = torch.min(q1_policy, q2_policy)
    
    policy_loss = (alpha * current_log_prob - min_q).mean()
    
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    # Soft update target networks
    soft_update_target_networks(q1_network, q1_target, tau)
    soft_update_target_networks(q2_network, q2_target, tau)


# Initialize networks
policy_network = PolicyNetwork(obs_dim, act_dim).to(device)

q1_network = QNetwork(obs_dim, act_dim).to(device)
q2_network = QNetwork(obs_dim, act_dim).to(device)
q1_target = QNetwork(obs_dim, act_dim).to(device)
q2_target = QNetwork(obs_dim, act_dim).to(device)

# Copy parameters to target networks
q1_target.load_state_dict(q1_network.state_dict())
q2_target.load_state_dict(q2_network.state_dict())

# Initialize optimizers
policy_optimizer = optim.Adam(policy_network.parameters(), lr=lr)
q1_optimizer = optim.Adam(q1_network.parameters(), lr=lr)
q2_optimizer = optim.Adam(q2_network.parameters(), lr=lr)

# Initialize replay buffer
replay_buffer = ReplayBuffer()

# Training loop
total_steps = 0

for episode in range(max_episodes):
    obs, _ = env.reset()
    episode_reward = 0
    
    for step in range(max_steps_per_episode):
        # Select action
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            continuous_action, _ = policy_network.sample(obs_tensor)
            continuous_action = continuous_action.cpu().numpy()[0]
        
        # Map continuous action to discrete environment action
        discrete_action = 0 if continuous_action < 0 else 1
        
        # Take action in environment
        next_obs, reward, done, _, _ = env.step(discrete_action)
        
        # Store transition
        replay_buffer.push(obs, continuous_action, reward, next_obs, float(done))
        
        obs = next_obs
        episode_reward += reward
        total_steps += 1

        # Update networks
        if total_steps > warmup_steps and len(replay_buffer) >= batch_size:
            update_networks()

        if done:
            break

    print(f"Episode {episode + 1} | Reward: {episode_reward:.1f}")

env.close()