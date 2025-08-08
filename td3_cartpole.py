import gym
import torch
import torch.nn as nn
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

lr = 1e-3

noise_std = 0.2
noise_clip = 0.5
policy_delay = 2

batch_size = 64
buffer_size = 100000
warmup_steps = 0
max_episodes = 500
max_steps_per_episode = 200


class PolicyNetwork(nn.Module):
    """Actor network for continuous actions"""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, act_dim), 
            nn.Tanh()  # output in [-1, 1]
        )

    def forward(self, obs):
        return self.network(obs)


class QNetwork(nn.Module):
    """Critic network (Q-value) for continuous actions"""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, size=buffer_size):
        self.buffer = deque(maxlen=size)

    def push(self, obs, action, reward, next_obs, done):
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

    # Target actions with noise
    noise = (torch.randn_like(action) * noise_std).clamp(-noise_clip, noise_clip)
    target_action = (policy_target(next_obs) + noise).clamp(-1, 1)

    # Compute target Q-values
    with torch.no_grad():
        q1_target_value = q1_target(next_obs, target_action)
        q2_target_value = q2_target(next_obs, target_action)
        target_q = reward.unsqueeze(1) + gamma * (1 - done.unsqueeze(1)) * torch.min(q1_target_value, q2_target_value)

    # Update Q-networks
    q1_current = q1_network(obs, action)
    q2_current = q2_network(obs, action)
    
    q1_loss = nn.MSELoss()(q1_current, target_q)
    q2_loss = nn.MSELoss()(q2_current, target_q)
    critic_loss = q1_loss + q2_loss
    
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Delayed policy update
    if total_steps % policy_delay == 0:
        policy_loss = -q1_network(obs, policy_network(obs)).mean()
        
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # Soft update target networks
        soft_update_target_networks(policy_network, policy_target, tau)
        soft_update_target_networks(q1_network, q1_target, tau)
        soft_update_target_networks(q2_network, q2_target, tau)


# Initialize networks
policy_network = PolicyNetwork(obs_dim, act_dim).to(device)
policy_target = PolicyNetwork(obs_dim, act_dim).to(device)
policy_target.load_state_dict(policy_network.state_dict())

q1_network = QNetwork(obs_dim, act_dim).to(device)
q2_network = QNetwork(obs_dim, act_dim).to(device)
q1_target = QNetwork(obs_dim, act_dim).to(device)
q2_target = QNetwork(obs_dim, act_dim).to(device)

# Copy parameters to target networks
q1_target.load_state_dict(q1_network.state_dict())
q2_target.load_state_dict(q2_network.state_dict())

# Initialize optimizers
policy_optimizer = optim.Adam(policy_network.parameters(), lr=lr)
critic_optimizer = optim.Adam(list(q1_network.parameters()) + list(q2_network.parameters()), lr=lr)

# Initialize replay buffer
replay_buffer = ReplayBuffer()

# Training loop
total_steps = 0

for episode in range(max_episodes):
    obs = env.reset()[0]
    episode_reward = 0
    
    for step in range(max_steps_per_episode):
        total_steps += 1
        
        # Select action
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            continuous_action = policy_network(obs_tensor).cpu().numpy()[0]
        
        # Map continuous action to discrete environment action
        discrete_action = 0 if continuous_action < 0 else 1

        # Take action in environment
        next_obs, reward, done, _, _ = env.step(discrete_action)
        
        # Store transition
        replay_buffer.push(obs, continuous_action, reward, next_obs, float(done))
        
        episode_reward += reward
        obs = next_obs

        # Update networks
        if len(replay_buffer) >= batch_size:
            update_networks()

        if done:
            break

    print(f"Episode {episode + 1} | Reward: {episode_reward:.1f}")

env.close()