import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Shared policy network
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        logits = self.model(x)
        return Categorical(logits=logits)

def collect_trajectory(env, policy):
    states, actions, log_probs, rewards = [], [], [], []
    state = env.reset()
    done = False
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        dist = policy(state_tensor)
        action = dist.sample()
        next_state, reward, done, _ = env.step(action.item())
        
        states.append(state_tensor)
        actions.append(action)
        log_probs.append(dist.log_prob(action))
        rewards.append(reward)
        state = next_state

    # Compute return-to-go
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + G  # no discount
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    
    return states, actions, log_probs, returns


# ------------------------------------------------------------------

def ppo_loss(policy, states, actions, old_log_probs, returns, clip_eps=0.2):
    new_log_probs = torch.stack([policy(s).log_prob(a) for s, a in zip(states, actions)])
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    loss = -torch.min(ratio * returns, clipped_ratio * returns).mean()
    return loss


env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

policy = PolicyNetwork(obs_dim, n_actions)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)


for epoch in range(100):
    states, actions, log_probs, returns = collect_trajectory(env, policy)
    old_log_probs = torch.stack(log_probs).detach()

    for _ in range(5):  # multiple epochs per batch
        loss = ppo_loss(policy, states, actions, old_log_probs, returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"[PPO] Epoch {epoch}, Total Reward: {sum(returns):.1f}")
