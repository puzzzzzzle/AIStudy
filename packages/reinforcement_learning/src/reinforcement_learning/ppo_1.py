import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# 超参数
GAMMA = 0.99
LR = 2.5e-4
CLIP_EPS = 0.2
UPDATE_EPOCHS = 4
BATCH_SIZE = 2048
MINI_BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 网络结构
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, act_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

def select_action(model, obs):
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    logits, value = model(obs)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action), dist.entropy(), value

def compute_gae(rewards, values, dones, gamma=GAMMA, lam=0.95):
    values = values + [0]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        returns.insert(0, gae + values[step])
    return returns

def ppo_update(model, optimizer, obs, actions, log_probs, returns, advantages):
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    old_log_probs = torch.tensor(log_probs, dtype=torch.float32).to(device)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(UPDATE_EPOCHS):
        idx = np.arange(len(obs))
        np.random.shuffle(idx)
        for start in range(0, len(obs), MINI_BATCH_SIZE):
            end = start + MINI_BATCH_SIZE
            mb_idx = idx[start:end]
            logits, values = model(obs[mb_idx])
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions[mb_idx])
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_log_probs - old_log_probs[mb_idx])
            surr1 = ratio * advantages[mb_idx]
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages[mb_idx]
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(), returns[mb_idx])
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def main():
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    model = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    episode_rewards = []
    obs, _ = env.reset()
    for update in range(1000):
        obs_list, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        ep_reward = 0
        for _ in range(BATCH_SIZE):
            action, log_prob, entropy, value = select_action(model, obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            obs_list.append(obs)
            actions.append(action)
            log_probs.append(log_prob.item())
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())
            obs = next_obs
            ep_reward += reward
            if done:
                episode_rewards.append(ep_reward)
                obs, _ = env.reset()
                ep_reward = 0

        # 计算GAE和returns
        with torch.no_grad():
            _, next_value = model(torch.tensor(obs, dtype=torch.float32).to(device))
        values.append(next_value.item())
        returns = compute_gae(rewards, values, dones)
        advantages = np.array(returns) - np.array(values[:-1])

        # PPO更新
        ppo_update(model, optimizer, obs_list, actions, log_probs, returns, advantages)

        if (update + 1) % 10 == 0:
            mean_reward = np.mean(episode_rewards[-10:])
            print(f"Update {update+1}, Mean Reward: {mean_reward:.2f}")
            if mean_reward >= 475:
                print("Solved!")
                break

if __name__ == "__main__":
    main()