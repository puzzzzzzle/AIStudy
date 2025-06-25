import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import ray

device = torch.device("mps") if torch.backends.mps.is_available() else \
    torch.device("cuda") if torch.cuda.is_available() else \
        torch.device("cpu")
print(f"device: {device}")


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def show_plt(return_list):
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO')
    plt.show()

    # mv_return = moving_average(return_list, 9)
    # plt.plot(episodes_list, mv_return)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('PPO')
    # plt.show()


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    ''' PPO算法,采用截断方式 '''

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    @staticmethod
    def compute_advantage(gamma, lmbda, td_delta):
        td_delta = td_delta.detach().cpu().numpy().flatten()  # 保证是一维
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        advantage_arr = np.array(advantage_list, dtype=np.float32)  # 先转成np.array
        return torch.from_numpy(advantage_arr)  # 返回torch tensor

    def take_action(self, state):
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.from_numpy(np.array(transition_dict['states'])).float().to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.from_numpy(np.array(transition_dict['rewards'])).float().view(-1, 1).to(self.device)
        next_states = torch.from_numpy(np.array(transition_dict['next_states'])).float().to(self.device)
        dones = torch.from_numpy(np.array(transition_dict['dones'])).float().view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = self.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


actor_lr = 1e-3
critic_lr = 1e-2

hidden_dim = 128

gamma = 0.98
lmbda = 0.95
eps = 0.2


def train(num_episodes=500,
          epochs=10,
          env_name='CartPole-v1', ):
    env = gym.make(env_name)
    env.reset(seed=0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)
    return_list = []
    for _ in tqdm(range(num_episodes)):
        episode_return = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        state, info = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
        agent.update(transition_dict)
    return return_list


@ray.remote
class Worker:
    def __init__(self, env_name, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma,
                 device, seed):
        import gymnasium as gym
        self.env = gym.make(env_name)
        self.env.reset(seed=seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
        self.device = device

    def set_weights(self, actor_weights, critic_weights):
        self.agent.actor.load_state_dict(actor_weights)
        self.agent.critic.load_state_dict(critic_weights)

    def sample(self):
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        state, info = self.env.reset()
        done = False
        episode_return = 0
        while not done:
            action = self.agent.take_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            episode_return += reward
        return transition_dict, episode_return
    def sample_n(self, n:int):
        transition_dicts = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        returns = []
        for _ in range(n):
            transition_dict, episode_return = self.sample()
            for k in transition_dicts:
                transition_dicts[k].extend(transition_dict[k])
            returns.append(episode_return)
        return transition_dicts, returns

def minibatch_generator(transitions, batch_size):
    N = len(transitions['states'])
    indices = np.arange(N)
    np.random.shuffle(indices)
    for start in range(0, N, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        batch = {k: np.array(v)[batch_idx] for k, v in transitions.items()}
        yield batch
def train_ray(num_workers=4,
              each_worker_samples=10,
              batch_size=256,
              num_episodes=500,
              epochs=10,
              env_name='CartPole-v1',
              ):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

    ray.init(ignore_reinit_error=True)
    workers = [
        Worker.remote(env_name, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma,
                      device, seed=i)
        for i in range(num_workers)
    ]

    return_list = []
    pbar = tqdm(total=num_episodes)
    for i_episode in range(num_episodes):
        actor_weights = agent.actor.state_dict()
        critic_weights = agent.critic.state_dict()
        set_weights_tasks = [w.set_weights.remote(actor_weights, critic_weights) for w in workers]
        ray.get(set_weights_tasks)

        sample_tasks = [w.sample_n.remote(each_worker_samples) for w in workers]
        results = ray.get(sample_tasks)

        all_transitions = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        episode_returns = []
        for transition_dict, episode_return in results:
            for k in all_transitions:
                all_transitions[k].extend(transition_dict[k])
            episode_returns.append(episode_return)

        for batch in minibatch_generator(all_transitions, batch_size):
            agent.update(batch)
        return_mean=float(np.mean(episode_returns))
        return_list.append(return_mean)
        pbar.set_postfix(
            {'episode': f'{i_episode}', 'return': f'{return_mean:.3f}'})
        pbar.update(1)


    ray.shutdown()
    return return_list


if __name__ == '__main__':
    # result_list = train()
    # show_plt(result_list)
    # print(result_list)
    # print(np.mean(result_list))
    # print(np.std(result_list))

    result_list = train_ray(num_workers=8,each_worker_samples=4,num_episodes=500,batch_size=4096) # 32*100 = 3200 episodes
    show_plt(result_list)
    print(result_list)
    print(np.mean(result_list))
    print(np.std(result_list))