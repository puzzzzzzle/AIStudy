# RL算法

```mermaid
graph TD
RL[强化学习算法]
RL --> VB[值函数优化]
RL --> PB[策略梯度优化]
RL --> HY[混合架构]

VB --> QL[Q-Learning]
QL --> DQN[Deep Q-Network]
QL --> SARSA
QL --> DynaQ[Dyna-Q]
QL --> DoubleQ[Double Q-Learning]

PB --> RF[REINFORCE]
RF --> AC[Actor-Critic]
AC --> A3C
AC --> A2C
RF --> TRPO
TRPO --> PPO

HY --> AC_HY[Actor-Critic]
AC_HY --> DDPG
DDPG --> TD3
DDPG --> SAC

```

| 算法          | 策略类型   | 动作空间   | 核心思想                                                                 | 优点                                                                 | 缺点                                                                 |
|---------------|------------|------------|--------------------------------------------------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|
| **DQN**       | 离线策略   | 离散       | 使用深度网络近似Q值，经验回放+目标网络                                   | 数据利用率高，训练稳定                                               | 无法处理连续动作，Q值高估问题                                        |
| **REINFORCE** | 在线策略   | 离散/连续  | 蒙特卡洛策略梯度，直接优化策略                                           | 实现简单，无需值函数                                                 | 高方差，数据效率低                                                   |
| **Actor-Critic** | 混合     | 离散/连续  | Actor更新策略，Critic评估动作价值                                        | 方差比REINFORCE低                                                   | 需平衡Actor和Critic训练                                              |
| **TRPO**      | 在线策略   | 连续       | 通过KL散度约束策略更新（信任域）                                         | 理论保证单调改进                                                     | 计算复杂，实现难度高                                                 |
| **PPO**       | 在线策略   | 连续       | 裁剪策略更新比例，近似实现信任域                                         | 实现简单，性能稳定                                                   | 对超参数敏感                                                         |
| **DDPG**      | 离线策略   | 连续       | Actor-Critic框架+DQN扩展，确定性策略                                     | 适合高维连续控制                                                     | 超参数敏感，易陷入局部最优                                           |
| **SAC**       | 离线策略   | 连续       | 最大化策略熵，自动调节熵系数                                             | 探索能力强，样本效率高                                               | 实现较复杂                                                           |

## 基本分类概念

- 基于模型的强化学习（model-based reinforcement learning）
  - 不需要实际与环境交互采样数据
  - 直接利用已知的状态转移概率P(s'|s,a)和奖励函数R(s,a,s')进行计算
  - 没有行为策略与目标策略的区分，本质是数学上的迭代求解过程
- 无模型的强化学习（model-free reinforcement learning）
  - 行为策略（behavior policy）: 用来采样数据的策略
  - 目标策略（target policy）: 用数据来更新的策略
  - 在线策略（on-policy）: 行为策略和目标策略是同一个策略
  - 离线策略（off-policy）: 行为策略和目标策略不是同一个策略

## 动态规划

### 策略迭代

- 基于模型的强化学习
- 包含两个核心步骤：
    1. 策略评估（Policy Evaluation）: 迭代计算当前策略的状态价值函数
       $$V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$
    2. 策略改进（Policy Improvement）: 根据价值函数更新策略
       $$\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

## 价值迭代

- 基于模型的强化学习
- 直接寻找最优价值函数：
  $$V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^*(s')]$$

## 时序差分算法（temporal difference，TD）
- 蒙特卡洛方法价值更新方式:
  $$V(S_t) \leftarrow V(S_t) + \alpha[G_t - V(S_t)]$$
  （使用完整回报 $G_t = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{T-1}R_T$）

- 时序差分（temporal difference，TD）
  $$V(S_t) \leftarrow V(S_t) + \alpha[R_{t} + \gamma V(S_{t+1}) - V(S_t)]$$
  （误差(error) $\delta_t = R_{t} + \gamma V(S_{t+1}) - V(S_t)$）

- 关键区别：
    - 蒙特卡洛：必须等待episode结束才能更新
    - TD方法：可进行单步增量式更新
    - 统一形式：$V(S_t) \leftarrow V(S_t) + \alpha[\text{目标} - V(S_t)]$

### Sarsa算法

- 无模型的强化学习
- 在线策略（同轨策略）
- TD(0)更新规则：
  $$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha[r_t + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)]$$

### 多步sarsa算法

- 无模型的强化学习
- 在线策略
- n-step TD更新：
  $$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\left[\sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n Q(s_{t+n},a_{t+n}) - Q(s_t,a_t)\right]$$

### Q-Learning算法

- 无模型的强化学习
- 离线策略（异轨策略）
- 最优Q值更新：
  $$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha[r_t + \gamma \max_a Q(s_{t+1},a) - Q(s_t,a_t)]$$


## Dyna-Q算法
- 混合方法（模型基础+无模型）
- 基于Q-Learning的离线策略
  - 如果Q-planning次数为0, 就是Q-Learning
- 核心思想：结合实际经验与模拟经验
- 算法步骤：
    1. 真实环境交互更新Q值：
       $$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$
    2. 模型学习（记录转移关系）：
       $$\text{Model}(s,a) \rightarrow (s',r)$$
    3. 模拟采样更新（n次）：
       $$Q(s_k,a_k) \leftarrow Q(s_k,a_k) + \alpha[\hat{r}_k + \gamma \max_{a'} Q(\hat{s}'_k,a') - Q(s_k,a_k)]$$
- 关键优势：用模型模拟提升样本效率