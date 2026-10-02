import torch
from base_agent import BaseAgent
import gymnasium as gym
import numpy as np
from config import DQNConfig
from copy import deepcopy


class QNetwork(torch.nn.Module):
    """
        Q网络结构：在线网络与目标网络共用结构，但目标网络内参数为在线网络的冻结参数（每隔一定step进行冻结）
        输入维度：n_states
        输出维度:n_actions
    """
    def __init__(self, n_states: int, n_hidden: int, n_actions: int):
        super(QNetwork, self).__init__()
        self._n_states = n_states
        self._n_actions = n_actions

        # 3层MLP
        self._fc1 = torch.nn.Linear(in_features=n_states, out_features=n_hidden)
        self._fc2 = torch.nn.Linear(in_features=n_hidden, out_features=n_hidden)
        self._fc3 = torch.nn.Linear(in_features=n_hidden, out_features=n_actions)

    def forward(self, x):
        x = x.view(-1, self._n_states)
        x = self._fc1(x)
        x = torch.relu(x)

        x = self._fc2(x)
        x = torch.relu(x)

        x = self._fc3(x)
        return x

class ReplayBuffer(object):
    """
        经验池：用于存放每一条探索过的路径组合，并提供随机选取的方法
        作用：打乱与时间高度相关的数据，使得Q网络数据独立同分布，以进行反向传播
        存储格式：(经验条数索引，当前状态s，当前动作a，当前奖励r，下一状态s`，完成标志done)
    """
    def __init__(self, cap: int, n_states: int, device: str):
        self._cap = cap
        self._device = device
        # 状态为向量，动作、奖励、完成标志为标量（即维度为1）；
        # 统一存CPU，采样时搬运到指定设备，避免每次存都做一次 CPU→GPU 拷贝
        self._s = torch.zeros((cap, n_states), dtype=torch.float32, device='cpu')
        self._a = torch.zeros((cap,), dtype=torch.long, device='cpu')
        self._r = torch.zeros((cap,), dtype=torch.float32, device='cpu')
        self._s_next = torch.zeros((cap, n_states), dtype=torch.float32, device='cpu')
        self._done = torch.zeros((cap,), dtype=torch.float32, device='cpu')
        # 位置指针
        self._idx = 0
        # 有实际数据的个数
        self._size = 0

    def append_one(self, s, a, r, s_next, done) -> None:
        """
            添加一条经验
        """
        self._s[self._idx] = torch.from_numpy(s)
        self._a[self._idx] = torch.tensor(a, dtype=torch.long, device='cpu')
        self._r[self._idx] = torch.tensor(r, dtype=torch.float32, device='cpu')
        self._s_next[self._idx] = torch.from_numpy(s_next)
        self._done[self._idx] = torch.tensor(done, dtype=torch.float32, device='cpu')

        # 索引自增
        self._idx += 1

        # 超过容量，走环形逻辑，清开头，再从开头走
        # 未超过容量，则增加一次大小；超过容量时，大小等于cap
        self._idx = (self._idx + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def random_sample(self, batch_size: int):
        """
            随机采样
        """
        # 当前有数据的个数少于batch_size，则无法进行采样
        if self._size < batch_size:
            return None

        # 随机挑选一个返回
        idx = np.random.randint(0, self._size, size=batch_size)
        return (
            torch.as_tensor(self._s[idx], dtype=torch.float32, device=self._device),      # (B, n_states)
            torch.as_tensor(self._a[idx], dtype=torch.long, device=self._device),         # (B,)
            torch.as_tensor(self._r[idx], dtype=torch.float32, device=self._device),      # (B,)
            torch.as_tensor(self._s_next[idx], dtype=torch.float32, device=self._device), # (B, n_states)
            torch.as_tensor(self._done[idx], dtype=torch.float32, device=self._device),   # (B,)
        )

class DQN(BaseAgent):
    """
        DQN算法实现
    """
    def __init__(self, config: DQNConfig, env: gym.Env):
        super(DQN, self).__init__()
        self._config = config
        self._env = env
        self._epsilon = deepcopy(config.epsilon)
        self._total_step_cnt = 0    # 全局步数以保证目标网络的稳定性，从而保证训练的稳定性

        # 在线网络
        self._online_network = QNetwork(
            n_states=config.n_states,
            n_hidden=config.n_hidden_q_net,
            n_actions=config.n_actions,
        ).to(config.device)

        # 目标网络
        self._target_network = QNetwork(
            n_states=config.n_states,
            n_hidden=config.n_hidden_q_net,
            n_actions=config.n_actions,
        ).to(config.device)

        # 经验池
        self._replay_buffer = ReplayBuffer(
            cap=config.replay_buffer_cap,
            n_states=config.n_states,
            device=config.device
        )

        # 初始化网络参数
        self._init_q_network_weights()

        # 梯度下降优化器
        self._optimizer = torch.optim.Adam(self._online_network.parameters(), lr=config.learning_rate)

    def _copy_weights_to_target_net(self):
        """
            拷贝在线网络参数到目标网络
        """
        self._target_network.load_state_dict(self._online_network.state_dict())

    def _init_q_network_weights(self):
        """
            初始化在线网络、目标网络参数
        """
        # 1. 初始化在线网络参数；无需实际编码，pytorch会默认使用均匀分布进行初始化
        # 2. 拷贝在线网络参数到目标网络
        self._copy_weights_to_target_net()

    def update_schedule(self, n_episode: int):
        """探索率衰减（指数衰减）"""
        cur_epsilon = self._config.epsilon * (self._config.epsilon_decay ** n_episode)
        self._epsilon = max(self._config.min_epsilon, cur_epsilon)

    def _epsilon_greedy(self, state, epsilon: float):
        """epsilon-贪婪随机探索：评估时纯贪心；训练时 ε-贪婪"""
        # 探索：随机数小于epsilon，则对动作空间进行一次采样，返回随机的action
        if not self.is_evaluating and np.random.rand() < epsilon:
            return self._env.action_space.sample()
        # 利用：随机数大于等于epsilon，则选取最优动作（即通过Q网络计算得出，然后取max）
        with torch.no_grad():
            s_tensor = torch.as_tensor(state, dtype=torch.float32, device=self._config.device)
            q = self._online_network(s_tensor)
            return int(q.argmax(dim=1).item())

    def _update_q_net(self):
        """更新一次Q网络（反向传播）"""
        # 1. 采样单个batch
        batch = self._replay_buffer.random_sample(self._config.batch_size)
        if batch is None:   # 数据不够，暂时不进行学习
            return None
        s, a, r, s_next, done = batch

        # 2. 算当前 Q 值
        q_values = self._online_network(s)  # (B, n_actions)
        # 取出每个状态下，实际执行的动作 a 对应的 Q 值
        q = q_values.gather(1, a.unsqueeze(1)).squeeze(1)

        # 3. 用目标网络，计算算目标值（即监督学习的Label）
        with torch.no_grad():
            # 下一状态的 Q 值
            target_q_values = self._target_network(s_next)
            # 取下一状态的最大 Q 值
            target_q_max = target_q_values.max(dim=1)[0]
            # 终止状态只保留reward，而done取值为0（false）或1（true），因此可以用减法而不用if直接表达
            device = self._config.device
            target_q = r.to(device) + self._config.gamma * target_q_max * (1 - done.to(device))

        # 4. 计算Loss
        loss = torch.nn.MSELoss()(q, target_q)

        # 5. 梯度下降
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # 6. 把在线网络参数定期同步目标网络
        self._total_step_cnt = self._total_step_cnt + 1
        if self._total_step_cnt % self._config.target_update == 0:
            self._copy_weights_to_target_net()

        return loss.item()

    def run_episode(self) -> dict:
        """运行一个回合"""
        # 重置环境
        state, _ = self._env.reset()
        total_reward = 0
        total_loss = 0.0
        loss_cnt = 0
        done = False

        while not done:
            # 1. 决策最优动作
            action = self._epsilon_greedy(state=state, epsilon=self._epsilon)
            # 2. 执行最优动作，反馈给环境
            next_state, reward, terminated, truncated, _ = self._env.step(action)
            # 3. 设置此时结束标志：状态实际到达终态 or 时间步达到最大限制
            done = terminated or truncated 

            # 4. 存一次经验
            self._replay_buffer.append_one(state, action, reward, next_state, done)

            # 5. 非评估模式时，更新一次Q网络
            if not self.is_evaluating:
                loss_val = self._update_q_net()
                if loss_val is not None:
                    total_loss += loss_val
                    loss_cnt += 1
            
            # 6. 更新当前state
            state = next_state
            # 7. 累计奖励，用于可视化绘图
            total_reward += reward

        return {
            "reward": total_reward,
            "loss": total_loss / loss_cnt if loss_cnt > 0 else 0.0,
            "success": 1.0 if total_reward >= 500 else 0.0,
            "epsilon": self._epsilon,
        }
    