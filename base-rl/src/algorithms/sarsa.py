import gymnasium as gym
import numpy as np
from config import QLearningConfig
from copy import deepcopy
from base_agent import BaseAgent

class Sarsa(BaseAgent):
    def __init__(self, config: QLearningConfig, env: gym.Env):
        super(Sarsa, self).__init__()
        self._config = config
        self._env = env
        self._epsilon = deepcopy(self._config.epsilon)
        self._q_table = np.zeros((self._config.n_states, self._config.n_actions))

    # epsilon-贪婪随机探索
    def _epsilon_greedy(self, state, epsilon: float):
        # 探索：随机数小于epsilon，则对动作空间进行一次采样，返回随机的action
        if np.random.rand() < epsilon:
            return self._env.action_space.sample()
        # 利用：随机数大于等于epsilon，则选取最优动作（即Q表里当前状态的最大奖励对应的动作）
        return np.argmax(self._q_table[state])

    def update_schedule(self, n_episode: int):
        """探索率衰减（指数衰减）"""
        cur_epsilon = self._config.epsilon * (self._config.epsilon_decay ** n_episode)
        self._epsilon = max(self._config.min_epsilon, cur_epsilon)

    def act(self, state):
        """评估时纯贪心；训练时 ε-贪婪"""
        # 评估模式：不允许探索
        if self.is_evaluating:
            return np.argmax(self._q_table[state])

        # 探索：随机数小于epsilon，则对动作空间进行一次采样，返回随机的action
        if np.random.rand() < self._epsilon:
            return self._env.action_space.sample()
        # 利用：随机数大于等于epsilon，则选取最优动作（即Q表里当前状态的最大奖励对应的动作）
        return np.argmax(self._q_table[state])

    def run_episode(self) -> dict:
        """更新Q表"""
        # 重置环境
        state, _ = self._env.reset()
        total_reward = 0    
        done = False

        while not done:
            # 1. 决策最优动作
            action = self._epsilon_greedy(state=state, epsilon=self._epsilon)
            # 2. 执行最优动作，反馈给环境
            next_state, reward, terminated, truncated, _ = self._env.step(action)
            # 3. 设置此时结束标志：状态实际到达终态 or 时间步达到最大限制
            done = terminated or truncated 
            # 4. 估计当前Q值
            old_q = self._q_table[state, action]

            if terminated:
                target = reward
            else:
                '''
                    与Q-Learning的区别：
                    1. Q-Learning是选取np.max(self._q_table[next_state])，即取具备最优奖励的下一动作，SARSA是选取实际下一动作；
                    2. Q-Learning为Off-Policy（离线策略），其目标策略是"贪婪"（即此处的np.max），行为策略是"ε-贪婪"；
                    3. SARSA是On-Policy（在线策略），其目标策略与行为策略都是 ε-贪婪（需要根据下一状态，通过ε-贪婪获取下一动作）
                '''
                next_action = self._epsilon_greedy(state=next_state, epsilon=self._epsilon)
                target = reward + self._config.gamma * self._q_table[next_state, next_action]

            new_q = old_q + self._config.alpha * (target - old_q)
            self._q_table[state, action] = new_q
            # 5. 更新当前state
            state = next_state
            # 6. 累计奖励，用于可视化绘图
            total_reward += reward

        return {
            "reward": total_reward,
            "success": 1.0 if total_reward > 0 else 0.0,
            "epsilon": self._epsilon,
        }
