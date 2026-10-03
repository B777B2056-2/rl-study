import torch
from base_agent import BaseAgent
import gymnasium as gym
import numpy as np
from config import PPOConfig
from torch.utils.data import TensorDataset, DataLoader

class Actor(torch.nn.Module):
    """
        策略模型：输出动作概率，选动作
        输入：当前状态
        输出：各状态对应选取的动作概率（一个状态对应多个动作，因此输出的维度为(batch_size, n_states, n_actions)）
    """
    def __init__(self, n_states: int, n_actions: int, n_hidden: int):
        super(Actor, self).__init__()
        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(n_states, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_actions),
        )

    def forward(self, x, is_evaluating: bool = False, action = None):
        # 获取决策出来的动作原始概率分布
        a_logits = self._mlp(x)
        # 归一化概率分布，内部会自动 softmax：
        #   动作 0 概率 = exp(2.0) / (exp(2.0) + exp(0.5)) ≈ 0.82
        #   动作 1 概率 = exp(0.5) / (exp(2.0) + exp(0.5)) ≈ 0.18
        dist = torch.distributions.Categorical(logits=a_logits)
        # 采样获取动作
        if is_evaluating:
            action = dist.probs.argmax(dim=1)   # 评估：贪心
        elif action is None:
            # 按概率随机采样（仅收集trace时使用）：动作 0 被选中的概率是 0.82，动作 1 是 0.18
            action = dist.sample()
        # 算采样动作的对数概率
        log_prob = dist.log_prob(action)
        # 熵
        entropy = dist.entropy()
        return action, log_prob, entropy

class Critic(torch.nn.Module):
    """
        价值模型：估计状态价值，算优势
        输入：当前状态
        输出：各状态对应的状态价值（一个状态对应一个价值，因此输出的维度为(batch_size, n_states, 1)）
    """
    def __init__(self, n_states: int, n_hidden: int):
        super(Critic, self).__init__()
        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(n_states, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, 1),
        )
    
    def forward(self, x):
        return self._mlp(x)

class PPO(BaseAgent):
    """
        Actor（策略网络）：输入状态，输出动作概率，负责选动作。
        Critic（价值网络）：输入状态，输出状态价值 V(s)，负责评估状态好坏。

        状态 s
            ├→ Actor → 动作分布 π(a|s) → 采样动作 a
            └→ Critic → V(s)
                            ↓
                    算优势 A(s,a)
                            ↓
                Actor 和 Critic 更新
    """
    def __init__(self, config: PPOConfig, env: gym.Env):
        super(PPO, self).__init__()
        self._config = config
        self._env = env

        self._actor = Actor(
            n_states=config.n_states, 
            n_actions=config.n_actions, 
            n_hidden=config.n_hidden_actor,
        ).to(config.device)

        self._critic = Critic(
            n_states=config.n_states, 
            n_hidden=config.n_hidden_critic,
        ).to(config.device)

        self._optimizer = torch.optim.Adam(
            list(self._actor.parameters()) + list(self._critic.parameters()), 
            lr=config.learning_rate,
        )

    def _collect_traces(self):
        """
            单次迭代中采样数据轨迹
            方法：用当前策略和环境交互，收集的一批 (s, a, r, s', done, log_prob, V) 数据；不进行梯度更新
        """
        traces = []  # 轨迹集合
        rewards = [] # 奖励集合
        state, _ = self._env.reset()    # 获取一个状态

        ep_reward = 0.0 # 单轮奖励
        while len(traces) < self._config.n_traces:
            with torch.no_grad():
                # 加上batch维度
                s = torch.as_tensor(state, dtype=torch.float32, device=self._config.device).unsqueeze(0)   # (1, n_states)
                # 输入策略模型，获取决策出来的动作（按概率随机采样）、采样动作的对数概率
                if self.is_evaluating:
                    # 评估：纯贪心
                    action, log_prob, _ = self._actor(s, True)
                else:
                    # 训练：需要采样
                    action, log_prob, _ = self._actor(s)
                
                # 输入价值模型，计算该状态对应的价值
                value = self._critic(s)

            # 执行动作
            next_state, reward, terminated, truncated, _ = self._env.step(action.item())
            done = terminated or truncated

            # 记录trace（去掉 batch 维，存成标量）
            traces.append({
                "state": torch.as_tensor(state, dtype=torch.float32),
                "action": action.squeeze(0).cpu(),
                "reward": reward,
                "done": float(terminated), 
                "log_prob": log_prob.squeeze(0).cpu(),
                "value": value.squeeze(0).cpu(),
            })

            # 变为下一状态
            state = next_state
            ep_reward += reward # 累计单回合奖励

            # 回合结束，重置环境
            if done:
                rewards.append(ep_reward)
                ep_reward = 0.0
                state, _ = self._env.reset()

        return traces, rewards

    def _compute_gae(self, traces):
        """
            广义优势估计：用多步 TD 误差（时序差分误差）的加权和来估计优势，平衡偏差和方差。
            优势 > 0：这个动作比平均好 → 提高它的概率
            优势 < 0：这个动作比平均差 → 降低它的概率
        """
        # 1. 获取奖励、价值
        rewards = [r["reward"] for r in traces]
        values = [r["value"].item() for r in traces] + [0.0]    # 末尾补0，因为公式是t -> t+1，方便计算
        dones = [r["done"] for r in traces]

        # 2. 反向遍历：因t时刻的GAE值依赖t+1时刻的GAE值，因此只能反向遍历
        advantages = [0.0 for _ in range(len(traces) + 1)]
        for t in reversed(range(len(traces))):
            # 1. 计算当前t时刻TD误差（时序差分误差）
            td_error = rewards[t] + self._config.gamma * values[t+1] * (1 - dones[t]) - values[t]
            # 2. 计算计算当前t时刻的优势估计值
            advantages[t] = td_error + self._config.gamma * self._config.lambda_val * (1 - dones[t]) * advantages[t+1]

        # 3. 计算回报：t时刻的回报 = t时刻的优势 + t时刻的价值
        advantages = torch.tensor(advantages[:-1], dtype=torch.float32)   # (N,)
        values_t = torch.tensor(values[:-1], dtype=torch.float32)    # (N,)
        returns = advantages + values_t                              # (N,)
        return advantages, returns

    def _clip(self, logp_ratio):
        """裁剪：限制概率比的范围，每次策略网络训练只走一小步，保持训练稳定"""
        return torch.clamp(logp_ratio, 1 - self._config.clip_eps, 1 + self._config.clip_eps)

    def _update_batch(self, traces, advantages: torch.Tensor, returns: torch.Tensor):
        """
            迭代训练一个batch
        """
        # 1. 重整数据为tensor
        states = torch.stack([r["state"] for r in traces])           # (N, n_states)
        actions = torch.stack([r["action"] for r in traces])         # (N,)
        old_log_probs = torch.stack([r["log_prob"] for r in traces]) # (N,)

        # 2. 数据准备
        dataset = TensorDataset(states, actions, old_log_probs, advantages, returns)
        dataloader = DataLoader(
            dataset,
            batch_size=self._config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # 2. batch迭代
        total_loss, loss_cnt = 0.0, 0
        for epoch in range(self._config.n_epoch):
            for s, a, old_logp, adv, ret in dataloader:
                s = s.to(self._config.device)
                a = a.to(self._config.device)
                old_logp = old_logp.to(self._config.device)
                adv = adv.to(self._config.device)
                ret = ret.to(self._config.device)

                # 策略网络前向传播，获取新动作、新动作对应的新概率分布、熵
                _, new_logp, entropy = self._actor(x=s, action=a)
                # 计算概率比：新概率 / 旧概率，使用exp保证数值稳定性，避免极小概率相除导致溢出
                ratio = torch.exp(new_logp - old_logp)
                # 策略损失：让"好动作"的概率变大，让"坏动作"的概率变小
                policy_loss = -torch.min(ratio * adv,  self._clip(ratio) * adv).mean()
                # 价值损失：价值网络估计的 V(s) 和实际回报 R 的均方误差。
                value = self._critic(s).squeeze(1)
                value_loss = torch.nn.functional.mse_loss(value, ret)
                # 熵平均值：衡量策略的随机性，越大越随机；鼓励探索，防止策略过早确定
                mean_entropy = entropy.mean()
                # 总损失 = 策略损失 + 价值损失 + 熵平均值
                loss = policy_loss + 0.5 * value_loss - 0.01 * mean_entropy
                total_loss += loss.item()
                loss_cnt += 1

                # 梯度更新
                self._optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪：限制梯度范数不超过 0.5，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self._actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self._critic.parameters(), 0.5)
                self._optimizer.step()

        return total_loss / loss_cnt

    def update_schedule(self, n_episode: int):
        pass

    def run_episode(self) -> dict:
        """运行一个回合"""
        # 1. 采样轨迹
        traces, rewards = self._collect_traces()

        # 2. 计算优势（GAE）
        advantages, returns = self._compute_gae(traces)

        # 3. 优势归一化：减少梯度方差
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 4. 多轮训练
        if not self.is_evaluating:
            loss = self._update_batch(traces, advantages, returns)
        else:
            loss = 0.0  # 评估模式下，跳过网络参数更新

        return {
            "reward": float(np.mean(rewards)) if rewards else 0.0,
            "loss": loss,
            "success": 1.0 if (rewards and np.mean(rewards) >= 500) else 0.0,
        }
