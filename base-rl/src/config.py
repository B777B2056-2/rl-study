from dataclasses import dataclass

@dataclass
class QLearningConfig:
    n_states: int   # 离散状态的状态个数
    n_actions: int
    alpha: float
    epsilon: float
    epsilon_decay: float
    min_epsilon: float
    gamma: float

@dataclass
class DQNConfig:
    replay_buffer_cap: int  # 经验池大小
    n_states: int   # 离散状态的状态个数，连续状态的状态空间元信息个数
    n_actions: int
    learning_rate: float
    epsilon: float
    epsilon_decay: float
    min_epsilon: float
    gamma: float
    n_hidden_q_net: int # q网络隐含层维度
    batch_size : int
    target_update: int  # 单个回合中，每隔target_update步，触发一次在线网络参数向目标网络同步
    device: str

@dataclass
class PPOConfig:
    batch_size : int
    n_states: int   # 离散状态的状态个数，连续状态的状态空间元信息个数
    n_actions: int
    learning_rate: float
    n_traces: int   # 单次迭代中，一个batch的数据轨迹数量
    n_hidden_actor: int
    n_hidden_critic: int
    n_epoch: int # 策略模型、价值模型训练轮次
    gamma: float
    device: str
    lambda_val: float = 0.95
    clip_eps: float = 0.2
