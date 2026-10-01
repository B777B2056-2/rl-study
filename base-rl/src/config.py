from dataclasses import dataclass

@dataclass
class QLearningConfig:
    n_epoch: int
    n_states: int
    n_actions: int
    alpha: float
    epsilon: float
    epsilon_decay: float
    min_epsilon: float
    gamma: float