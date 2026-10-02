from enum import Enum
from config import QLearningConfig, DQNConfig
from algorithms.qlearning import QLearning
from algorithms.sarsa import Sarsa
from algorithms.dqn import DQN


class AgentType(Enum):
    Q_LEARNING = "q_learning"
    SARSA = "sarsa"
    DQN = "dqn"


def _make_q_learning_agent(env) -> QLearning:
    config = QLearningConfig(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=0.1,
        epsilon=0.99,
        epsilon_decay=0.995,
        min_epsilon=0.05,
        gamma=0.97
    )

    return QLearning(env=env, config=config)

def _make_sarsa_agent(env) -> QLearning:
    config = QLearningConfig(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=0.1,
        epsilon=0.99,
        epsilon_decay=0.995,
        min_epsilon=0.05,
        gamma=0.97
    )

    return Sarsa(env=env, config=config)

def _make_dqn_agent(env) -> QLearning:
    config = DQNConfig(
        replay_buffer_cap=10000,
        n_states=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        learning_rate=1e-3,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.05,
        gamma=0.99,
        n_hidden_q_net=128,
        batch_size=64,
        target_update=100,
        device="cpu",
    )

    return DQN(env=env, config=config)

_FACTORIES = {
    AgentType.Q_LEARNING: _make_q_learning_agent,
    AgentType.SARSA: _make_sarsa_agent,
    AgentType.DQN: _make_dqn_agent,
}


def create_agent(agent_type: AgentType, env):
    return _FACTORIES[agent_type](env)
