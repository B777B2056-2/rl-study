from enum import Enum
from config import QLearningConfig
from algorithms.qlearning import QLearning
from algorithms.sarsa import Sarsa


class AgentType(Enum):
    Q_LEARNING = "q_learning"
    SARSA = "sarsa"


def _make_q_learning_agent(env) -> QLearning:
    config = QLearningConfig(
        n_epoch=5000,
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
        n_epoch=5000,
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=0.1,
        epsilon=0.99,
        epsilon_decay=0.995,
        min_epsilon=0.05,
        gamma=0.97
    )

    return Sarsa(env=env, config=config)

_FACTORIES = {
    AgentType.Q_LEARNING: _make_q_learning_agent,
    AgentType.SARSA: _make_sarsa_agent,
}


def create_agent(agent_type: AgentType, env):
    return _FACTORIES[agent_type](env)
