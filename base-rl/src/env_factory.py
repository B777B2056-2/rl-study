import gymnasium as gym
from enum import Enum


class FrozenLakeShaping(gym.Wrapper):
    """到达 +1，掉洞 -1，每步 -0.01，靠近目标 +0.05"""

    GOAL = 15
    GRID = 4

    def __init__(self, env, step_penalty=0.01, distance_coef=0.05):
        super().__init__(env)
        self._step_penalty = step_penalty
        self._dist_coef = distance_coef

    def _dist(self, state):
        gr, gc = divmod(self.GOAL, self.GRID)
        sr, sc = divmod(state, self.GRID)
        return abs(gr - sr) + abs(gc - sc)

    def step(self, action):
        s = self.env.unwrapped.s
        next_state, _, terminated, truncated, info = self.env.step(action)

        if terminated and next_state == self.GOAL:
            reward = 1.0
        elif terminated:
            reward = -1.0
        else:
            old_d = self._dist(s)
            new_d = self._dist(next_state)
            reward = -self._step_penalty + self._dist_coef * (old_d - new_d)

        return next_state, reward, terminated, truncated, info

class EnvType(Enum):
    FrozenLake = "q_learning"


def _make_frozen_lake_env(is_slippery: bool):
    env = gym.make("FrozenLake-v1", is_slippery=False)
    env = FrozenLakeShaping(env)
    return env

_FACTORIES = {
    EnvType.FrozenLake: _make_frozen_lake_env,
}

def create_env(env_type: EnvType, is_slippery: bool):
    return _FACTORIES[env_type](is_slippery)