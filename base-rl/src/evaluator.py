import numpy as np


class EvalModeMixin:
    """给算法加评估模式，仿 torch.no_grad()"""

    def _init_eval_mode(self):
        self._eval_depth = 0

    @property
    def is_evaluating(self) -> bool:
        return getattr(self, "_eval_depth", 0) > 0

    def eval_mode(self):
        return _EvalContext(self)


class _EvalContext:
    def __init__(self, agent):
        self._agent = agent

    def __enter__(self):
        self._agent._eval_depth += 1
        return self._agent

    def __exit__(self, *args):
        self._agent._eval_depth -= 1
        return False


class Evaluator:
    def __init__(self, env, n_episodes=100, seed=None, success_fn=None):
        self._env = env
        self._n_episodes = n_episodes
        self._seed = seed
        self._success_fn = success_fn

    def evaluate(self, agent) -> dict:
        with agent.eval_mode():
            return self._run(agent)

    def _run(self, agent) -> dict:
        rewards, lengths, successes = [], [], []

        for i in range(self._n_episodes):
            if self._seed is not None:
                state, info = self._env.reset(seed=self._seed + i)
            else:
                state, info = self._env.reset()

            done = False
            ep_reward = 0.0
            ep_length = 0

            while not done:
                action = agent.act(state)
                state, reward, terminated, truncated, info = self._env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_length += 1

            rewards.append(ep_reward)
            lengths.append(ep_length)
            if self._success_fn is not None:
                successes.append(1.0 if self._success_fn(ep_reward, ep_length, info) else 0.0)
            else:
                successes.append(1.0 if ep_reward > 0 else 0.0)

        r = np.array(rewards)
        l = np.array(lengths)
        s = np.array(successes)
        return {
            "mean_reward": float(r.mean()),
            "std_reward": float(r.std()),
            "min_reward": float(r.min()),
            "max_reward": float(r.max()),
            "mean_length": float(l.mean()),
            "success_rate": float(s.mean()),
        }