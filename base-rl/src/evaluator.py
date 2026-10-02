from plotter import Plotter


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
    def __init__(self, env, n_episodes=100, seed=None, plotter=None):
        self._env = env
        self._n_episodes = n_episodes
        self._seed = seed
        self._plotter = plotter or Plotter(window=0)

    def evaluate(self, agent, save_path=None, show=True):
        with agent.eval_mode():
            rewards, successes = [], []

            for ep in range(self._n_episodes):
                seed = self._seed + ep if self._seed is not None else None
                self._env.reset(seed=seed)
                self._env.action_space.seed(seed)

                metrics = agent.run_episode()
                rewards.append(metrics["reward"])
                successes.append(metrics["success"])

            self._plotter.plot_eval(rewards, successes, save=save_path, show=show)
    
            self._env.close()
