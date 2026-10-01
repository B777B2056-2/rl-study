from plotter import Plotter
from evaluator import Evaluator


class Trainer:
    """训练编排：内部创建 Plotter / Evaluator，驱动训练循环，定期评估，训练完出图。"""

    def __init__(self, agent, eval_env, n_episodes,
                 eval_interval=500, eval_episodes=100, eval_seed=1000,
                 success_fn=None, plotter=None, verbose=True):
        self._agent = agent
        self._eval_env = eval_env
        self._n_episodes = n_episodes
        self._eval_interval = eval_interval
        self._verbose = verbose

        self._plotter = plotter or Plotter(window=100)
        self._evaluator = Evaluator(
            eval_env, n_episodes=eval_episodes,
            seed=eval_seed, success_fn=success_fn,
        )

    def train(self, plot=True, save_path=None, action_names=None):
        recent = []

        for ep in range(self._n_episodes):
            # 1. 更新调度
            self._agent.update_schedule(ep)

            # 2. 训练一个回合
            metrics = self._agent.train_episode()

            # 3. 记录训练指标
            for k, v in metrics.items():
                self._plotter.log(f"train_{k}", v)

            # 4. 打印进度
            recent.append(metrics.get("reward", 0.0))
            if len(recent) > 100:
                recent.pop(0)
            if self._verbose and (ep + 1) % 100 == 0:
                print(f"ep={ep+1:6d} | reward(100)={sum(recent)/len(recent):.3f}")

            # 5. 定期评估
            if (ep + 1) % self._eval_interval == 0:
                result = self._evaluator.evaluate(self._agent)
                for k, v in result.items():
                    self._plotter.log(f"eval_{k}", v)
                if self._verbose:
                    print(f"  [Eval] ep={ep+1} | "
                          f"reward={result['mean_reward']:.3f} | "
                          f"success={result['success_rate']:.2%}")

        # 6. 出图
        if plot:
            q_table = self._agent.q_table() if hasattr(self._agent, "q_table") else None
            self._plotter.plot(
                q_table=q_table,
                action_names=action_names,
                save=save_path,
            )

        self._eval_env.close()