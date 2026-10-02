from plotter import Plotter


class Trainer:
    """只负责训练：驱动训练循环 + 记录指标 + 训练完立刻出图。"""

    def __init__(self, agent, train_env, seed: int, n_episodes,
                 plotter=None, verbose=True):
        train_env.reset(seed=seed)
        train_env.action_space.seed(seed)
        
        self._agent = agent
        self._train_env = train_env
        self._n_episodes = n_episodes
        self._plotter = plotter or Plotter(window=100)
        self._verbose = verbose

    def train(self, save_path=None, action_names=None, show=True):
        recent = []

        for ep in range(self._n_episodes):
            self._agent.update_schedule(ep)
            metrics = self._agent.run_episode()

            # 记录训练指标
            for k, v in metrics.items():
                self._plotter.log(f"train_{k}", v, step=ep + 1)

            # 打印进度
            recent.append(metrics.get("reward", 0.0))
            if len(recent) > 100:
                recent.pop(0)
            if self._verbose and (ep + 1) % 100 == 0:
                print(f"ep={ep+1:6d} | reward(100)={sum(recent)/len(recent):.3f}")

        self._train_env.close()

        # 训练完立刻出图
        q_table = self._agent.q_table() if hasattr(self._agent, "q_table") else None
        self._plotter.plot(
            q_table=q_table,
            action_names=action_names,
            save=save_path,
            show=show,
        )