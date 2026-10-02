from evaluator import EvalModeMixin


class BaseAgent(EvalModeMixin):
    """所有强化学习算法的统一接口。

    子类必须实现：
        - act(state, deterministic)    : 给定状态返回动作
        - run_episode()              : 训练一个回合，返回指标 dict

    子类可选实现：
        - update_schedule(n_episode)   : 更新 epsilon 等调度参数
    """

    def __init__(self):
        """初始化评估模式计数器（由 EvalModeMixin 提供）。

        子类必须调用 super().__init__()，否则 eval_mode() 无法工作。
        """
        self._init_eval_mode()

    # ============================================================
    # 必须实现
    # ============================================================

    def run_episode(self) -> dict:
        """运行一个回合，返回本回合的指标。

        返回:
            metrics: dict，键是指标名，值是标量。例如:
                {
                    "reward":  10.0,   # 本回合总奖励
                    "success": 1.0,    # 是否成功
                    "epsilon": 0.15,   # 当前探索率
                    "loss":    0.032,  # 平均损失（DQN/PPO）
                }

        Trainer/Evaluator 会把 metrics 里的每个键自动记录到 Plotter，
        命名为 "train_<key>/eval_<key>"，例如 "train_reward"。

        子类实现要点:
            1. 调用 env.reset() 开始新回合；
            2. 循环 act / env.step 直到 done；
            3. 用 self.act(state) 选动作（不要绕过 act）；
            4. 更新 Q 表或网络（评估模式下应自动跳过，由 is_evaluating 控制）；
            5. 返回包含至少 "reward" 的 dict。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 必须实现 run_episode() 方法"
        )

    # ============================================================
    # 可选实现
    # ============================================================

    def update_schedule(self, n_episode: int):
        """更新随训练进度变化的超参数。

        参数:
            n_episode: 当前回合编号，从 0 开始。

        典型用途:
            - 衰减 epsilon（ε-贪婪）
            - 衰减学习率
            - 更新温度参数（SAC）
            - 调整 clip 范围（PPO）

        子类实现示例:
            def update_schedule(self, n_episode):
                self._epsilon = max(
                    self._min_eps,
                    self._eps0 * self._eps_decay ** n_episode
                )

        默认是空操作，适合不需要调度的算法（如 PPO）。
        """
        pass
