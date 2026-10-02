import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, window=100):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        self._data = {}          # name -> list of (x, y)
        self._window = window

    def log(self, name, value, step=None):
        """step 是 x 坐标；不传则自增"""
        if step is None:
            step = len(self._data.get(name, [])) + 1
        self._data.setdefault(name, []).append((int(step), float(value)))

    def _ma(self, y):
        w = self._window
        if w <= 0 or len(y) < w:
            return None
        return np.convolve(y, np.ones(w) / w, mode="valid")

    def plot(self, names=None, q_table=None, action_names=None,
             save=None, show=True):
        names = names or list(self._data.keys())
        names = [n for n in names if len(self._data.get(n, [])) > 0]

        n_curves = len(names)
        has_q = q_table is not None
        n_plots = n_curves + (1 if has_q else 0)
        if n_plots == 0:
            print("[Plotter] 没有数据")
            return

        cols = min(n_plots, 3)
        rows = (n_plots + cols - 1) // cols
        fig = plt.figure(figsize=(5 * cols, 3.5 * rows))
        gs = fig.add_gridspec(rows, cols)

        for i, name in enumerate(names):
            r, c = divmod(i, cols)
            ax = fig.add_subplot(gs[r, c])

            xy = self._data[name]
            xs = np.array([p[0] for p in xy])
            ys = np.array([p[1] for p in xy])

            ax.plot(xs, ys, alpha=0.25, color="gray", linewidth=0.8)

            ma = self._ma(ys)
            if ma is not None:
                x_ma = xs[len(xs) - len(ma):]
                ax.plot(x_ma, ma, color="C0", linewidth=1.8)
                ax.set_title(f"{name} (MA {self._window})")
            else:
                ax.set_title(name)

            ax.set_xlabel("Episode")
            ax.grid(alpha=0.3)

        # 画 Q 表（不变）
        if has_q:
            i = n_curves
            r, c = divmod(i, cols)
            span = min(2, cols - c)
            ax = fig.add_subplot(gs[r, c:c + span])
            q = np.asarray(q_table)
            n_states, n_actions = q.shape
            if action_names is None:
                action_names = [f"a{j}" for j in range(n_actions)]
            im = ax.imshow(q, cmap="coolwarm", aspect="auto")
            ax.set_xticks(range(n_actions))
            ax.set_xticklabels(action_names)
            ax.set_yticks(range(n_states))
            ax.set_yticklabels([f"s{k}" for k in range(n_states)])
            ax.set_title("Q Table")
            for a in range(n_states):
                for b in range(n_actions):
                    ax.text(b, a, f"{q[a, b]:.3f}",
                            ha="center", va="center", fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=120, bbox_inches="tight")
            print(f"[Plotter] 已保存到 {save}")
        if show:
            plt.show()
        plt.close(fig)

    def plot_eval(self, rewards, successes, save=None, show=True):
        """画评估结果：每回合 reward 曲线 + 统计柱状图"""
        rewards = np.array(rewards)
        successes = np.array(successes)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # 左：每回合 reward
        ax = axes[0]
        ax.plot(rewards, alpha=0.4, color="gray", label="per episode")
        ax.axhline(rewards.mean(), color="C0", linewidth=1.8,
                label=f"mean={rewards.mean():.3f}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Eval Reward")
        ax.grid(alpha=0.3)
        ax.legend()

        # 右：统计柱状图
        ax = axes[1]
        labels = ["mean", "std", "min", "max", "success"]
        values = [rewards.mean(), rewards.std(),
                rewards.min(), rewards.max(), successes.mean()]
        colors = ["C0", "C1", "C2", "C3", "C4"]
        bars = ax.bar(labels, values, color=colors, alpha=0.7)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(), f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title("Eval Metrics")
        ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=120, bbox_inches="tight")
            print(f"[Plotter] 已保存到 {save}")
        if show:
            plt.show()
        plt.close(fig)