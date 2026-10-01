# plotting/plotter.py
import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, window: int = 100):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        self._data = {}
        self._window = window

    def log(self, name, value):
        self._data.setdefault(name, []).append(float(value))

    def _ma(self, x):
        w = self._window
        if w <= 0 or len(x) < w:
            return None
        return np.convolve(x, np.ones(w) / w, mode="valid")

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
            data = np.array(self._data[name])
            ax.plot(data, alpha=0.25, color="gray", linewidth=0.8)
            ma = self._ma(data)
            if ma is not None:
                x = np.arange(len(data) - len(ma), len(data))
                ax.plot(x, ma, color="C0", linewidth=1.8)
                ax.set_title(f"{name} (MA {self._window})")
            else:
                ax.set_title(name)
            ax.set_xlabel("Episode")
            ax.grid(alpha=0.3)

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
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=120, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)