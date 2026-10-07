"""
Plotter：通用绘图器
- 只接收 log(name, value)，不关心谁调它
- 训练完统一 plot() 出图
"""
import os
import numpy as np
import matplotlib.pyplot as plt


class Plotter(object):
    """通用 RL/SFT 训练曲线绘图器"""

    def __init__(self, window: int = 100):
        """
        window: 滑动平均窗口大小。设为 0 则不画平滑曲线。
        """
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        self._data = {}          # name -> list of (x, y)
        self._window = window

    # ============ 数据记录 ============

    def log(self, name: str, value, step: int = None):
        """记录一个标量。step 是 x 坐标；不传则自增"""
        if step is None:
            step = len(self._data.get(name, [])) + 1
        self._data.setdefault(name, []).append((int(step), float(value)))

    def log_dict(self, data: dict, step: int = None):
        """批量记录"""
        for k, v in data.items():
            self.log(k, v, step=step)

    def history(self, name: str) -> list:
        """取出某条曲线的全部数据"""
        return self._data.get(name, [])

    def keys(self):
        """所有已记录的曲线名"""
        return list(self._data.keys())

    def clear(self):
        self._data.clear()

    # ============ 内部工具 ============

    def _moving_average(self, y):
        w = self._window
        if w <= 0 or len(y) < w:
            return None
        return np.convolve(y, np.ones(w) / w, mode="valid")

    # ============ 绘图 ============

    def plot(self, names=None, save=None, show=True):
        """
        names: 要画的指标名列表，None 表示画全部
        save:  图片保存路径
        show:  是否弹窗
        """
        names = names or list(self._data.keys())
        names = [n for n in names if len(self._data.get(n, [])) > 0]

        if not names:
            print("[Plotter] 没有数据可以画")
            return

        n = len(names)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(
            rows, cols,
            figsize=(5 * cols, 3.5 * rows),
            squeeze=False,
        )
        axes = axes.flatten()

        for ax, name in zip(axes, names):
            xy = self._data[name]
            xs = np.array([p[0] for p in xy])
            ys = np.array([p[1] for p in xy])

            # 原始曲线（淡灰色）
            ax.plot(xs, ys, alpha=0.25, color="gray", linewidth=0.8)

            # 滑动平均曲线
            ma = self._moving_average(ys)
            if ma is not None:
                x_ma = xs[len(xs) - len(ma):]
                ax.plot(x_ma, ma, color="C0", linewidth=1.8)
                ax.set_title(f"{name}  (MA {self._window})")
            else:
                ax.set_title(name)

            # 统计信息
            stats = (f"mean={ys.mean():.4f}\n"
                     f"std={ys.std():.4f}\n"
                     f"max={ys.max():.4f}\n"
                     f"min={ys.min():.4f}")
            ax.text(0.98, 0.98, stats,
                    transform=ax.transAxes,
                    ha="right", va="top",
                    fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

            ax.set_xlabel("Step")
            ax.grid(alpha=0.3)

        # 隐藏多余子图
        for ax in axes[n:]:
            ax.axis("off")

        plt.tight_layout()

        if save:
            d = os.path.dirname(save)
            if d:
                os.makedirs(d, exist_ok=True)
            plt.savefig(save, dpi=120, bbox_inches="tight")
            print(f"[Plotter] 图已保存到 {save}")

        if show:
            plt.show()

        plt.close(fig)

    def state_dict(self) -> dict:
        return {
            "data": {
                k: [(int(x), float(y)) for x, y in v]
                for k, v in self._data.items()
            },
            "window": self._window,
        }

    def load_state_dict(self, state: dict):
        self._data = {
            k: [(int(x), float(y)) for x, y in v]
            for k, v in state["data"].items()
        }
        if "window" in state:
            self._window = state["window"]