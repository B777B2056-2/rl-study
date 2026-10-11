"""
本地 Plotter：不依赖 llm_rl。

- log():  先用 tqdm.write 打印到控制台，再把 (step, value) 追加写入持久化存储。
- plot(): 从持久化存储加载全部指标到内存，绘图后清空存储。

持久化存储采用 JSONL（每行一条 {name, step, value}），方便追加写入。
"""
import json
import os
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm


class Plotter:
    """指标记录 + 绘图器。"""

    def __init__(self, storage_path: str = "metrics.jsonl"):
        self._storage_path = storage_path

    # ============================================================
    # 记录
    # ============================================================

    def log(self, name: str, value, step: Optional[int] = None) -> None:
        """打印一条指标并追加落盘。"""
        if step is None:
            step = len(self._load().get(name, [])) + 1

        value = float(value)
        step = int(step)

        # 1. 先打印到控制台
        tqdm.write(f"[metrics] {name}: {value:.6f} (step={step})")

        # 2. 追加写入持久化存储
        self._append(name, step, value)

    # ============================================================
    # 绘图
    # ============================================================

    def plot(
        self,
        names: Optional[List[str]] = None,
        save: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """加载全部指标 → 绘图 → 清空存储。"""
        data = self._load()  # {name: [(step, value), ...]}
        names = names or list(data.keys())
        names = [n for n in names if data.get(n)]

        if not names:
            tqdm.write("[Plotter] 没有数据可以画")
            return

        self._draw(data, names, save=save, show=show)
        self._clear()

    # ============================================================
    # 持久化存储
    # ============================================================

    def _append(self, name: str, step: int, value: float) -> None:
        parent = os.path.dirname(self._storage_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self._storage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"name": name, "step": step, "value": value}) + "\n")

    def _load(self) -> Dict[str, List[Tuple[int, float]]]:
        data: Dict[str, List[Tuple[int, float]]] = {}
        if not os.path.isfile(self._storage_path):
            return data
        with open(self._storage_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                data.setdefault(rec["name"], []).append((rec["step"], rec["value"]))
        return data

    def _clear(self) -> None:
        if os.path.isfile(self._storage_path):
            open(self._storage_path, "w", encoding="utf-8").close()

    # ============================================================
    # 绘图实现
    # ============================================================

    def _draw(
        self,
        data: Dict[str, List[Tuple[int, float]]],
        names: List[str],
        save: Optional[str] = None,
        show: bool = True,
    ) -> None:
        import matplotlib.pyplot as plt
        import numpy as np

        n = len(names)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(
            rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False
        )
        axes = axes.flatten()

        for ax, name in zip(axes, names):
            xs = np.array([p[0] for p in data[name]])
            ys = np.array([p[1] for p in data[name]])
            ax.plot(xs, ys, marker="o", linewidth=1.2)
            ax.set_title(name)
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)

        for ax in axes[n:]:
            ax.axis("off")

        plt.tight_layout()

        if save:
            d = os.path.dirname(save)
            if d:
                os.makedirs(d, exist_ok=True)
            plt.savefig(save, dpi=120, bbox_inches="tight")
            tqdm.write(f"[Plotter] 图已保存到 {save}")

        if show:
            plt.show()

        plt.close(fig)

    # ============================================================
    # checkpoint 兼容（CheckpointableTrainer 会调用这两个方法）
    # ============================================================

    def state_dict(self) -> dict:
        return {"metrics": self._load()}

    def load_state_dict(self, state: dict) -> None:
        parent = os.path.dirname(self._storage_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self._storage_path, "w", encoding="utf-8") as f:
            for name, points in state.get("metrics", {}).items():
                for x, y in points:
                    f.write(
                        json.dumps({"name": name, "step": int(x), "value": float(y)})
                        + "\n"
                    )
