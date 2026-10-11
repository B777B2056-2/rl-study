import os
import torch
import random
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class CkptConfig:
    ckpt_dir: str
    save_steps: int = 500
    keep_last: int = 3
    prefix: str = "checkpoint"

class CheckpointManager:
    """通用 Checkpoint 管理器（纯 IO，不感知 Trainer）"""

    def __init__(self, config: CkptConfig):
        self._save_dir = config.ckpt_dir
        self._save_steps = config.save_steps
        self._keep_last = config.keep_last
        self._prefix = config.prefix
        os.makedirs(config.ckpt_dir, exist_ok=True)

    def should_save(self, step: int) -> bool:
        return step > 0 and step % self._save_steps == 0

    def save(self, step: int, state: dict):
        path = os.path.join(self._save_dir, f"{self._prefix}-{step}.pt")
        torch.save({**state, "_step": step}, path)
        tqdm.write(f"[Ckpt] 已保存 {path}")
        self._cleanup()

    def load(self, path: str = None, device="cpu") -> dict:
        if path is None:
            path = self.latest_path()
            if path is None:
                raise FileNotFoundError("没有 checkpoint")
            print(f"[Ckpt] 加载最新 {path}")
        return torch.load(path, map_location=device, weights_only=False)

    def latest_path(self):
        paths = self._list()
        return paths[-1] if paths else None

    def has_checkpoint(self) -> bool:
        return len(self._list()) > 0

    def _list(self):
        files = [
            f for f in os.listdir(self._save_dir)
            if f.startswith(self._prefix) and f.endswith(".pt")
        ]
        files.sort(key=lambda f: int(f[len(self._prefix)+1:-3]))
        return [os.path.join(self._save_dir, f) for f in files]

    def _cleanup(self):
        paths = self._list()
        for old in paths[:-self._keep_last]:
            os.remove(old)
            tqdm.write(f"[Ckpt] 删除旧 {old}")


class CheckpointableTrainer:
    """
    所有 Trainer 的基类

    基类管:
        - _global_step / _epoch
        - _plotter（可选）
        - state_dict / load_state_dict 的包装
        - checkpoint 保存和恢复

    子类实现:
        - _state_dict() -> dict
        - _load_state_dict(state)
    """

    def __init__(self, ckpt_manager: CheckpointManager = None,
                 plotter=None):
        self._ckpt = ckpt_manager
        self._global_step = 0
        self._epoch = 0
        self._plotter = plotter

    # ============ 对外接口 ============

    def state_dict(self) -> dict:
        state = {
            "epoch": self._epoch,
            "global_step": self._global_step,
            **self._state_dict(),
        }
        if self._plotter is not None:
            state["_plotter"] = self._plotter.state_dict()
        return state

    def load_state_dict(self, state: dict):
        self._epoch = state["epoch"]
        self._global_step = state["global_step"]
        self._load_state_dict(state)
        if self._plotter is not None and "_plotter" in state:
            self._plotter.load_state_dict(state["_plotter"])

    # ============ 子类实现 ============

    def _state_dict(self) -> dict:
        raise NotImplementedError

    def _load_state_dict(self, state: dict):
        raise NotImplementedError

    # ============ 流程控制 ============

    def maybe_resume(self) -> bool:
        if self._ckpt is None or not self._ckpt.has_checkpoint():
            print("[Trainer] 无 checkpoint，从头开始")
            return False
        state = self._ckpt.load(device=self._get_device())
        self.load_state_dict(state)
        print(f"[Trainer] 已恢复: epoch={self._epoch}, step={self._global_step}")
        return True

    def maybe_save(self):
        self._global_step += 1
        if self._ckpt and self._ckpt.should_save(self._global_step):
            self._ckpt.save(self._global_step, self.state_dict())

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    @property
    def global_step(self):
        return self._global_step

    @property
    def epoch(self):
        return self._epoch

    def _get_device(self):
        """子类可覆盖"""
        return "cpu"

# ---- RNG 辅助 ----

def get_rng_state() -> dict:
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict):
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "random" in state:
        random.setstate(state["random"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])