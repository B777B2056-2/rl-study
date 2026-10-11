from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, Any
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class BaseSample:
    """所有任务样本的最小契约。Trainer 只依赖这两个字段。"""
    id: str
    prompt: str


class BaseDataset(ABC):
    """
    所有任务数据集的基类。
    
    子类负责：
        - 加载原始数据（自己写）
        - 定义 collate_fn（把 raw row 转成 Sample）
        - 实现 train_data_loader / test_data_loader
    
    基类提供：
        - batch_size / num_workers 存储
        - _build_data_loader 通用构造逻辑
    """
    
    def __init__(self, batch_size: int = 4, num_workers: int = 0):
        super().__init__()
        self._batch_size = batch_size
        self._num_workers = num_workers
    
    # ── 子类必须实现 ──
    @abstractmethod
    def train_data_loader(self) -> DataLoader: ...
    
    @abstractmethod
    def test_data_loader(self) -> DataLoader: ...
    
    # ── 基类通用实现 ──
    def _build_data_loader(
        self,
        dataset: Any,
        shuffle: bool,
        collate_fn: Callable,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
        )