"""
ToolRL 数据集实现。
"""
import os
import uuid
from dataclasses import dataclass
from typing import List

import torch
from datasets import load_dataset

from src.infra.dataset.dataset import BaseDataset, BaseSample
from src.tool_rl.tools import ToolRegistryBuilder, ToolRLRouter

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


@dataclass(frozen=True)
class ToolRLSample(BaseSample):
    """
    ToolRL 样本。
    """

    @classmethod
    def from_row(cls, row: dict) -> "ToolRLSample":
        system_prompt = row["instruction"]
        user_prompt = row["input"]
        outputs = row["output"]

        return cls(
            id=str(uuid.uuid4()),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            outputs=outputs,
        )


class ToolRLDataset(BaseDataset):
    """ToolRL 数据集。"""

    REPO_ID = "emrecanacikgoz/ToolRL"

    def __init__(self, batch_size: int = 4, num_workers: int = 0):
        super().__init__(batch_size, num_workers)

        raw_dataset = load_dataset(self.REPO_ID)
        raw_train_dataset = raw_dataset['train']

        self._tool_registry_builder = ToolRegistryBuilder()
        self._tool_registry_builder.add_tools_from_dataset(raw_dataset)

        split = raw_train_dataset.train_test_split(test_size=0.1, seed=42)
        raw_train_dataset = split['train']
        raw_test_dataset  = split['test']

        self._train_data_loader = self._build_data_loader(
            raw_train_dataset, shuffle=True, collate_fn=self._collate_fn
        )
        self._test_data_loader = self._build_data_loader(
            raw_test_dataset, shuffle=False, collate_fn=self._collate_fn
        )

    @staticmethod
    def _collate_fn(batch: List[dict]) -> List[ToolRLSample]:
        return [ToolRLSample.from_row(row) for row in batch]

    def train_data_loader(self) -> torch.utils.data.DataLoader:
        return self._train_data_loader

    def test_data_loader(self) -> torch.utils.data.DataLoader:
        return self._test_data_loader

    def build_tool_router(self) -> ToolRLRouter:
        return self._tool_registry_builder.build_router()
    