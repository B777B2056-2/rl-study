"""
Search-R1 数据集实现（NQ + HotpotQA）。
"""
import os
from dataclasses import dataclass
from typing import List

import pyarrow.parquet as pq
import torch
from datasets import Dataset
from huggingface_hub import snapshot_download

from src.infra.dataset.dataset import BaseDataset, BaseSample

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class NqAndHotpotqaSample(BaseSample):
    """
    NQ + HotpotQA 样本。

    继承 BaseSample 的 id / prompt，额外携带：
        - data_source:   数据集来源（"nq" / "hotpotqa"）
        - messages:      保留 role 结构，方便走 chat template
    """
    data_source: str
    messages: tuple[Message, ...]

    @classmethod
    def from_row(cls, row: dict) -> "NqAndHotpotqaSample":
        messages = tuple(
            Message(role=m["role"], content=m["content"])
            for m in row["prompt"]
        )

        # prompt 取最后一条 user message 的 content
        prompt = ""
        for msg in reversed(messages):
            if msg.role == "user":
                prompt = msg.content
                break

        return cls(
            id=row["id"],
            system_prompt=None,
            user_prompt=prompt,
            outputs=tuple(row["golden_answers"]),
            data_source=row["data_source"],
            messages=messages,
        )


class NqAndHotpotqaDataset(BaseDataset):
    """NQ + HotpotQA 数据集。"""

    REPO_ID = "PeterJinGo/nq_hotpotqa_train"

    def __init__(self, batch_size: int = 4, num_workers: int = 0):
        super().__init__(batch_size, num_workers)

        data_dir = snapshot_download(
            repo_id=self.REPO_ID,
            repo_type="dataset",
            allow_patterns=["*.parquet"],
        )
        raw_train = Dataset(pq.read_table(f"{data_dir}/train.parquet"))
        raw_test = Dataset(pq.read_table(f"{data_dir}/test.parquet"))

        self._train_data_loader = self._build_data_loader(
            raw_train, shuffle=True, collate_fn=self._collate_fn
        )
        self._test_data_loader = self._build_data_loader(
            raw_test, shuffle=False, collate_fn=self._collate_fn
        )

    @staticmethod
    def _collate_fn(batch: List[dict]) -> List[NqAndHotpotqaSample]:
        return [NqAndHotpotqaSample.from_row(row) for row in batch]

    def train_data_loader(self) -> torch.utils.data.DataLoader:
        return self._train_data_loader

    def test_data_loader(self) -> torch.utils.data.DataLoader:
        return self._test_data_loader