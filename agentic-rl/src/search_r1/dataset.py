import torch
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from datasets import Dataset
from typing import List
from dataclasses import dataclass
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


@dataclass(frozen=True)
class Message:
    role: str
    content: str

@dataclass(frozen=True)
class NqAndHotpotqaSample:
    id: str
    question: str
    golden_answers: tuple[str, ...]
    data_source: str
    messages: tuple[Message, ...]

    @classmethod
    def from_row(cls, row: dict) -> "NqAndHotpotqaSample":
        return cls(
            id=row["id"],
            question=row["question"],
            golden_answers=tuple(row["golden_answers"]),
            data_source=row["data_source"],
            messages=tuple(
                Message(role=m["role"], content=m["content"])
                for m in row["prompt"]
            ),
        )


class NqAndHotpotqaDataset(object):
    def __init__(self, batch_size: int = 4, num_workers: int = 0):
        data_dir = snapshot_download(
            repo_id="PeterJinGo/nq_hotpotqa_train",
            repo_type="dataset",
            allow_patterns=["*.parquet"],
        )

        train_table = pq.read_table(f"{data_dir}/train.parquet")
        test_table = pq.read_table(f"{data_dir}/test.parquet")
        raw_train_dataset = Dataset(train_table)
        raw_test_dataset = Dataset(test_table)

        self._train_data_loader: torch.utils.data.DataLoader[list[NqAndHotpotqaSample]] = self._build_data_loader(
            dataset=raw_train_dataset, 
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
        self._test_data_loader: torch.utils.data.DataLoader[list[NqAndHotpotqaSample]] = self._build_data_loader(
            dataset=raw_test_dataset, 
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

    def _build_data_loader(
            self, 
            dataset: Dataset,
            batch_size: int,
            num_workers: int,
            shuffle: bool,
        ) -> torch.utils.data.DataLoader[list[NqAndHotpotqaSample]]:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            drop_last=False,
        )

    @staticmethod
    def _collate_fn(batch: List[dict]) -> List[dict]:
        return [NqAndHotpotqaSample.from_row(row) for row in batch]

    def train_data_loader(self) -> torch.utils.data.DataLoader:
        return self._train_data_loader

    def test_data_loader(self) -> torch.utils.data.DataLoader:
        return self._test_data_loader

if __name__ == "__main__":
    dataset = NqAndHotpotqaDataset()
    train_dataloader = dataset.train_data_loader()
    for batch in train_dataloader:
        print(batch)
        break